import json
import io
import hashlib
import random
import re
import tempfile
import zipfile
import base64
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from azure.identity import DefaultAzureCredential
except Exception:  # pragma: no cover - handled at runtime when feature is enabled
    DefaultAzureCredential = None
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods, require_POST
from openai import AzureOpenAI

from .forms import ProfileForm, SignUpForm
from .models import ChatMessage, ChatSession, GeneratedProject, PublishJob, UserProfile

MAX_FILES_PER_PROJECT = 40
GITHUB_API_BASE = "https://api.github.com"


def home_page(request):
    return render(request, "chatbot/home.html")


def signup_page(request):
    if request.user.is_authenticated:
        return redirect("chat_page")

    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("chat_page")
    else:
        form = SignUpForm()
    return render(request, "chatbot/signup.html", {"form": form})


@login_required
def profile_page(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    if request.method == "POST":
        form = ProfileForm(request.POST, instance=profile, user=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully.")
            return redirect("profile_page")
    else:
        form = ProfileForm(instance=profile, user=request.user)

    generated_items = GeneratedProject.objects.filter(user=request.user).select_related("session")[:50]
    return render(
        request,
        "chatbot/profile.html",
        {
            "form": form,
            "generated_items": generated_items,
        },
    )


@login_required
def user_settings_page(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    if request.method == "POST":
        mode = str(request.POST.get("deployment_mode") or "").strip()
        valid_modes = {
            UserProfile.DEPLOYMENT_MODE_VSCODE,
            UserProfile.DEPLOYMENT_MODE_AZURE,
        }
        if mode not in valid_modes:
            messages.error(request, "Invalid deployment mode.")
            return redirect("user_settings_page")

        profile.deployment_mode = mode
        profile.deployment_mode_prompt_seen = True
        profile.save(update_fields=["deployment_mode", "deployment_mode_prompt_seen", "updated_at"])
        if mode == UserProfile.DEPLOYMENT_MODE_AZURE:
            messages.info(request, "Azure App Service mode is coming soon..")
        else:
            messages.success(request, "Deployment mode updated to VS Code.")
        return redirect("user_settings_page")

    return render(
        request,
        "chatbot/settings.html",
        {
            "deployment_mode": profile.deployment_mode,
        },
    )


@login_required
@require_http_methods(["POST"])
def set_deployment_mode(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    mode = str(payload.get("deployment_mode") or "").strip()
    valid_modes = {
        UserProfile.DEPLOYMENT_MODE_VSCODE,
        UserProfile.DEPLOYMENT_MODE_AZURE,
    }
    if mode not in valid_modes:
        return JsonResponse({"error": "Invalid deployment mode."}, status=400)

    profile.deployment_mode = mode
    profile.deployment_mode_prompt_seen = True
    profile.save(update_fields=["deployment_mode", "deployment_mode_prompt_seen", "updated_at"])

    message = "Deployment mode saved."
    if mode == UserProfile.DEPLOYMENT_MODE_AZURE:
        message = "coming soon.."
    return JsonResponse({"ok": True, "deployment_mode": mode, "message": message})


@login_required
def chat_page(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    sessions = list(
        ChatSession.objects.filter(user=request.user).prefetch_related("messages", "generated_projects")
    )
    for session in sessions:
        latest_project = next(iter(session.generated_projects.all()), None)
        session.latest_project_id = latest_project.id if latest_project else None
        session.session_download_url = (
            reverse("download_session_project", args=[session.id]) if latest_project else ""
        )
        session.session_test_url = reverse("session_test_ide", args=[session.id]) if latest_project else ""
    return render(
        request,
        "chatbot/chat.html",
        {
            "sessions": sessions,
            "show_deployment_mode_prompt": not profile.deployment_mode_prompt_seen,
            "deployment_mode": profile.deployment_mode,
        },
    )


def _missing_azure_settings() -> list[str]:
    required = {
        "AZURE_OPENAI_ENDPOINT": settings.AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": settings.AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_API_VERSION": settings.AZURE_OPENAI_API_VERSION,
        "AZURE_OPENAI_DEPLOYMENT": settings.AZURE_OPENAI_DEPLOYMENT,
    }
    return [name for name, value in required.items() if not value]


def _build_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
    )


def _session_title_from_message(message: str) -> str:
    clean = " ".join(message.split())
    return clean[:60] or "New code task"


def _extract_json_object(raw_text: str) -> dict:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw_text)
        if not match:
            raise
        return json.loads(match.group(0))


def _safe_relative_path(raw_path: str) -> Path:
    candidate = Path(raw_path.replace("\\", "/")).as_posix().strip("/")
    if not candidate:
        raise ValueError("Empty file path in generated output.")
    if ".." in candidate.split("/"):
        raise ValueError("Unsafe file path in generated output.")
    return Path(candidate)


def _normalize_generated_project(data: dict) -> tuple[str, list[dict]]:
    project_name = str(data.get("project_name") or "generated-project").strip()
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", project_name).strip("-").lower() or "generated-project"

    files = data.get("files", [])
    if not isinstance(files, list) or not files:
        raise ValueError("No files were generated.")
    if len(files) > MAX_FILES_PER_PROJECT:
        raise ValueError(f"Too many files generated. Limit is {MAX_FILES_PER_PROJECT}.")

    normalized_files = []
    for item in files:
        if not isinstance(item, dict):
            continue
        rel_path = _safe_relative_path(str(item.get("path", "")).strip())
        normalized_files.append(
            {
                "path": rel_path.as_posix(),
                "content": str(item.get("content", "")),
            }
        )

    if not normalized_files:
        raise ValueError("Generated output did not include valid file entries.")
    return safe_name, normalized_files


def _build_project_files(user_prompt: str, conversation_messages: list[dict]) -> dict:
    client = _build_client()
    completion = client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a code generator that must return valid JSON only. "
                    "Generate a project structure from the user request.\n"
                    "Output schema:\n"
                    "{\n"
                    '  "project_name": "short-kebab-name",\n'
                    '  "files": [\n'
                    '    {"path": "relative/path.ext", "content": "file content"}\n'
                    "  ]\n"
                    "}\n"
                    "Rules: no markdown, no explanations, no absolute paths, "
                    "no '..' path segments."
                ),
            },
            *conversation_messages,
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = completion.choices[0].message.content if completion.choices else "{}"
    return _extract_json_object(content or "{}")


def _is_project_generation_request(message: str) -> bool:
    text = message.lower().strip()
    if not text:
        return False

    generation_verbs = (
        "generate",
        "create",
        "build",
        "make",
        "scaffold",
        "develop",
        "write code",
        "implement",
    )
    target_words = (
        "project",
        "app",
        "api",
        "website",
        "code",
        "script",
        "service",
        "backend",
        "frontend",
    )
    return any(v in text for v in generation_verbs) and any(t in text for t in target_words)


def _is_download_intent(message: str) -> bool:
    text = message.lower().strip()
    if not text:
        return False
    keywords = (
        "download",
        "zip",
        "export",
        "push github",
        "push to github",
        "github push",
        "publish",
        "deploy",
    )
    return any(word in text for word in keywords)


def _is_project_update_request(message: str) -> bool:
    text = message.lower().strip()
    if not text:
        return False

    update_verbs = (
        "add",
        "modify",
        "update",
        "change",
        "edit",
        "delete",
        "remove",
        "rename",
        "replace",
    )
    file_markers = (
        "file",
        "folder",
        "path",
        ".py",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".json",
        ".md",
        ".java",
        ".go",
        ".php",
        ".cs",
    )
    return any(v in text for v in update_verbs) and any(marker in text for marker in file_markers)


def _build_updated_project_files(
    user_prompt: str,
    conversation_messages: list[dict],
    latest_project: GeneratedProject,
) -> dict:
    client = _build_client()
    existing_files_json = json.dumps(latest_project.files, ensure_ascii=False)
    completion = client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a codebase update engine. "
                    "You will receive an existing virtual project and a requested change. "
                    "Return valid JSON only with the full updated project state.\n"
                    "Output schema:\n"
                    "{\n"
                    '  "project_name": "short-kebab-name",\n'
                    '  "files": [\n'
                    '    {"path": "relative/path.ext", "content": "file content"}\n'
                    "  ]\n"
                    "}\n"
                    "Rules:\n"
                    "- Apply user request exactly.\n"
                    "- For delete/remove requests, omit those files from final files list.\n"
                    "- For add/create requests, include new files.\n"
                    "- For modify/update requests, update file content.\n"
                    "- Keep unrelated existing files unchanged.\n"
                    "- No markdown, no explanations, no absolute paths, no '..' path segments."
                ),
            },
            {
                "role": "system",
                "content": (
                    f"Existing project name: {latest_project.project_name}\n"
                    f"Existing files JSON: {existing_files_json}"
                ),
            },
            *conversation_messages,
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.15,
    )
    content = completion.choices[0].message.content if completion.choices else "{}"
    return _extract_json_object(content or "{}")


def _build_conversational_reply(
    user_prompt: str,
    conversation_messages: list[dict],
    latest_project: GeneratedProject | None,
) -> str:
    project_context = "No generated project exists yet in this session."
    if latest_project:
        file_paths = [str(item.get("path", "")) for item in latest_project.files[:30]]
        project_context = (
            f"Latest generated project name: {latest_project.project_name}\n"
            f"File count: {len(latest_project.files)}\n"
            f"Files: {', '.join(file_paths)}"
        )

    client = _build_client()
    completion = client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an interactive coding copilot. "
                    "Chat naturally like human-to-human. "
                    "If user greets, greet back. "
                    "If user asks technology used or expected output, explain clearly using context. "
                    "Keep answers concise but helpful."
                ),
            },
            {
                "role": "system",
                "content": f"Session project context:\n{project_context}",
            },
            *conversation_messages,
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )
    return completion.choices[0].message.content if completion.choices else ""


def _infer_stack(files: list[dict]) -> str:
    paths = [str(item.get("path", "")).lower() for item in files]
    if any(path.endswith(".py") for path in paths):
        return "Python"
    if any(path.endswith(".js") or path.endswith(".ts") for path in paths):
        return "JavaScript/TypeScript"
    if any(path.endswith(".java") for path in paths):
        return "Java"
    if any(path.endswith(".cs") for path in paths):
        return "C#"
    if any(path.endswith(".go") for path in paths):
        return "Go"
    return "Mixed"


def _dynamic_user_reply(username: str, user_prompt: str, project_name: str, files: list[dict]) -> str:
    count = len(files)
    stack = _infer_stack(files)
    sample = [item.get("path", "") for item in files[:3]]
    sample_text = ", ".join(sample) if sample else "no files"
    if count > 3:
        sample_text = f"{sample_text}, +{count - 3} more"

    tone_seed = int(hashlib.sha256(f"{username}:{project_name}:{count}:{user_prompt}".encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(tone_seed)

    openers = [
        f"{username}, nice request.",
        f"Great, {username}.",
        f"Done, {username}.",
        f"Awesome direction, {username}.",
        f"{username}, I finished your build.",
    ]
    summaries = [
        f"I generated `{project_name}` with {count} file(s) in {stack}.",
        f"Your project `{project_name}` is ready with {count} {stack} file(s).",
        f"`{project_name}` is now created with {count} file(s), targeting {stack}.",
    ]
    previews = [
        f"Quick preview: {sample_text}.",
        f"Here are key files: {sample_text}.",
        f"Starter files include: {sample_text}.",
    ]
    next_steps = [
        "Want me to optimize folder structure next?",
        "If you want, I can add tests in the next step.",
        "You can run it in Test now, and I can patch errors from output.",
        "Need auth, DB, or API enhancements next? I can generate that too.",
    ]

    return " ".join(
        [
            rng.choice(openers),
            rng.choice(summaries),
            rng.choice(previews),
            rng.choice(next_steps),
        ]
    )


@login_required
def download_project(request, project_id: int):
    project = get_object_or_404(GeneratedProject, pk=project_id, user=request.user)
    return _build_zip_response(project)


def _build_zip_bytes(project: GeneratedProject) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in project.files:
            path = str(item.get("path", "")).strip()
            content = str(item.get("content", ""))
            if not path:
                continue
            archive.writestr(path, content)
    return zip_buffer.getvalue()


def _build_zip_response(project: GeneratedProject) -> HttpResponse:
    response = HttpResponse(_build_zip_bytes(project), content_type="application/zip")
    response["Content-Disposition"] = f'attachment; filename="{project.project_name}.zip"'
    return response


def _detect_runtime_from_project(project: GeneratedProject | None) -> str:
    if not project or not isinstance(project.files, list):
        return "PYTHON:3.11"

    paths = [str(item.get("path", "")).strip().lower() for item in project.files if str(item.get("path", "")).strip()]
    if not paths:
        return "PYTHON:3.11"

    if any(path.endswith((".py",)) for path in paths) or "manage.py" in paths:
        return "PYTHON:3.11"
    if any(path.endswith((".js", ".mjs", ".cjs")) for path in paths) or "package.json" in paths:
        return "NODE:20-lts"
    if any(path.endswith((".csproj", ".sln", ".cs")) for path in paths):
        return "DOTNETCORE:8.0"
    if any(path.endswith(".java") for path in paths) or "pom.xml" in paths or "build.gradle" in paths:
        return "JAVA:17-java17"
    if any(path.endswith(".php") for path in paths):
        return "PHP:8.2"
    return "PYTHON:3.11"


def _detect_startup_command_from_project(project: GeneratedProject | None, runtime: str) -> str:
    if not project or not isinstance(project.files, list):
        return ""

    paths = [str(item.get("path", "")).strip().lower() for item in project.files if str(item.get("path", "")).strip()]
    runtime_upper = (runtime or "").strip().upper()

    if runtime_upper.startswith("PYTHON"):
        if "manage.py" in paths:
            return "gunicorn chatbot_project.wsgi --bind=0.0.0.0:$PORT --timeout 600"
        if "app.py" in paths:
            return "gunicorn app:app --bind=0.0.0.0:$PORT --timeout 600"
        if "main.py" in paths:
            return "gunicorn main:app --bind=0.0.0.0:$PORT --timeout 600"
        return "gunicorn app:app --bind=0.0.0.0:$PORT --timeout 600"

    if runtime_upper.startswith("NODE"):
        if "package.json" in paths:
            return "npm start"
        if "server.js" in paths:
            return "node server.js"
        if "app.js" in paths:
            return "node app.js"
        if "main.js" in paths:
            return "node main.js"
        return "npm start"

    return ""


@login_required
def download_session_project(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    project = session.generated_projects.order_by("-created_at").first()
    if not project:
        return JsonResponse({"error": "No generated project for this session yet."}, status=404)
    return _build_zip_response(project)


@login_required
def session_publish_page(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    project = session.generated_projects.order_by("-created_at").first()
    detected_runtime = _detect_runtime_from_project(project)
    return render(
        request,
        "chatbot/publish.html",
        {
            "session": session,
            "project": project,
            "detected_runtime": detected_runtime,
            "detected_startup_command": _detect_startup_command_from_project(project, detected_runtime),
            "download_url": reverse("download_session_project", args=[session.id]) if project else "",
            "test_url": reverse("session_test_ide", args=[session.id]) if project else "",
        },
    )


def _runtime_to_linux_fx_version(runtime: str) -> str:
    raw = (runtime or "").strip().upper()
    if not raw:
        return "PYTHON|3.11"
    if ":" in raw:
        lang, version = raw.split(":", 1)
        return f"{lang}|{version}"
    if "|" in raw:
        return raw
    return f"PYTHON|{raw}"


def _detect_swa_build_properties(project: GeneratedProject | None) -> dict:
    if not project or not isinstance(project.files, list):
        return {
            "appLocation": "/",
            "apiLocation": "",
            "appArtifactLocation": "",
        }

    files = project.files
    raw_paths = [str(item.get("path", "")).strip() for item in files if str(item.get("path", "")).strip()]
    paths = set(raw_paths)

    # If all files are under one top-level folder, treat it as appLocation.
    # This avoids SWA looking at repo root when generated code is nested.
    root_level_files = [p for p in raw_paths if "/" not in p]
    top_dirs = {p.split("/", 1)[0] for p in raw_paths if "/" in p}
    app_root = ""
    if not root_level_files and len(top_dirs) == 1:
        app_root = next(iter(top_dirs))

    app_location = "/" if not app_root else f"/{app_root}"

    def trim_root(path: str) -> str:
        if not app_root:
            return path
        prefix = f"{app_root}/"
        return path[len(prefix) :] if path.startswith(prefix) else path

    trimmed_paths = {trim_root(p) for p in paths}
    package_json = next(
        (
            item
            for item in files
            if trim_root(str(item.get("path", "")).strip()) == "package.json"
        ),
        None,
    )

    app_build_command = ""
    app_artifact_location = ""
    if package_json:
        try:
            package_data = json.loads(str(package_json.get("content", "") or "{}"))
            scripts = package_data.get("scripts") if isinstance(package_data, dict) else {}
            if isinstance(scripts, dict) and "build" in scripts:
                app_build_command = "npm run build"
        except (ValueError, TypeError):
            pass

    if any(path.startswith("dist/") for path in trimmed_paths):
        app_artifact_location = "dist"
    elif any(path.startswith("build/") for path in trimmed_paths):
        app_artifact_location = "build"

    props = {
        "appLocation": app_location,
        "apiLocation": "",
        "appArtifactLocation": app_artifact_location,
    }
    if app_build_command:
        props["appBuildCommand"] = app_build_command
    return props


def _arm_request(
    method: str,
    url: str,
    token: str,
    payload: dict | None = None,
    timeout: int = 300,
) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace").strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with {exc.code}: {error_text[:600]}")


def _run_publish_via_managed_identity(
    project: GeneratedProject,
    job_payload: dict,
    github_token: str = "",
) -> dict:
    if DefaultAzureCredential is None:
        raise RuntimeError("azure-identity is missing. Install dependencies and redeploy.")

    deployment_mode = str(job_payload.get("deployment_mode") or "zip").strip().lower()
    repo_url = str(job_payload.get("repo_url") or "").strip()
    repo_branch = str(job_payload.get("repo_branch") or "").strip() or "main"
    static_web_app_name = str(job_payload.get("static_web_app_name") or "").strip()
    swa_url = str(job_payload.get("swa_url") or "").strip()
    swa_auto_create = bool(job_payload.get("swa_auto_create", False))

    if deployment_mode == "swa_publish":
        if not repo_url:
            return {"ok": False, "error": "repo_url is required for SWA publish mode.", "logs": []}
        logs: list[str] = [
            "SWA publish mode selected.",
            f"Repository source: {repo_url} (branch: {repo_branch})",
        ]
        app_url = swa_url or ""
        if swa_auto_create:
            subscription_id = str(job_payload.get("subscription_id") or "").strip()
            resource_group = str(job_payload.get("resource_group") or "").strip()
            region = str(job_payload.get("region") or "").strip() or "centralindia"
            if not subscription_id or not resource_group or not static_web_app_name:
                return {
                    "ok": False,
                    "error": "For SWA auto-create, provide subscription_id, resource_group, and static_web_app_name.",
                    "logs": logs,
                }
            if not github_token:
                return {
                    "ok": False,
                    "error": (
                        "GitHub token missing in profile. Add GitHub token in Profile to link SWA with your repo."
                    ),
                    "logs": logs,
                }
            credential = DefaultAzureCredential()
            token = credential.get_token("https://management.azure.com/.default").token
            api_version = "2022-09-01"
            base = f"https://management.azure.com/subscriptions/{subscription_id}"
            rg_url = f"{base}/resourcegroups/{resource_group}?api-version=2023-07-01"
            _arm_request("PUT", rg_url, token, {"location": region})
            logs.append(f"Resource group ensured: {resource_group}")

            swa_url_api = (
                f"{base}/resourceGroups/{resource_group}/providers/Microsoft.Web/"
                f"staticSites/{static_web_app_name}?api-version={api_version}"
            )
            swa_payload = {
                "location": region,
                "sku": {"name": "Free", "tier": "Free"},
                "properties": {
                    "repositoryUrl": repo_url,
                    "branch": repo_branch,
                    "provider": "GitHub",
                    "repositoryToken": github_token,
                    "stagingEnvironmentPolicy": "Enabled",
                    "allowConfigFileUpdates": True,
                    "buildProperties": _detect_swa_build_properties(project),
                },
            }
            response = _arm_request("PUT", swa_url_api, token, swa_payload, timeout=900)
            default_hostname = (
                ((response or {}).get("properties") or {}).get("defaultHostname")
                if isinstance(response, dict)
                else ""
            )
            if default_hostname and not app_url:
                app_url = f"https://{default_hostname}"
            logs.append(f"Static Web App ensured: {static_web_app_name}")
            if app_url:
                logs.append(f"SWA endpoint: {app_url}")
            logs.append("SWA linked with GitHub repository using repository token.")
        logs.append("Assuming Azure Static Web Apps is linked to this repository and branch.")
        logs.append("Push new commits to trigger Static Web Apps deployment workflow.")
        if static_web_app_name:
            logs.append(f"Static Web App name: {static_web_app_name}")
        if app_url:
            logs.append(f"Static Web App URL: {app_url}")
        return {
            "ok": True,
            "app_url": app_url or repo_url,
            "logs": logs,
            "message": "SWA publish configured. GitHub push will trigger deployment if repo linkage exists.",
        }

    subscription_id = str(job_payload.get("subscription_id") or "").strip()
    resource_group = str(job_payload.get("resource_group") or "").strip()
    plan_name = str(job_payload.get("app_service_plan") or "").strip()
    webapp_name = str(job_payload.get("webapp_name") or "").strip()
    region = str(job_payload.get("region") or "").strip() or "centralindia"
    runtime = str(job_payload.get("runtime") or "").strip() or "PYTHON:3.11"
    startup_command = str(job_payload.get("startup_command") or "").strip()
    auto_create = bool(job_payload.get("auto_create", False))

    required = {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "app_service_plan": plan_name,
        "webapp_name": webapp_name,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        return {"ok": False, "error": f"Missing required fields: {', '.join(missing)}", "logs": []}

    logs: list[str] = []
    credential = DefaultAzureCredential()
    token = credential.get_token("https://management.azure.com/.default").token
    logs.append("Managed identity token acquired.")

    api_version = "2023-12-01"
    base = f"https://management.azure.com/subscriptions/{subscription_id}"
    rg_url = f"{base}/resourcegroups/{resource_group}?api-version=2023-07-01"
    plan_url = (
        f"{base}/resourceGroups/{resource_group}/providers/Microsoft.Web/"
        f"serverfarms/{plan_name}?api-version={api_version}"
    )
    app_url_api = (
        f"{base}/resourceGroups/{resource_group}/providers/Microsoft.Web/"
        f"sites/{webapp_name}?api-version={api_version}"
    )

    if auto_create:
        _arm_request("PUT", rg_url, token, {"location": region})
        logs.append(f"Resource group ensured: {resource_group}")

        plan_payload = {
            "location": region,
            "kind": "linux",
            "sku": {"name": "B1", "tier": "Basic", "size": "B1", "capacity": 1},
            "properties": {"reserved": True},
        }
        _arm_request("PUT", plan_url, token, plan_payload)
        logs.append(f"App Service plan ensured: {plan_name}")

        app_payload = {
            "location": region,
            "kind": "app,linux",
            "properties": {
                "serverFarmId": (
                    f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}"
                    f"/providers/Microsoft.Web/serverfarms/{plan_name}"
                ),
                "siteConfig": {"linuxFxVersion": _runtime_to_linux_fx_version(runtime)},
                "httpsOnly": True,
            },
        }
        _arm_request("PUT", app_url_api, token, app_payload, timeout=900)
        logs.append(f"Web App ensured: {webapp_name}")

    if startup_command:
        config_url = (
            f"{base}/resourceGroups/{resource_group}/providers/Microsoft.Web/"
            f"sites/{webapp_name}/config/web?api-version={api_version}"
        )
        _arm_request("PUT", config_url, token, {"properties": {"appCommandLine": startup_command}})
        logs.append("Startup command updated.")

    if deployment_mode == "zip":
        publish_cred_url = (
            f"{base}/resourceGroups/{resource_group}/providers/Microsoft.Web/"
            f"sites/{webapp_name}/config/publishingcredentials/list?api-version={api_version}"
        )
        cred_resp = _arm_request("POST", publish_cred_url, token, {})
        props = cred_resp.get("properties", {}) if isinstance(cred_resp, dict) else {}
        publish_user = (
            props.get("publishingUserName")
            or props.get("publishingUsername")
            or props.get("name")
            or ""
        ).strip()
        publish_password = (props.get("publishingPassword") or props.get("password") or "").strip()
        if not publish_user or not publish_password:
            raise RuntimeError("Could not fetch publishing credentials for ZIP deploy.")

        zip_bytes = _build_zip_bytes(project)
        # Some ARM responses include scmUri with embedded userinfo that can break URL parsing.
        # Build a clean Kudu endpoint from web app name and send Basic auth header explicitly.
        scm_uri = f"https://{webapp_name}.scm.azurewebsites.net"
        deploy_url = f"{scm_uri}/api/zipdeploy?isAsync=true"
        auth = base64.b64encode(f"{publish_user}:{publish_password}".encode("utf-8")).decode("utf-8")
        deploy_req = urllib.request.Request(
            deploy_url,
            data=zip_bytes,
            method="POST",
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/zip",
            },
        )
        with urllib.request.urlopen(deploy_req, timeout=900):
            pass
        logs.append("ZIP deployment submitted to Kudu.")
    elif deployment_mode == "external_git":
        if not repo_url:
            return {"ok": False, "error": "repo_url is required for external git mode.", "logs": logs}
        source_url = (
            f"{base}/resourceGroups/{resource_group}/providers/Microsoft.Web/"
            f"sites/{webapp_name}/sourcecontrols/web?api-version={api_version}"
        )
        source_payload = {
            "properties": {
                "repoUrl": repo_url,
                "branch": repo_branch,
                "isManualIntegration": True,
                "isMercurial": False,
            }
        }
        _arm_request("PUT", source_url, token, source_payload)
        logs.append("External Git deployment source configured.")
    else:
        return {"ok": False, "error": "Invalid deployment mode.", "logs": logs}

    app_url = f"https://{webapp_name}.azurewebsites.net"
    return {"ok": True, "app_url": app_url, "logs": logs, "message": "Azure publish completed."}


def _publish_job_json(job: PublishJob) -> dict:
    return {
        "job_id": str(job.id),
        "status": job.status,
        "deployment_mode": job.deployment_mode,
        "result_url": job.result_url or "",
        "error_message": job.error_message or "",
        "logs": job.logs or "",
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


@login_required
@require_http_methods(["POST"])
def publish_session_to_azure(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    project = session.generated_projects.order_by("-created_at").first()
    if not project:
        return JsonResponse({"error": "No saved code found in this session."}, status=404)

    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    subscription_id = str(payload.get("subscription_id") or "").strip()
    resource_group = str(payload.get("resource_group") or "").strip()
    plan_name = str(payload.get("app_service_plan") or "").strip()
    webapp_name = str(payload.get("webapp_name") or "").strip()
    region = str(payload.get("region") or "").strip() or "centralindia"
    runtime = str(payload.get("runtime") or "").strip() or _detect_runtime_from_project(project)
    startup_command = str(payload.get("startup_command") or "").strip() or _detect_startup_command_from_project(
        project, runtime
    )
    auto_create = bool(payload.get("auto_create", False))
    deployment_mode = str(payload.get("deployment_mode") or "zip").strip().lower()
    repo_url = str(payload.get("repo_url") or "").strip()
    repo_branch = str(payload.get("repo_branch") or "").strip() or "main"

    valid_modes = {"zip", "external_git", "swa_publish"}
    if deployment_mode not in valid_modes:
        return JsonResponse({"error": "Invalid deployment mode."}, status=400)
    if deployment_mode in {"external_git", "swa_publish"} and not repo_url:
        return JsonResponse({"error": "repo_url is required for git deployment modes."}, status=400)

    job_payload = {
        "deployment_mode": deployment_mode,
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "app_service_plan": plan_name,
        "webapp_name": webapp_name,
        "region": region,
        "runtime": runtime,
        "startup_command": startup_command,
        "auto_create": auto_create,
        "repo_url": repo_url,
        "repo_branch": repo_branch,
        "static_web_app_name": str(payload.get("static_web_app_name") or "").strip(),
        "swa_url": str(payload.get("swa_url") or "").strip(),
        "swa_auto_create": bool(payload.get("swa_auto_create", False)),
    }
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    github_token = str(profile.github_token or "").strip()

    try:
        result = _run_publish_via_managed_identity(project, job_payload, github_token=github_token)
        status = 200 if result.get("ok") else 502
        return JsonResponse(result, status=status)
    except Exception as exc:
        return JsonResponse({"error": f"Azure publish failed: {exc}"}, status=500)


@require_http_methods(["POST"])
def start_publish_job(request, session_id: int):
    if not request.user.is_authenticated:
        return JsonResponse({"ok": False, "error": "Authentication required. Please login again."}, status=401)

    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    project = session.generated_projects.order_by("-created_at").first()
    if not project:
        return JsonResponse({"error": "No saved code found in this session."}, status=404)

    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    deployment_mode = str(payload.get("deployment_mode") or "zip").strip().lower()
    if deployment_mode not in {"zip", "external_git", "swa_publish"}:
        return JsonResponse({"error": "Invalid deployment mode."}, status=400)

    job_payload = {
        "project_id": project.id,
        "session_id": session.id,
        "deployment_mode": deployment_mode,
        "resource_group": str(payload.get("resource_group") or "").strip(),
        "app_service_plan": str(payload.get("app_service_plan") or "").strip(),
        "webapp_name": str(payload.get("webapp_name") or "").strip(),
        "region": str(payload.get("region") or "").strip() or "centralindia",
        "runtime": str(payload.get("runtime") or "").strip() or _detect_runtime_from_project(project),
        "startup_command": str(payload.get("startup_command") or "").strip(),
        "auto_create": bool(payload.get("auto_create", False)),
        "repo_url": str(payload.get("repo_url") or "").strip(),
        "repo_branch": str(payload.get("repo_branch") or "").strip() or "main",
        "static_web_app_name": str(payload.get("static_web_app_name") or "").strip(),
        "swa_url": str(payload.get("swa_url") or "").strip(),
        "swa_auto_create": bool(payload.get("swa_auto_create", False)),
        "subscription_id": str(payload.get("subscription_id") or "").strip(),
    }
    if not job_payload["startup_command"]:
        job_payload["startup_command"] = _detect_startup_command_from_project(project, job_payload["runtime"])
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    github_token = str(profile.github_token or "").strip()

    job = PublishJob.objects.create(
        user=request.user,
        session=session,
        deployment_mode=deployment_mode,
        status=PublishJob.STATUS_QUEUED,
        payload=job_payload,
        logs="Publish started from Django using Azure SDK.",
    )

    job.status = PublishJob.STATUS_RUNNING
    job.logs = "Managed identity publish started from Django."
    job.save(update_fields=["status", "logs", "updated_at"])
    try:
        result = _run_publish_via_managed_identity(project, job_payload, github_token=github_token)
        result_logs = result.get("logs", [])
        if result_logs:
            job.logs = f"{job.logs}\n" + "\n".join(str(line) for line in result_logs)
        if result.get("app_url"):
            job.result_url = str(result.get("app_url"))
        if result.get("ok"):
            job.status = PublishJob.STATUS_SUCCESS
        else:
            job.status = PublishJob.STATUS_FAILED
            job.error_message = str(result.get("error") or "Managed identity publish failed.")
    except Exception as exc:
        job.status = PublishJob.STATUS_FAILED
        job.error_message = f"Managed identity publish failed: {exc}"
        job.logs = f"{job.logs}\n{job.error_message}".strip()
    job.save(update_fields=["status", "logs", "result_url", "error_message", "updated_at"])

    return JsonResponse({"ok": True, "job": _publish_job_json(job)})


@require_http_methods(["GET"])
def publish_job_status(request, job_id):
    if not request.user.is_authenticated:
        return JsonResponse({"ok": False, "error": "Authentication required. Please login again."}, status=401)

    job = get_object_or_404(PublishJob, pk=job_id, user=request.user)
    return JsonResponse({"ok": True, "job": _publish_job_json(job)})


def _github_request(
    method: str,
    path: str,
    token: str,
    payload: dict | None = None,
    query: dict | None = None,
) -> tuple[int, dict]:
    url = f"{GITHUB_API_BASE}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "srm-ai-code-generator",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            return response.status, parsed
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"message": body or "GitHub API error"}
        return exc.code, parsed


def _github_get_file_sha(owner: str, repo: str, path: str, branch: str, token: str) -> str | None:
    status, data = _github_request(
        "GET",
        f"/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}",
        token,
        query={"ref": branch},
    )
    if status == 200 and isinstance(data, dict):
        return data.get("sha")
    return None


def _github_put_file(
    owner: str,
    repo: str,
    path: str,
    content: str,
    branch: str,
    token: str,
    sha: str | None = None,
) -> tuple[int, dict]:
    payload = {
        "message": f"Update {path} via SRM AI Code Generator",
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    return _github_request(
        "PUT",
        f"/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}",
        token,
        payload=payload,
    )


@login_required
def session_test_ide(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    project = session.generated_projects.order_by("-created_at").first()
    files = project.files if project else []
    virtual_path = (
        f"/virtual/{request.user.username}/session-{session.id}/{project.project_name}"
        if project
        else f"/virtual/{request.user.username}/session-{session.id}/no-project"
    )
    return render(
        request,
        "chatbot/test_ide.html",
        {
            "session": session,
            "project": project,
            "files": files,
            "virtual_path": virtual_path,
            "download_url": reverse("download_session_project", args=[session.id]) if project else "",
        },
    )


@login_required
@require_http_methods(["POST"])
def save_session_file(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    project = session.generated_projects.order_by("-created_at").first()
    if not project:
        return JsonResponse({"error": "No generated project found for this session."}, status=404)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    path = str(payload.get("path") or "").strip()
    content = str(payload.get("content") or "")
    if not path:
        return JsonResponse({"error": "path is required."}, status=400)

    try:
        safe_path = _safe_relative_path(path).as_posix()
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    files = project.files if isinstance(project.files, list) else []
    updated = False
    for item in files:
        if str(item.get("path", "")).strip() == safe_path:
            item["content"] = content
            updated = True
            break

    if not updated:
        return JsonResponse({"error": "File not found in virtual project."}, status=404)

    project.files = files
    project.save(update_fields=["files"])
    return JsonResponse({"ok": True, "message": "File saved successfully."})


@login_required
@require_http_methods(["POST"])
def delete_session(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    session.delete()
    return JsonResponse({"ok": True, "session_id": session_id})


@login_required
@require_http_methods(["POST"])
def delete_saved_code(request, project_id: int):
    project = get_object_or_404(GeneratedProject, pk=project_id, user=request.user)
    project.delete()
    messages.success(request, "Saved code deleted.")
    return redirect("profile_page")


@login_required
@require_http_methods(["POST"])
def push_session_to_github(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    project = session.generated_projects.order_by("-created_at").first()
    if not project:
        return JsonResponse({"error": "No saved code found in this session."}, status=404)

    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    github_username = (profile.github_username or "").strip()
    github_token = (profile.github_token or "").strip()
    if not github_username or not github_token:
        return JsonResponse(
            {"error": "GitHub username/token missing. Update your profile first."},
            status=400,
        )

    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    repo_name = str(payload.get("repo_name") or "").strip()
    overwrite = bool(payload.get("overwrite", False))
    if not repo_name:
        return JsonResponse({"error": "repo_name is required."}, status=400)

    status_user, user_data = _github_request("GET", "/user", github_token)
    if status_user != 200:
        return JsonResponse(
            {"error": f"GitHub auth failed: {user_data.get('message', 'Invalid token')}"},
            status=401,
        )

    status_repo, repo_data = _github_request(
        "GET",
        f"/repos/{github_username}/{repo_name}",
        github_token,
    )

    repo_exists = status_repo == 200
    if repo_exists and not overwrite:
        return JsonResponse(
            {
                "need_overwrite": True,
                "message": f"Repository '{repo_name}' already exists. Overwrite code?",
            },
            status=200,
        )

    if not repo_exists:
        status_create, create_data = _github_request(
            "POST",
            "/user/repos",
            github_token,
            payload={"name": repo_name, "private": False, "auto_init": True},
        )
        if status_create not in (201, 202):
            return JsonResponse(
                {"error": f"Failed to create repo: {create_data.get('message', 'unknown error')}"},
                status=502,
            )
        repo_data = create_data

    default_branch = str(repo_data.get("default_branch") or "main")
    files = project.files if isinstance(project.files, list) else []
    if not files:
        return JsonResponse({"error": "No files available to push."}, status=400)

    pushed_count = 0
    for item in files:
        path = str(item.get("path", "")).strip()
        content = str(item.get("content", ""))
        if not path:
            continue

        sha = _github_get_file_sha(github_username, repo_name, path, default_branch, github_token)
        put_status, put_data = _github_put_file(
            github_username,
            repo_name,
            path,
            content,
            default_branch,
            github_token,
            sha=sha,
        )
        if put_status not in (200, 201):
            return JsonResponse(
                {"error": f"Failed to push '{path}': {put_data.get('message', 'unknown error')}"},
                status=502,
            )
        pushed_count += 1

    repo_url = f"https://github.com/{github_username}/{repo_name}"
    return JsonResponse(
        {
            "ok": True,
            "message": f"Pushed {pushed_count} file(s) to {repo_name}.",
            "repo_url": repo_url,
            "branch": default_branch,
        }
    )


@login_required
@require_POST
def chat_api(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    message = (payload.get("message") or "").strip()
    session_id = payload.get("session_id")
    if not message:
        return JsonResponse({"error": "message is required."}, status=400)

    missing_settings = _missing_azure_settings()
    if missing_settings:
        return JsonResponse(
            {
                "error": "Azure OpenAI settings are missing.",
                "missing": missing_settings,
            },
            status=500,
        )

    session = None
    if session_id:
        session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    else:
        session = ChatSession.objects.create(
            user=request.user,
            title=_session_title_from_message(message),
        )

    previous_messages = list(session.messages.values("role", "content"))
    request_messages = [
        {
            "role": "system",
            "content": (
                "You are an interactive coding copilot. "
                "Handle normal chat questions and coding help naturally."
            ),
        },
        *previous_messages,
    ]

    latest_project = session.generated_projects.order_by("-created_at").first()
    generated = None
    download_url = ""
    session_download_url = reverse("download_session_project", args=[session.id]) if latest_project else ""
    trigger_download = False

    try:
        if _is_download_intent(message):
            if latest_project:
                download_url = reverse("download_project", args=[latest_project.id])
                session_download_url = reverse("download_session_project", args=[session.id])
                trigger_download = True
                answer = (
                    f"{request.user.username}, I understood your download request. "
                    "Starting ZIP download for your latest virtual project now."
                )
            else:
                answer = (
                    f"{request.user.username}, I could not find generated files in this session yet. "
                    "Generate a project first, then ask me to download."
                )
        elif _is_project_generation_request(message):
            project_data = _build_project_files(message, request_messages)
            project_name, files = _normalize_generated_project(project_data)
            # Keep only the latest generated project per session.
            GeneratedProject.objects.filter(user=request.user, session=session).delete()
            generated = GeneratedProject.objects.create(
                user=request.user,
                session=session,
                project_name=project_name,
                files=files,
            )
            answer = _dynamic_user_reply(request.user.username, message, generated.project_name, files)
            download_url = reverse("download_project", args=[generated.id])
            session_download_url = reverse("download_session_project", args=[session.id])
        elif latest_project and _is_project_update_request(message):
            project_data = _build_updated_project_files(message, request_messages, latest_project)
            project_name, files = _normalize_generated_project(project_data)
            latest_project.project_name = project_name or latest_project.project_name
            latest_project.files = files
            latest_project.save(update_fields=["project_name", "files"])
            generated = latest_project
            answer = (
                f"{request.user.username}, your requested project changes are updated in this session. "
                f"I saved the latest virtual code with {len(files)} file(s). "
                "Open Test to verify or run it."
            )
            download_url = reverse("download_project", args=[generated.id])
            session_download_url = reverse("download_session_project", args=[session.id])
        else:
            answer = _build_conversational_reply(message, request_messages, latest_project)
    except Exception as exc:
        return JsonResponse(
            {"error": f"Bot response failed: {exc}"},
            status=502,
        )

    final_answer = answer or "No reply received."
    ChatMessage.objects.create(
        session=session,
        role=ChatMessage.ROLE_USER,
        content=message,
    )
    ChatMessage.objects.create(
        session=session,
        role=ChatMessage.ROLE_ASSISTANT,
        content=final_answer,
    )

    session.save(update_fields=["updated_at"])
    return JsonResponse(
        {
            "reply": final_answer,
            "session_id": session.id,
            "session_title": session.title or f"Chat {session.id}",
            "download_url": download_url,
            "session_download_url": session_download_url,
            "session_test_url": reverse("session_test_ide", args=[session.id]),
            "project_name": generated.project_name if generated else "",
            "trigger_download": trigger_download,
        }
    )
