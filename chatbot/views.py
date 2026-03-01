import json
import io
import hashlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
import base64
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from azure.storage.queue import QueueClient
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.core.mail import EmailMessage
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
    return render(
        request,
        "chatbot/publish.html",
        {
            "session": session,
            "project": project,
            "download_url": reverse("download_session_project", args=[session.id]) if project else "",
            "test_url": reverse("session_test_ide", args=[session.id]) if project else "",
        },
    )


@login_required
@require_http_methods(["POST"])
def send_session_code_email(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    project = session.generated_projects.order_by("-created_at").first()
    if not project:
        return JsonResponse({"error": "No saved code found in this session."}, status=404)

    to_email = str(request.POST.get("to_email") or "").strip()
    subject = str(request.POST.get("subject") or "").strip() or f"Saved Code - Session {session.id}"
    body = str(request.POST.get("body") or "").strip() or "Attached is the requested saved code zip."
    if not to_email:
        return JsonResponse({"error": "Recipient email is required."}, status=400)

    from_email = (
        getattr(settings, "DEFAULT_FROM_EMAIL", "").strip()
        or getattr(settings, "EMAIL_HOST_USER", "").strip()
        or "no-reply@srm-ai.local"
    )
    attachment_name = f"{project.project_name}.zip"
    attachment_bytes = _build_zip_bytes(project)
    email = EmailMessage(
        subject=subject,
        body=body,
        from_email=from_email,
        to=[to_email],
    )
    email.attach(attachment_name, attachment_bytes, "application/zip")
    try:
        sent = email.send(fail_silently=False)
    except Exception as exc:
        return JsonResponse({"error": f"Email send failed: {exc}"}, status=502)
    if sent < 1:
        return JsonResponse({"error": "Email was not sent."}, status=502)

    return JsonResponse({"ok": True, "message": f"Email sent to {to_email} with {attachment_name}."})


def _run_process(command: list[str], timeout: int = 300) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    return completed.returncode, out


def _get_publish_queue_client() -> QueueClient:
    conn = (settings.PUBLISH_QUEUE_CONNECTION_STRING or "").strip()
    if not conn:
        raise ValueError("PUBLISH_QUEUE_CONNECTION_STRING is missing.")
    queue_name = (settings.PUBLISH_QUEUE_NAME or "publish-jobs").strip() or "publish-jobs"
    return QueueClient.from_connection_string(conn, queue_name)


def _enqueue_publish_job(job: PublishJob) -> None:
    queue_client = _get_publish_queue_client()
    try:
        queue_client.create_queue()
    except Exception:
        pass
    queue_client.send_message(json.dumps({"job_id": str(job.id)}))


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


def _is_worker_authorized(request) -> bool:
    configured = (settings.PUBLISH_WORKER_TOKEN or "").strip()
    if not configured:
        return False
    provided = (
        request.headers.get("X-Worker-Token")
        or request.META.get("HTTP_X_WORKER_TOKEN")
        or ""
    ).strip()
    return provided == configured


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

    tenant_id = str(payload.get("tenant_id") or "").strip()
    client_id = str(payload.get("client_id") or "").strip()
    client_secret = str(payload.get("client_secret") or "").strip()
    subscription_id = str(payload.get("subscription_id") or "").strip()
    resource_group = str(payload.get("resource_group") or "").strip()
    plan_name = str(payload.get("app_service_plan") or "").strip()
    webapp_name = str(payload.get("webapp_name") or "").strip()
    region = str(payload.get("region") or "").strip() or "centralindia"
    runtime = str(payload.get("runtime") or "").strip() or "PYTHON:3.11"
    startup_command = str(payload.get("startup_command") or "").strip()
    auto_create = bool(payload.get("auto_create", False))
    deployment_mode = str(payload.get("deployment_mode") or "zip").strip().lower()
    repo_url = str(payload.get("repo_url") or "").strip()
    repo_branch = str(payload.get("repo_branch") or "").strip() or "main"

    valid_modes = {"zip", "external_git", "git_cicd"}
    if deployment_mode not in valid_modes:
        return JsonResponse({"error": "Invalid deployment mode."}, status=400)
    if deployment_mode in {"external_git", "git_cicd"} and not repo_url:
        return JsonResponse({"error": "repo_url is required for git deployment modes."}, status=400)

    required = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_secret": client_secret,
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "app_service_plan": plan_name,
        "webapp_name": webapp_name,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        return JsonResponse({"error": f"Missing required fields: {', '.join(missing)}"}, status=400)

    az_cmd = _find_cmd(("az.cmd", "az"))
    if not az_cmd:
        return JsonResponse({"error": "Azure CLI not found. Install Azure CLI and try again."}, status=400)

    steps_log: list[str] = []

    try:
        code, output = _run_process(
            [
                az_cmd,
                "login",
                "--service-principal",
                "--username",
                client_id,
                "--password",
                client_secret,
                "--tenant",
                tenant_id,
            ]
        )
        steps_log.append("az login --service-principal")
        if code != 0:
            return JsonResponse({"error": "Azure login failed.", "logs": steps_log + [output]}, status=502)

        code, output = _run_process([az_cmd, "account", "set", "--subscription", subscription_id])
        steps_log.append(f"az account set --subscription {subscription_id}")
        if code != 0:
            return JsonResponse({"error": "Failed to select Azure subscription.", "logs": steps_log + [output]}, status=502)

        if auto_create:
            code, output = _run_process(
                [az_cmd, "group", "create", "--name", resource_group, "--location", region]
            )
            steps_log.append(f"az group create --name {resource_group} --location {region}")
            if code != 0:
                return JsonResponse({"error": "Failed to create resource group.", "logs": steps_log + [output]}, status=502)

            code, output = _run_process(
                [
                    az_cmd,
                    "appservice",
                    "plan",
                    "create",
                    "--name",
                    plan_name,
                    "--resource-group",
                    resource_group,
                    "--is-linux",
                    "--sku",
                    "B1",
                ]
            )
            steps_log.append(f"az appservice plan create --name {plan_name}")
            if code != 0:
                return JsonResponse({"error": "Failed to create App Service Plan.", "logs": steps_log + [output]}, status=502)

            code, output = _run_process(
                [
                    az_cmd,
                    "webapp",
                    "create",
                    "--name",
                    webapp_name,
                    "--resource-group",
                    resource_group,
                    "--plan",
                    plan_name,
                    "--runtime",
                    runtime,
                ]
            )
            steps_log.append(f"az webapp create --name {webapp_name} --runtime {runtime}")
            if code != 0:
                return JsonResponse({"error": "Failed to create Web App.", "logs": steps_log + [output]}, status=502)

        if startup_command:
            code, output = _run_process(
                [
                    az_cmd,
                    "webapp",
                    "config",
                    "set",
                    "--resource-group",
                    resource_group,
                    "--name",
                    webapp_name,
                    "--startup-file",
                    startup_command,
                ]
            )
            steps_log.append(f"az webapp config set --name {webapp_name} --startup-file <provided>")
            if code != 0:
                return JsonResponse({"error": "Failed to set startup command.", "logs": steps_log + [output]}, status=502)

        if deployment_mode == "zip":
            with tempfile.TemporaryDirectory(prefix="srm_publish_") as temp_dir:
                zip_path = Path(temp_dir) / f"{project.project_name}.zip"
                zip_path.write_bytes(_build_zip_bytes(project))

                code, output = _run_process(
                    [
                        az_cmd,
                        "webapp",
                        "deploy",
                        "--resource-group",
                        resource_group,
                        "--name",
                        webapp_name,
                        "--src-path",
                        str(zip_path),
                        "--type",
                        "zip",
                    ],
                    timeout=900,
                )
                steps_log.append(f"az webapp deploy --name {webapp_name} --type zip")
                if code != 0:
                    code2, output2 = _run_process(
                        [
                            az_cmd,
                            "webapp",
                            "deployment",
                            "source",
                            "config-zip",
                            "--resource-group",
                            resource_group,
                            "--name",
                            webapp_name,
                            "--src",
                            str(zip_path),
                        ],
                        timeout=900,
                    )
                    steps_log.append(f"az webapp deployment source config-zip --name {webapp_name}")
                    if code2 != 0:
                        return JsonResponse(
                            {"error": "Azure zip deploy failed.", "logs": steps_log + [output, output2]},
                            status=502,
                        )
        elif deployment_mode == "external_git":
            cmd = [
                az_cmd,
                "webapp",
                "deployment",
                "source",
                "config",
                "--resource-group",
                resource_group,
                "--name",
                webapp_name,
                "--repo-url",
                repo_url,
                "--branch",
                repo_branch,
            ]
            # Azure CLI treats --manual-integration as a switch flag (no true/false value).
            cmd.append("--manual-integration")

            code, output = _run_process(cmd, timeout=900)
            mode_label = "external git"
            steps_log.append(
                f"az webapp deployment source config --name {webapp_name} --mode {mode_label}"
            )
            if code != 0:
                return JsonResponse(
                    {"error": f"Azure {mode_label} deployment configuration failed.", "logs": steps_log + [output]},
                    status=502,
                )
        else:
            # CI/CD linkage usually requires an interactive user context (GitHub authorization).
            # Service principal auth can provision Azure resources but often cannot finalize GitHub Actions binding.
            steps_log.append(
                "git ci/cd mode selected: resource prerequisites completed; manual GitHub/App Service linkage required"
            )
            app_url = f"https://{webapp_name}.azurewebsites.net"
            return JsonResponse(
                {
                    "ok": True,
                    "message": (
                        "Azure app prerequisites are ready. Complete CI/CD binding in Azure Deployment Center "
                        "using GitHub Actions for this Web App."
                    ),
                    "app_url": app_url,
                    "logs": steps_log,
                    "manual_action_required": True,
                    "next_steps": [
                        "Open Azure Portal -> App Service -> Deployment Center.",
                        "Choose GitHub as source and authorize GitHub account.",
                        f"Select repository URL: {repo_url}",
                        f"Select branch: {repo_branch}",
                        "Save and let GitHub Actions workflow run.",
                    ],
                }
            )

        app_url = f"https://{webapp_name}.azurewebsites.net"
        mode_msg = {
            "zip": "ZIP deployment completed",
            "external_git": "External Git deployment configured",
            "git_cicd": "Git CI/CD deployment configured",
        }.get(deployment_mode, "Publish completed")
        return JsonResponse(
            {
                "ok": True,
                "message": f"{mode_msg} for {webapp_name}.",
                "app_url": app_url,
                "logs": steps_log,
            }
        )
    except subprocess.TimeoutExpired:
        return JsonResponse({"error": "Azure publish timed out."}, status=504)
    except Exception as exc:
        return JsonResponse({"error": f"Azure publish failed: {exc}"}, status=500)


@login_required
@require_http_methods(["POST"])
def start_publish_job(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    project = session.generated_projects.order_by("-created_at").first()
    if not project:
        return JsonResponse({"error": "No saved code found in this session."}, status=404)

    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    deployment_mode = str(payload.get("deployment_mode") or "zip").strip().lower()
    if deployment_mode not in {"zip", "external_git", "git_cicd"}:
        return JsonResponse({"error": "Invalid deployment mode."}, status=400)

    job_payload = {
        "project_id": project.id,
        "session_id": session.id,
        "deployment_mode": deployment_mode,
        "resource_group": str(payload.get("resource_group") or "").strip(),
        "app_service_plan": str(payload.get("app_service_plan") or "").strip(),
        "webapp_name": str(payload.get("webapp_name") or "").strip(),
        "region": str(payload.get("region") or "").strip() or "centralindia",
        "runtime": str(payload.get("runtime") or "").strip() or "PYTHON:3.11",
        "startup_command": str(payload.get("startup_command") or "").strip(),
        "auto_create": bool(payload.get("auto_create", False)),
        "repo_url": str(payload.get("repo_url") or "").strip(),
        "repo_branch": str(payload.get("repo_branch") or "").strip() or "main",
        "tenant_id": str(payload.get("tenant_id") or "").strip(),
        "subscription_id": str(payload.get("subscription_id") or "").strip(),
        "client_id": str(payload.get("client_id") or "").strip(),
        "client_secret": str(payload.get("client_secret") or "").strip(),
    }

    job = PublishJob.objects.create(
        user=request.user,
        session=session,
        deployment_mode=deployment_mode,
        status=PublishJob.STATUS_QUEUED,
        payload=job_payload,
        logs="Job queued from Django publish page.",
    )

    try:
        _enqueue_publish_job(job)
    except Exception as exc:
        job.status = PublishJob.STATUS_FAILED
        job.error_message = f"Queue enqueue failed: {exc}"
        job.logs = f"{job.logs}\n{job.error_message}".strip()
        job.save(update_fields=["status", "error_message", "logs", "updated_at"])
        return JsonResponse({"error": job.error_message}, status=502)

    return JsonResponse({"ok": True, "job": _publish_job_json(job)})


@login_required
@require_http_methods(["GET"])
def publish_job_status(request, job_id):
    job = get_object_or_404(PublishJob, pk=job_id, user=request.user)
    return JsonResponse({"ok": True, "job": _publish_job_json(job)})


@require_http_methods(["GET"])
def worker_publish_job_detail(request, job_id):
    if not _is_worker_authorized(request):
        return JsonResponse({"error": "Unauthorized worker request."}, status=401)

    job = get_object_or_404(PublishJob, pk=job_id)
    project = job.session.generated_projects.order_by("-created_at").first()
    files = project.files if project and isinstance(project.files, list) else []
    return JsonResponse(
        {
            "ok": True,
            "job": _publish_job_json(job),
            "payload": job.payload or {},
            "project_name": project.project_name if project else "",
            "files": files,
            "owner_username": job.user.username,
        }
    )


@require_http_methods(["POST"])
def worker_publish_job_update(request, job_id):
    if not _is_worker_authorized(request):
        return JsonResponse({"error": "Unauthorized worker request."}, status=401)

    job = get_object_or_404(PublishJob, pk=job_id)
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    new_status = str(payload.get("status") or "").strip().lower()
    if new_status in {
        PublishJob.STATUS_QUEUED,
        PublishJob.STATUS_RUNNING,
        PublishJob.STATUS_SUCCESS,
        PublishJob.STATUS_FAILED,
    }:
        job.status = new_status

    incoming_logs = str(payload.get("logs") or "").strip()
    if incoming_logs:
        if job.logs:
            job.logs = f"{job.logs}\n{incoming_logs}"
        else:
            job.logs = incoming_logs

    error_message = str(payload.get("error_message") or "").strip()
    if error_message:
        job.error_message = error_message

    result_url = str(payload.get("result_url") or "").strip()
    if result_url:
        job.result_url = result_url

    job.save(update_fields=["status", "logs", "error_message", "result_url", "updated_at"])
    return JsonResponse({"ok": True, "job": _publish_job_json(job)})


def _pick_default_entry_file(files: list[dict]) -> str | None:
    preferred = (
        "main.py",
        "app.py",
        "run.py",
        "manage.py",
        "main.js",
        "index.js",
        "main.ts",
        "main.go",
        "main.java",
        "main.c",
        "main.cpp",
        "main.cs",
        "main.rb",
        "main.php",
        "main.sh",
        "main.ps1",
    )
    paths = [str(item.get("path", "")).strip() for item in files if str(item.get("path", "")).strip()]
    for name in preferred:
        if name in paths:
            return name
    runnable_exts = (
        ".py",
        ".js",
        ".ts",
        ".java",
        ".c",
        ".cpp",
        ".go",
        ".rb",
        ".php",
        ".pl",
        ".sh",
        ".ps1",
        ".rs",
    )
    for path in paths:
        if path.lower().endswith(runnable_exts):
            return path
    return None


def _find_cmd(candidates: tuple[str, ...]) -> str | None:
    for cmd in candidates:
        if shutil.which(cmd):
            return cmd
    return None


def _resolve_run_steps(entry_path: str, entry_abs: Path, base_dir: Path) -> tuple[list[list[str]], str]:
    ext = entry_abs.suffix.lower()
    steps: list[list[str]] = []
    language = "Unknown"

    if ext == ".py":
        steps = [[sys.executable, str(entry_abs)]]
        language = "Python"
    elif ext == ".js":
        node = _find_cmd(("node",))
        if not node:
            raise ValueError("Node.js runtime not found.")
        steps = [[node, str(entry_abs)]]
        language = "JavaScript"
    elif ext == ".ts":
        ts_node = _find_cmd(("ts-node",))
        deno = _find_cmd(("deno",))
        if ts_node:
            steps = [[ts_node, str(entry_abs)]]
        elif deno:
            steps = [[deno, "run", str(entry_abs)]]
        else:
            raise ValueError("TypeScript runtime not found (ts-node or deno required).")
        language = "TypeScript"
    elif ext == ".java":
        javac = _find_cmd(("javac",))
        java = _find_cmd(("java",))
        if not javac or not java:
            raise ValueError("Java runtime not found (javac/java required).")
        class_name = entry_abs.stem
        steps = [[javac, str(entry_abs)], [java, "-cp", str(entry_abs.parent), class_name]]
        language = "Java"
    elif ext == ".c":
        cc = _find_cmd(("gcc", "clang", "cc"))
        if not cc:
            raise ValueError("C compiler not found (gcc/clang/cc required).")
        out_file = base_dir / "run_c_exec"
        steps = [[cc, str(entry_abs), "-o", str(out_file)], [str(out_file)]]
        language = "C"
    elif ext == ".cpp":
        cxx = _find_cmd(("g++", "clang++", "c++"))
        if not cxx:
            raise ValueError("C++ compiler not found (g++/clang++/c++ required).")
        out_file = base_dir / "run_cpp_exec"
        steps = [[cxx, str(entry_abs), "-o", str(out_file)], [str(out_file)]]
        language = "C++"
    elif ext == ".go":
        go = _find_cmd(("go",))
        if not go:
            raise ValueError("Go runtime not found.")
        steps = [[go, "run", str(entry_abs)]]
        language = "Go"
    elif ext == ".rb":
        ruby = _find_cmd(("ruby",))
        if not ruby:
            raise ValueError("Ruby runtime not found.")
        steps = [[ruby, str(entry_abs)]]
        language = "Ruby"
    elif ext == ".php":
        php = _find_cmd(("php",))
        if not php:
            raise ValueError("PHP runtime not found.")
        steps = [[php, str(entry_abs)]]
        language = "PHP"
    elif ext == ".pl":
        perl = _find_cmd(("perl",))
        if not perl:
            raise ValueError("Perl runtime not found.")
        steps = [[perl, str(entry_abs)]]
        language = "Perl"
    elif ext == ".sh":
        sh_cmd = _find_cmd(("bash", "sh"))
        if not sh_cmd:
            raise ValueError("Shell runtime not found (bash/sh required).")
        steps = [[sh_cmd, str(entry_abs)]]
        language = "Shell"
    elif ext == ".ps1":
        pwsh = _find_cmd(("pwsh", "powershell"))
        if not pwsh:
            raise ValueError("PowerShell runtime not found (pwsh/powershell required).")
        steps = [[pwsh, "-ExecutionPolicy", "Bypass", "-File", str(entry_abs)]]
        language = "PowerShell"
    elif ext == ".rs":
        rustc = _find_cmd(("rustc",))
        if not rustc:
            raise ValueError("Rust compiler not found.")
        out_file = base_dir / "run_rust_exec"
        steps = [[rustc, str(entry_abs), "-o", str(out_file)], [str(out_file)]]
        language = "Rust"
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Select a runnable source file.")

    return steps, language


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
def run_session_project(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    project = session.generated_projects.order_by("-created_at").first()
    if not project:
        return JsonResponse({"error": "No generated project found for this session."}, status=404)

    files = project.files if isinstance(project.files, list) else []
    file_paths = [str(item.get("path", "")).strip() for item in files if str(item.get("path", "")).strip()]

    stdin_data = ""
    timeout_seconds = 30
    requested_entry = ""
    try:
        if request.body:
            payload = json.loads(request.body.decode("utf-8"))
            stdin_data = str(payload.get("stdin") or "")
            requested_timeout = int(payload.get("timeout") or timeout_seconds)
            timeout_seconds = min(max(requested_timeout, 1), 60)
            requested_entry = str(payload.get("entry_path") or "").strip()
    except (ValueError, TypeError, json.JSONDecodeError):
        stdin_data = ""
        timeout_seconds = 30
        requested_entry = ""

    entry_path = requested_entry or _pick_default_entry_file(files)
    if not entry_path:
        return JsonResponse({"error": "No runnable file found. Select a source file and run again."}, status=400)
    if entry_path not in file_paths:
        return JsonResponse({"error": "Selected entry file is not part of saved project files."}, status=400)

    try:
        with tempfile.TemporaryDirectory(prefix="srm_ide_") as temp_dir:
            base = Path(temp_dir)
            for item in files:
                raw_path = str(item.get("path", "")).strip()
                content = str(item.get("content", ""))
                if not raw_path:
                    continue
                rel = _safe_relative_path(raw_path)
                full_path = base / rel
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")

            run_target = base / _safe_relative_path(entry_path)
            steps, language = _resolve_run_steps(entry_path, run_target, base)

            final_return_code = 0
            output_parts: list[str] = []
            for idx, step in enumerate(steps):
                is_last = idx == len(steps) - 1
                completed = subprocess.run(
                    step,
                    cwd=str(base),
                    capture_output=True,
                    text=True,
                    input=stdin_data if is_last else None,
                    timeout=timeout_seconds,
                )
                step_output = (completed.stdout or "") + (completed.stderr or "")
                if step_output.strip():
                    output_parts.append(step_output)
                final_return_code = completed.returncode
                if completed.returncode != 0:
                    break

            output = "\n".join(part for part in output_parts if part.strip())
            if not output.strip():
                output = "Execution completed with no output."
            return JsonResponse(
                {
                    "return_code": final_return_code,
                    "entry_file": entry_path,
                    "language": language,
                    "output": output,
                }
            )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except subprocess.TimeoutExpired:
        return JsonResponse(
            {
                "error": (
                    f"Execution timed out after {timeout_seconds} seconds. "
                    "If your code needs input(), provide sample input in the IDE input box."
                )
            },
            status=408,
        )
    except Exception as exc:
        return JsonResponse({"error": f"Execution failed: {exc}"}, status=500)


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
