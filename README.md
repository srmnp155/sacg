# SRM AI Code Generator v1.0.3

This project is a Django chatbot/code-generator with:

- Azure OpenAI (endpoint + API key)
- Session-based saved code
- Download/Test/GitHub actions
- Publish page (Mail, Azure, AWS tabs)

## Local Run

1. Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create env file:

```powershell
Copy-Item .env.example .env
```

3. Set required values in `.env`:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `DJANGO_SECRET_KEY`
- `DJANGO_DEBUG=true` (local only)

4. Run:

```powershell
python manage.py migrate
python manage.py runserver
```

In development, password reset email content is printed to terminal (console backend).

## Azure App Service (Linux) Deployment Notes

Set these App Settings in Azure Portal:

- `DJANGO_SECRET_KEY`
- `DJANGO_DEBUG=false`
- `DJANGO_ALLOWED_HOSTS=<your-app-name>.azurewebsites.net`
- `DJANGO_CSRF_TRUSTED_ORIGINS=https://<your-app-name>.azurewebsites.net`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION=2024-02-15-preview`
- `AZURE_OPENAI_DEPLOYMENT`

Startup command:

```bash
python manage.py migrate && python manage.py collectstatic --noinput && gunicorn chatbot_project.wsgi --bind=0.0.0.0:$PORT --timeout 600
```

Optional email settings:

- `EMAIL_BACKEND`
- `EMAIL_HOST`
- `EMAIL_PORT`
- `EMAIL_HOST_USER`
- `EMAIL_HOST_PASSWORD`
- `EMAIL_USE_TLS`
- `EMAIL_USE_SSL`
- `DEFAULT_FROM_EMAIL`

## Azure Publish Modes

The Publish page now supports two backend execution modes:

1. Direct publish from Django using Azure Managed Identity (recommended on App Service)
2. Queue + Function worker mode (legacy/fallback)

### Managed Identity Mode (Recommended)

Set in App Service Configuration:

- `AZURE_PUBLISH_USE_MANAGED_IDENTITY=true`

For this mode:

- Enable **System Assigned Managed Identity** on your Django App Service.
- Grant this identity RBAC permissions on target Azure scope (minimum: `Contributor` on target resource group).
- Tenant/Client ID/Client Secret form fields are optional in this mode.

Publish actions supported:

- ZIP deploy to App Service
- External Git source configuration
- Git CI/CD mode returns manual next steps for Deployment Center binding

### Queue + Function Mode (Fallback)

Set in App Service Configuration:

- `AZURE_PUBLISH_USE_MANAGED_IDENTITY=false`
- `PUBLISH_QUEUE_CONNECTION_STRING=<storage-connection-string>`
- `PUBLISH_QUEUE_NAME=publish-jobs`
- `PUBLISH_WORKER_TOKEN=<shared-secret>`

Function App settings must include:

- `AzureWebJobsStorage=<same storage account used by Django queue>`
- `DJANGO_PUBLISH_STATUS_API=https://<django-app>.azurewebsites.net`
- `DJANGO_WORKER_SHARED_TOKEN=<same value as PUBLISH_WORKER_TOKEN>`
