# SRM AI Code Generator v1.0.5

This project is a Django chatbot/code-generator with:

- Azure OpenAI (endpoint + API key)
- Session-based saved code
- Download/Test/GitHub actions
- Publish page (Azure, AWS tabs)

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

## Azure Publish (Direct SDK)

Publish is now handled directly by Django using Azure SDK + Managed Identity.

Required Azure setup:

- Enable **System Assigned Managed Identity** on your Django App Service.
- Grant this identity RBAC permissions on target Azure scope (minimum: `Contributor` on target resource group).

Publish actions supported:

- ZIP deploy to App Service
- External Git source configuration
- Git CI/CD mode returns manual next steps for Deployment Center binding

Notes:

- Tenant/Client ID/Client Secret fields are optional and ignored for managed identity flow.
- For local testing, `DefaultAzureCredential` can use your `az login` identity.
