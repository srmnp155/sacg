from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import UserProfile


class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email")

    def save(self, commit=True):
        user = super().save(commit=commit)
        profile, _ = UserProfile.objects.get_or_create(user=user)
        profile.deployment_mode = UserProfile.DEPLOYMENT_MODE_VSCODE
        profile.deployment_mode_prompt_seen = True
        profile.save(update_fields=["deployment_mode", "deployment_mode_prompt_seen", "updated_at"])
        return user


class ProfileForm(forms.ModelForm):
    first_name = forms.CharField(required=False, max_length=150)
    last_name = forms.CharField(required=False, max_length=150)
    email = forms.EmailField(required=True)
    github_token = forms.CharField(
        required=False,
        widget=forms.PasswordInput(render_value=False),
        help_text="Leave empty to keep your existing token.",
    )

    class Meta:
        model = UserProfile
        fields = ("phone_number", "github_username", "github_token")

    def __init__(self, *args, user: User, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user
        self.fields["first_name"].initial = user.first_name
        self.fields["last_name"].initial = user.last_name
        self.fields["email"].initial = user.email

    def save(self, commit=True):
        profile = super().save(commit=False)

        self.user.first_name = self.cleaned_data["first_name"]
        self.user.last_name = self.cleaned_data["last_name"]
        self.user.email = self.cleaned_data["email"]

        token = self.cleaned_data.get("github_token", "")
        if not token and profile.pk:
            profile.github_token = UserProfile.objects.get(pk=profile.pk).github_token
        else:
            profile.github_token = token

        if commit:
            self.user.save(update_fields=["first_name", "last_name", "email"])
            profile.save()

        return profile


class LoginForm(AuthenticationForm):
    username = forms.CharField(
        label="Username or Email",
        max_length=254,
        widget=forms.TextInput(attrs={"autofocus": True}),
    )

    def clean(self):
        username_or_email = str(self.cleaned_data.get("username") or "").strip()
        password = self.cleaned_data.get("password")
        if "@" in username_or_email:
            user = User.objects.filter(email__iexact=username_or_email).first()
            if user:
                self.cleaned_data["username"] = user.username
        else:
            self.cleaned_data["username"] = username_or_email
        self.cleaned_data["password"] = password
        return super().clean()
