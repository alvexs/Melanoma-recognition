from .models import Image
from django import forms
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.contrib.auth.models import User

class UploadFileForm(forms.Form):
    file = forms.FileField(widget=forms.FileInput({'class': 'inputfiles', 'onchange': 'ani()'}))


class LoginForm(forms.Form):
    login = forms.CharField(label='login')
    password = forms.CharField(label='password',
                               widget=forms.PasswordInput)


class SignupForm(forms.Form):
    login = forms.CharField(label='login', min_length=5, )
    email = forms.CharField(label='email')
    password = forms.CharField(label='password', min_length=8,
                               widget=forms.PasswordInput)
    repeat_password = forms.CharField(label='repeat_password',
                                      widget=forms.PasswordInput)

    def clean_login(self):
        login = self.cleaned_data['login']
        if User.objects.filter(username=login):
            raise ValidationError('Этот login уже занят')
        return login

    def clean_email(self):
        email = self.cleaned_data['email']
        validate_email(self.cleaned_data['email'])
        if User.objects.filter(email=email):
            raise ValidationError('Этот email уже зарегистрирован')
        return self.cleaned_data['email']

    def clean(self):
        cleaned_data = super(SignupForm, self).clean()
        if self.cleaned_data.get('password') \
                and self.cleaned_data.get('repeat_password'):
            if self.cleaned_data['password'] != \
                    self.cleaned_data['repeat_password']:
                raise ValidationError('Пароли не совпадают')
        return cleaned_data

    def save(self):
        user = User.objects.create_user(
            username=self.cleaned_data['login'],
            email=self.cleaned_data['email'],
            password=self.cleaned_data['password'],
        )
        return user
