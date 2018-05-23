from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import auth
from .models import Image, RecognitionResult, Mask, Session
from django.contrib.auth.models import User
from .forms import UploadFileForm, LoginForm, SignupForm
from .networks.train import predict_one
from django.shortcuts import render_to_response
from django.template import RequestContext


def handler404(request):
    return render(request, 'error.html', status=404)


def handler500(request):
    return render(request,'error.html',status=500)


# Create your views here.
def Upload_file(request):
  if request.method == 'POST':
    form = UploadFileForm(request.POST, request.FILES)
    print(form.errors)
    if form.is_valid():
        upload_image = Image(file_obj = request.FILES['file'])
        upload_image.save()
        prediction, featured_image_url, mask_url = predict_one(upload_image.file_obj.path)
        rec_res = RecognitionResult.objects.create(prediction=prediction,
                                                  featured_image_url=featured_image_url)
        mask = Mask.objects.create(mask_url=mask_url)
        session = Session.objects.create(upload_image=upload_image, rec_result=rec_res, mask=mask, user=request.user)
        return redirect('result', username=request.user.username, session_id=session.id)
    return render(request, 'in2.html', {'form': form})
  else:
    form = UploadFileForm()

  return render(request, 'in2.html', {'form': form})


def result(request, username, session_id):
    session = Session.objects.get(id=session_id)
    return render(request, 'result.html',
                  {'session': session})


def result_list(request, username):
    try:
        sessions = Session.objects.all().filter(user = request.user)
        return render(request, 'result_list.html',
                      {'sessions': sessions})
    except:
        return render(request, 'error.html')



def login(request):
    redirect_url = ''
    if request.method == 'POST':
        redirect_url = ''
        form = LoginForm(request.POST)
        if form.is_valid():
            user = auth.authenticate(username=form.cleaned_data['login'],
                                     password=form.cleaned_data['password'])
            if user:
                auth.login(request, user)
                return redirect('Upload_file')
            else:
                form.add_error(None, 'Неверное имя или пароль')
        else:
            form.add_error(None, 'Пожалуйста, заполните форму')
    else:
        form = LoginForm()
    return render(request, 'auth.html',
                  {'form': form, 'continue': redirect_url})


def logout(request):
    auth.logout(request)
    return HttpResponseRedirect("/login")


def signup(request):
    if request.user.is_authenticated:
        return redirect('Upload_file')
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            user = auth.authenticate(username=form.cleaned_data['login'],
                                     password=form.cleaned_data['password'])
            if user:
                auth.login(request, user)
                return redirect('Upload_file')
            else:
                form.add_error(None, 'Неверные данные или пользователь уже существует')
        else:
            form.add_error(None, 'Пожалуйста, заполните форму')
    else:
        form = SignupForm()
    return render(request, 'signup.html', {'form': form,
                                           'type': 'Registration'})
