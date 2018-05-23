"""recognizeMelanoma URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path, re_path
from recognizeApp import views
from django.conf import settings
from django.conf.urls.static import static


handler404 = views.handler404
handler500 = views.handler500

urlpatterns = [
    path('', views.Upload_file, name='Upload_file'),
    re_path(r'result/for-(?P<username>\w+)/session-(?P<session_id>\d+)/$', views.result, name='result'),
    path('admin/', admin.site.urls),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('signup/', views.signup, name='signup'),
    re_path(r'result-list/for-(?P<username>\w+)/$', views.result_list, name='result_list'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
