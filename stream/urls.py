"""stream URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
# 2019-07-23 Modified by Tran Le Anh

from django.contrib import admin
from django.urls import path
from webcam.views import index, video_feed_1, video_feed_2, camera_1, camera_2, model_eval, video_eval, stream_Upl_Video_feed, settings
# from webcam.views import database, search

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', index),
    path('video_feed_1/', video_feed_1, name="video-feed-1"),
    path('video_feed_2/', video_feed_2, name="video-feed-2"),
    path('index/camera1/', camera_1),
    path('index/camera2/', camera_2),
    path('index/model_eval/', model_eval, name='model_eval'),
    path('stream_Upl_Video_feed/', stream_Upl_Video_feed, name="stream_Upl_Video_feed"),
    path('index/video_analysis/', video_eval, name='video_eval'),
    path('index/settings/', settings, name='settings'),

]

from django.conf import settings
from django.conf.urls.static import static
if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)