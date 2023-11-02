from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import RequestContext, loader
from django.http.response import StreamingHttpResponse

import cv2
import numpy as np
import datetime
import time
import threading
import queue
from PIL import Image

from typing import Union

import matplotlib.pyplot as plt
import numpy as np

import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from .models import Image as ImageModel
from .models import Video as VideoModel
import random
from django.conf import settings as Settings
import os
import telepot
from datetime import datetime as dt
import pytz

#--------------Camera Settings-----------

camera1 = '0'
camera1_default = '0'

camera2 = '1'
camera2_default = '1'

timeout = '10'


tel_id = 'Your Telegram Bot id'
tel_id_default = 'Your Telegram Bot id'

video_path = None

#------------------------------------

# HOME PAGE -------------------------
def index(request):
	template = loader.get_template('index.html')
	return HttpResponse(template.render({}, request))
# -----------------------------------

# CAMERA 1 PAGE ---------------------
def camera_1(request):
	template = loader.get_template('camera1.html')
	return HttpResponse(template.render({}, request))
# -----------------------------------


# CAMERA 2 PAGE ---------------------
def camera_2(request):
	template = loader.get_template('camera2.html')
	return HttpResponse(template.render({}, request))
# -----------------------------------


# Image Analysis PAGE ---------------------
def model_eval(request):
    context = {}
    if request.POST:
        MEDIA_ROOT = Settings.MEDIA_ROOT
        print(MEDIA_ROOT)
        img = request.FILES['img']
        img_obj  = ImageModel()
        img_obj.img = img
        img_obj.name = random.randint(10000,1000000)
        img_obj.save()
        img_name = img_obj.img.url.split('/')[-1]
        # print(img_name)
        img_path = os.path.join(MEDIA_ROOT,'images',img_name)
        # print(img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predict_img_path = model_predict(image)
        context['img_obj'] = img_obj
        context['predict_img'] = "image/"+predict_img_path.split('\\')[-1]
        print(predict_img_path)
    template = loader.get_template('image_eval.html')
    return HttpResponse(template.render(context, request))
# -----------------------------------




# Video Analysis PAGE ---------------------
def video_eval(request):
    global video_path
    context = {}
    if request.POST:
        MEDIA_ROOT = Settings.MEDIA_ROOT
        print(MEDIA_ROOT)
        img = request.FILES['video']
        img_obj  = VideoModel()
        img_obj.img = img
        img_obj.name = random.randint(10000,1000000)
        img_obj.save()
        img_name = img_obj.img.url.split('/')[-1]
        # print(img_name)
        save_path = os.path.join(MEDIA_ROOT,'video',img_name)
        print(save_path)
        video_path = save_path
        context['img_obj'] = img_obj
    template = loader.get_template('video_eval.html')
    return HttpResponse(template.render(context, request))
# -----------------------------------


# SETTINGS PAGE ---------------------
def settings(request):
    global camera1, camera2, timeout, tel_id

    if request.POST:
        camera1 = request.POST.get('camera-1')
        camera2 = request.POST.get('camera-2')
        timeout = request.POST.get('timeout')
        tel_id = request.POST.get('tel_id')
    
    template = loader.get_template('settings.html')
    context = {'camera_1':camera1,'camera_2':camera2,'timeout':timeout,'tel_id':tel_id}
    return HttpResponse(template.render(context, request))
# -----------------------------------

# DISPLAY CAMERA 1 ------------------

def record_video1(frames):
    print("Entered Thread")
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # video codec
    now = datetime.datetime.now()
    timeMoment = getTime()
    location = "Bangalore"
    id = 941558875
    filename1 = r'media/telebot/violence.png'
    filename2 = now.strftime("media/telebot/"+"%Y-%m-%d_%H-%M-%S") + '.avi'
    out = cv2.VideoWriter(filename2, fourcc, 7, (640, 480)) # video writer

    while True: 
        print("Recording..")
        if not frames.empty():
            frame = frames.get()
            out.write(frame)
        else:
            break

    out.release()
    print("...Finished")

    #Telgram Configuration
    if tel_id != '':
        bot = telepot.Bot(tel_id)
    else:
        bot = telepot.Bot(tel_id_default)
    bot.sendMessage(id, f"VIOLENCE ALERT!! \nCamera: 1 \nLOCATION: {location} \nTIME: {timeMoment}")
    bot.sendPhoto(id, photo=open(filename1, 'rb'))
    print("Succesfully Sent Message") 
    bot.sendVideo(941558875, video=open(filename2, 'rb'))
    print("Succesfully Sent Video")

def stream_1():
    if camera1.isdigit() and camera1 != '':
        Number = int(camera1)
    else:
        Number = camera1_default
    cam_id =Number
    no_signal = r'webcam\templates\images\no signal.jpg'
    model = Model()
    cap = cv2.VideoCapture(cam_id)
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    print('FPS', fps)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    img_save_path = 'media/telebot/violence.png'
    success, frame = cap.read()
    frames = queue.Queue()
    success = True
    label_text = None
    label_updated_time = 0
    violence_detected = False
    detetcted_viol = []
    start_recording = False
    recording_started = False

    while success and cap.isOpened():
        orig_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(frame)
        label = prediction['label']
        conf = prediction['confidence']
        if label not in ['street violence', 'fight on a street', 'violence in office', 'fire on a street', 'fire in office'] and not violence_detected:
            label_text = label
            label_updated_time = time.time()
        # Check if label is one of the specified labels and update label text and timer
        elif violence_detected:
            frames.put(orig_frame)
            label_text = detetcted_viol[0]
            a = time.time() - label_updated_time
            print(label_text)
            print(a)
            if a < 1:
                cv2.imwrite(img_save_path, save_img(frame, label_text))
            violence_detected = True
            start_recording = True
            if start_recording and not recording_started and a >= int(timeout):
                recording_started = True
                t1 = threading.Thread(target=record_video1, args=(frames,))
                t1.start()
            if a >= int(timeout):  # Update condition here
                label_text = None
                violence_detected = False
                detetcted_viol.clear()
                recording_started = False
                start_recording = False
                frames.put(None)
        elif label in ['street violence', 'fight on a street', 'violence in office', 'fire on a street', 'fire in office']:
            a = label
            detetcted_viol.append(a)
            violence_detected = True
        
        if label_text:
            frame = cv2.putText(frame, label_text.title(), 
                                (0, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (800, 400))
        cv2.imwrite('currentframe.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('currentframe.jpg', 'rb').read() + b'\r\n')
        success, frame = cap.read()


def video_feed_1(request):
	return StreamingHttpResponse(stream_1(), content_type='multipart/x-mixed-replace; boundary=frame')
# -----------------------------------


# DISPLAY CAMERA 2 ------------------

def record_video2(frames):
    print("Entered Thread")
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # video codec
    now = datetime.datetime.now()
    timeMoment = getTime()
    location = "Bangalore"
    id = 941558875
    filename1 = r'media/telebot/violence.png'
    filename2 = now.strftime("media/telebot/"+"%Y-%m-%d_%H-%M-%S") + '.avi'
    out = cv2.VideoWriter(filename2, fourcc, 7, (640, 480)) # video writer

    while True: 
        print("Recording..")
        if not frames.empty():
            frame = frames.get()
            out.write(frame)
        else:
            break

    out.release()
    print("...Finished")

    #Telgram Configuration
    if tel_id != '':
        bot = telepot.Bot(tel_id)
    else:
        bot = telepot.Bot(tel_id_default)
    bot.sendMessage(id, f"VIOLENCE ALERT!! \nCamera: 2 \nLOCATION: {location} \nTIME: {timeMoment}")
    bot.sendPhoto(id, photo=open(filename1, 'rb'))
    print("Succesfully Sent Message") 
    bot.sendVideo(941558875, video=open(filename2, 'rb'))
    print("Succesfully Sent Video")

def stream_2():
    if camera2.isdigit() and camera2 != '':
        Number = int(camera2)
    else:
        Number = camera2_default
    cam_id =Number
    no_signal = r'webcam\templates\images\no signal.jpg'
    model = Model()
    cap = cv2.VideoCapture(cam_id)
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    print('FPS', fps)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    img_save_path = 'media/telebot/violence.png'
    success, frame = cap.read()
    frames = queue.Queue()
    success = True
    label_text = None
    label_updated_time = 0
    violence_detected = False
    detetcted_viol = []
    start_recording = False
    recording_started = False

    while success and cap.isOpened():
        orig_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(frame)
        label = prediction['label']
        conf = prediction['confidence']
        if label not in ['street violence', 'fight on a street', 'violence in office', 'fire on a street', 'fire in office'] and not violence_detected:
            label_text = label
            label_updated_time = time.time()
        # Check if label is one of the specified labels and update label text and timer
        elif violence_detected:
            frames.put(orig_frame)
            label_text = detetcted_viol[0]
            a = time.time() - label_updated_time
            print(label_text)
            print(a)
            if a < 1:
                cv2.imwrite(img_save_path, save_img(frame, label_text))
            violence_detected = True
            start_recording = True
            if start_recording and not recording_started and a >= int(timeout):
                recording_started = True
                t2 = threading.Thread(target=record_video2, args=(frames,))
                t2.start()
            if a >= int(timeout):  # Update condition here
                label_text = None
                violence_detected = False
                detetcted_viol.clear()
                recording_started = False
                start_recording = False
                frames.put(None)
        elif label in ['street violence', 'fight on a street', 'violence in office', 'fire on a street', 'fire in office']:
            a = label
            detetcted_viol.append(a)
            violence_detected = True
        
        if label_text:
            frame = cv2.putText(frame, label_text.title(), 
                                (0, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (800, 400))
        cv2.imwrite('currentframe.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('currentframe.jpg', 'rb').read() + b'\r\n')
        success, frame = cap.read()


def video_feed_2(request):
	return StreamingHttpResponse(stream_2(), content_type='multipart/x-mixed-replace; boundary=frame')
# -----------------------------------

# Predict on Image------------

def model_predict(image):
    model = Model()
    label = model.predict(image)['label']
    return plot(image=image, title=label)

def plot(image: np.array, title: str, title_size: int = 18,
         figsize: tuple = (13, 7)):
    plt.figure(figsize=figsize)
    plt.title(title, size=title_size)
    plt.axis('off')
    plt.imshow(image)
    STATIC_ROOT = Settings.STATICFILES_DIRS[0]
    save_path = os.path.join(STATIC_ROOT,'image','predict.jpg')
    plt.savefig(save_path)
    return save_path

#--------------------------------------------------

# DISPLAY Uploaded Video ------------------------

def stream_Upl_Video():
    cam_id = video_path
    no_signal = r'webcam\templates\images\no signal.jpg'
    model = Model()
    cap = cv2.VideoCapture(cam_id)
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    print('FPS', fps)
    success, frame = cap.read()
    success = True
    label_text = None
    label_updated_time = 0
    violence_detected = False
    detetcted_viol = []

    while success and cap.isOpened():
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(frame)
        label = prediction['label']
        conf = prediction['confidence']
        label_text = label
        print(label_text)
        
        if label_text:
            frame = cv2.putText(frame, label_text.title(), 
                                (0, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (800, 400))
        cv2.imwrite('currentframe.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('currentframe.jpg', 'rb').read() + b'\r\n')
        success, frame = cap.read()


def stream_Upl_Video_feed(request):
	return StreamingHttpResponse(stream_Upl_Video(), content_type='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------


#------------------Model--------------------------

class Model:
    def __init__(self, settings_path: str = 'webcam/settings.yaml'):
        with open(settings_path, "r") as file:
            self.settings = yaml.safe_load(file)

        self.device = self.settings['model-settings']['device']
        self.model_name = self.settings['model-settings']['model-name']
        self.threshold = self.settings['model-settings']['prediction-threshold']
        self.model, self.preprocess = clip.load(self.model_name,
                                                device=self.device)
        self.labels = self.settings['label-settings']['labels']
        self.labels_ = []
        for label in self.labels:
            text = 'a photo of ' + label  # will increase model's accuracy
            self.labels_.append(text)

        self.text_features = self.vectorize_text(self.labels_)
        self.default_label = self.settings['label-settings']['default-label']

    @torch.no_grad()
    def transform_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image).convert('RGB')
        tf_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return tf_image

    @torch.no_grad()
    def tokenize(self, text: list):
        text = clip.tokenize(text).to(self.device)
        return text

    @torch.no_grad()
    def vectorize_text(self, text: list):
        tokens = self.tokenize(text=text)
        text_features = self.model.encode_text(tokens)
        return text_features

    @torch.no_grad()
    def predict_(self, text_features: torch.Tensor,
                 image_features: torch.Tensor):
        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        values, indices = similarity[0].topk(1)
        return values, indices

    @torch.no_grad()
    def predict(self, image: np.array) -> dict:
        '''
        Does prediction on an input image

        Args:
            image (np.array): numpy image with RGB channel ordering type.
                              Don't forget to convert image to RGB if you
                              read images via opencv, otherwise model's accuracy
                              will decrease.

        Returns:
            (dict): dict that contains predictions:
                    {
                    'label': 'some_label',
                    'confidence': 0.X
                    }
                    confidence is calculated based on cosine similarity,
                    thus you may see low conf. values for right predictions.
        '''
        tf_image = self.transform_image(image)
        image_features = self.model.encode_image(tf_image)
        values, indices = self.predict_(text_features=self.text_features,
                                        image_features=image_features)
        label_index = indices[0].cpu().item()
        label_text = self.default_label
        model_confidance = abs(values[0].cpu().item())
        if model_confidance >= self.threshold:
            label_text = self.labels[label_index]

        prediction = {
            'label': label_text,
            'confidence': model_confidance
        }

        return prediction

    @staticmethod
    def plot_image(image: np.array, title_text: str):
        plt.figure(figsize=[13, 13])
        plt.title(title_text)
        plt.axis('off')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)

#------------------------------------------------------


def save_img(frame, label_text):
    frame = cv2.putText(frame, label_text.title(), 
                    (0, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 0, 255), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (800, 400))
    return frame

def getTime():
    IST = pytz.timezone('Asia/Kolkata')
    timeNow = dt.now(IST)
    return timeNow 

