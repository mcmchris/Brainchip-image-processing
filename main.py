import akida
import cv2
import math
import time
import os
import signal
import threading
import numpy as np
from queue import Queue
from scipy.special import softmax
from flask import Flask, render_template, Response

app = Flask(__name__, static_folder='templates/assets')
        
EI_CLASSIFIER_INPUT_WIDTH  = 192
EI_CLASSIFIER_INPUT_HEIGHT = 192
EI_CLASSIFIER_LABEL_COUNT = 4
EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD = 0.95
categories = ['ac','tv','light','other']
inference_speed = 0
power_consumption = 0


runner = None
countPeople = 0
inferenceSpeed = 0
videoCaptureDeviceId = int(0) # use 0 for web camera


def capture(video_file,queueIn):

    cap = cv2.VideoCapture(videoCaptureDeviceId)
    resize_dim = (EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT)

    while True:
        
        ret, frame = cap.read()[0]

        if ret:
            #cropped_img = frame[0:720, 280:280+720]
            #resized_img = cv2.resize(frame, resize_dim, interpolation = cv2.INTER_AREA)
            backendName = "dummy" #backendName = camera.getBackendName() this is fixed in opencv-python==4.5.2.52
            w = cap.get(3)
            h = cap.get(4)
            print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
            cap.release()

            resized_img = cv2.resize(frame, resize_dim)
            img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

            
            input_data = np.expand_dims(img, axis=0)
            if not queueIn.full():
                queueIn.put((frame, input_data))
        else:
            return
            #raise Exception("Couldn't initialize selected camera.")


def inferencing(model_file, queueIn, queueOut):
    akida_model = akida.Model(model_file)
    devices = akida.devices()
    print(f'Available devices: {[dev.desc for dev in devices]}')
    device = devices[0]
    device.soc.power_measurement_enabled = True
    akida_model.map(device)
    akida_model.summary()

    global inference_speed
    global power_consumption

    while True:
        if queueIn.empty():
            #print("queue empty, wait a while")
            time.sleep(0.01)
            continue
        img, input_data = queueIn.get()
        
        start_time = time.perf_counter()
        logits = akida_model.predict(input_data)
        end_time = time.perf_counter()
        inference_speed = (end_time - start_time) * 1000

        #pred = softmax(logits, axis=-1).squeeze()

        floor_power = device.soc.power_meter.floor
        power_events = device.soc.power_meter.events()
        active_power = 0
        for event in power_events:
            active_power += event.power
    
        power_consumption = f'{(active_power/len(power_events)) - floor_power : 0.2f}' 
        #print(akida_model.statistics)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if not queueOut.full():
            queueOut.put(img)
        
def now():
    return round(time.time() * 1000)

def gen_frames():
    while True:
        if queueOut.empty():
            time.sleep(0.01)
            continue
        img = queueOut.get()
        ret, buffer = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
    
def get_inference_speed():
    while True:
        yield f"data:{inference_speed:.2f}\n\n"
        time.sleep(0.1)

def get_power_consumption():
    while True:
        yield "data:" + str(power_consumption) + "\n\n"
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/model_inference_speed')
def model_inference_speed():
	return Response(get_inference_speed(), mimetype= 'text/event-stream')

@app.route('/model_power_consumption')
def model_power_consumption():
	return Response(get_power_consumption(), mimetype= 'text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    video_file = './video/aerial_1280_1280.avi'
    #model_file = './model/ei-object-detection-metatf-model.fbz'
    model_file = './model/akida_model.fbz'

    queueIn  = Queue(maxsize = 24)
    queueOut = Queue(maxsize = 24)
    t1 = threading.Thread(target=capture, args=(video_file,queueIn))
    t1.start()
    t2 = threading.Thread(target=inferencing, args=(model_file, queueIn, queueOut))
    t2.start()
    app.run(host="0.0.0.0", debug=False)
    t1.join()
    t2.join()
