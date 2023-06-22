import akida
import cv2
import math
import time
import signal
import threading
import numpy as np
from queue import Queue
from scipy.special import softmax
from flask import Flask, render_template, Response
from edge_impulse_linux.image import ImageImpulseRunner

app = Flask(__name__, static_folder='templates/assets')
        
EI_CLASSIFIER_INPUT_WIDTH  = 192
EI_CLASSIFIER_INPUT_HEIGHT = 192
EI_CLASSIFIER_LABEL_COUNT = 4
EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD = 0.95
categories = ['ac','tv','light','other']
inference_speed = 0
power_consumption = 0

videoCaptureDeviceId = int(0) # use 0 for web camera

def capture(queueIn):


    while True:
        cap = cv2.VideoCapture(videoCaptureDeviceId)
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        resize_dim = (EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT)

        ret, frame = cap.read()

        if ret:
            backendName = "dummy" #backendName = camera.getBackendName() this is fixed in opencv-python==4.5.2.52
            w = cap.get(3)
            h = cap.get(4)
            print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
            cap.release()
            #cropped_img = frame[0:720, 280:280+720]
            #resized_img = cv2.resize(frame, resize_dim, interpolation = cv2.INTER_AREA)
            resized_img = cv2.resize(frame, resize_dim)
            img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(img, axis=0)
            if not queueIn.full():
                queueIn.put((frame, input_data))
        else:
            return


def inferencing(model_file, queueIn, queueOut):
    akida_model = akida.Model(model_file)
    devices = akida.devices()
    print(f'Available devices: {[dev.desc for dev in devices]}')
    device = devices[0]
    device.soc.power_measurement_enabled = True
    akida_model.map(device)
    akida_model.summary()
    i_h, i_w, i_c = akida_model.input_shape
    o_h, o_w, o_c = akida_model.output_shape
    scale_x = int(i_w/o_w)
    scale_y = int(i_h/o_h)
    scale_out_x = 1280/EI_CLASSIFIER_INPUT_WIDTH
    scale_out_y = 1280/EI_CLASSIFIER_INPUT_HEIGHT

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

        pred = softmax(logits, axis=-1).squeeze()

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
        

def gen_frames():
    while True:
        with ImageImpulseRunner(model_file) as runner:
            try:
                model_info = runner.init()
                print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
                labels = model_info['model_parameters']['labels']
                
                camera = cv2.VideoCapture(videoCaptureDeviceId)
                ret = camera.read()[0]
                
                if ret:
                    backendName = "dummy" #backendName = camera.getBackendName() this is fixed in opencv-python==4.5.2.52
                    w = camera.get(3)
                    h = camera.get(4)
                    print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
                    camera.release()
                else:
                    raise Exception("Couldn't initialize selected camera.")
                
                next_frame = 0 # limit to ~10 fps here
                
                for res, img in runner.classifier(videoCaptureDeviceId):
                    count = 0
                    
                    if (next_frame > now()):
                        time.sleep((next_frame - now()) / 1000)

                    # print('classification runner response', res)

                    if "classification" in res["result"].keys():
                                                print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                        for label in labels:
                            score = res['result']['classification'][label]
                            print('%s: %.2f\t' % (label, score), end='')
                        print('', flush=True)

                    elif "bounding_boxes" in res["result"].keys():
                        # print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                        countPeople = len(res["result"]["bounding_boxes"])
                        # inferenceSpeed = res['timing']['classification']
                        for bb in res["result"]["bounding_boxes"]:
                            # print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                            img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (0, 0, 255), 2)
                        
                    ret, buffer = cv2.imencode('.jpg', img)
                    #buffer = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)

                    #/////////////////////////////////////////////////////////////

                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

                    next_frame = now() + 100
                    
            finally:
                if (runner):
                    runner.stop()
    
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
    #video_file = './video/aerial_1280_1280.avi'
    #model_file = './model/ei-object-detection-metatf-model.fbz'
    model_file = './model/akida_model.fbz'
    app.run(host="0.0.0.0", debug=True)

