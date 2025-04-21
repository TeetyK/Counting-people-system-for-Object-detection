import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
from datetime import datetime
from tensorflow.lite.python.interpreter import Interpreter ,load_delegate # tensorflow 2.8.0
import tensorflow as tf
from deep_sort_realtime.deepsort_tracker import DeepSort
from threading import Thread , Lock
import os
import numpy as np
from random import randint
#from ultralytics import YOLO
class MyPerson:
    tracks = []
    def __init__(self, i, xi, yi , wi , hi , max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False
        self.state = '0'
        self.state_walk = '0'
        self.moving_state = ''
        self.going = ''
        self.state_going = False
        self.age = 0
        self.max_age = max_age
        self.dir = None
        # add
        self.h = hi
        self.w = wi
    def getRGB(self):
        return (self.R,self.G,self.B)
    def getTracks(self):
        return self.tracks
    def getId(self):
        return self.i
    def getState(self):
        return self.state
    def getDir(self):
        return self.dir
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getH(self):
        return self.h
    def getW(self):
        return self.w
    def getDone(self):
        return self.done
    def getMstate(self):
        return self.moving_state
    def getGstate(self):
        return self.state_going
    def getGoing(self):
        return self.going
    def set_GoingUp(self):
        self.going = "up"
    def set_GoingDown(self):
        self.going = "down"
    def setState(self):
        self.state_going = True
    def updateCoords(self, xn, yn , wn , hn):
        self.age = 0
        self.tracks.append([self.x,self.y])
        self.x = xn
        self.y = yn
        self.h = hn
        self.w = wn
        if len(self.tracks) >= 2:
            if self.tracks[-1][1] < self.tracks[-2][1]:  # ถ้า y ลดลง แสดงว่าเดินขึ้น
                self.moving_state = "up"
            elif self.tracks[-1][1] > self.tracks[-2][1]:  # ถ้า y เพิ่มขึ้น แสดงว่าเดินลง
                self.moving_state = "down"
    def setDone(self):
        self.done = True
    def timedOut(self):
        return self.done
    def going_UP(self,mid_start,mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end and self.going =="up": #cruzo la linea
                    state = '1'
                    self.dir = 'up'
                    return True
            else:
                return False
        else:
            return False
    def going_DOWN(self,mid_start,mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start and self.going=="down": #cruzo la linea
                    state = '1'
                    self.dir = 'down'
                    return True
            else:
                return False
        else:
            return False
    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True
    def reversed_direction(self, mid_start, mid_end):
        if len(self.tracks) >= 2:
            if self.dir == 'up' and self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start:
                self.dir = 'down'
                return 'down'
            elif self.dir == 'down' and self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end:
                self.dir = 'up'
                return 'up'
        return None
class MultiPerson:
    def __init__(self, persons, xi, yi):
        self.persons = persons
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.count_in = 0
        self.count_out =0
        self.cap = None
        modelpath="detect2.tflite"
        lblpath="labelmap.txt"
        self.video_writer = None
        self.deepsort = DeepSort(
                    max_age=5,
                    n_init=1,
                   max_iou_distance=0.9,
                    nms_max_overlap=0.6,
                    max_cosine_distance=0.4,
                    gating_only_position=True,
                    override_track_class=None,
                    half=True,
                    bgr=True,
                    nn_budget=5,
                    embedder_gpu=True,
                    )
        self.detection_thread = None
        self.detection_lock = Lock()
        self.detection_results = []
        self.labels = ["head_human"]
        self.max_p_age = 5
        # using model
        self.persons = []
        self.multi_person = MultiPerson(self.persons,0,0)
        self.max_p_age = 5
        self.already_counted = set()
        self.interpreter = Interpreter(model_path="detect2.tflite")
        self.interpreter.allocate_tensors()
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.float_input = (self.input_details[0]['dtype'] == np.float32)
        self.input_mean = 127.5
        self.input_std = 127.5
        self.cam_icon = tk.PhotoImage(file="icon\\new_cam1.png")
        self.notcam_icon = tk.PhotoImage(file="icon\\not_cam1.png")
        self.detect_icon = tk.PhotoImage(file="icon\\detect_icon.png")
        self.notdetect_icon = tk.PhotoImage(file="icon\\nondetect_icon.png")
        self.image_icon = tk.PhotoImage(file="icon\\1159798.png")
        self.nolight_icon = tk.PhotoImage(file="icon\\light_off_1.png")
        self.light_icon = tk.PhotoImage(file="icon\\light_on_1.png")

        self.label = tk.Label(window, image=self.image_icon, width= 1200, height=720)
        self.label.pack(expand=False, fill="both",anchor="n")

        self.label_clock = tk.Label(window, text="", font=("Arial", 20))
        self.label_clock.place(x=10, y=10)

        self.label_light = tk.Label(window,image=self.nolight_icon,width=85,height=68)
        self.label_light.place(x=1150,y=700)
    
        self.btn_toggle_camera = tk.Button(window, image=self.cam_icon, command=self.toggle_camera, width=85, height=68)
        self.btn_toggle_camera.place(x=1250, y=700)
        #self.btn_toggle_camera.pack(side="right", padx=50, pady=50)
        
        self.btn_toggle_detecting = tk.Button(window, image=self.detect_icon, command=self.toggle_detecting, width=85, height=68)
        self.btn_toggle_detecting.place(x=1350, y=700)
        #self.btn_toggle_detecting.pack(side="right", padx=50, pady=50)
        
        self.detecting_enabled = False
        
        self.label_countin = tk.Label(window, text="Count IN: ", font=("Arial", 25))
        self.label_countin.place(x=10, y=700)

        self.label_countout = tk.Label(window, text= "Count OUT: ", font=("Arial", 25))
        self.label_countout.place(x=250, y=700)

        self.label_diff = tk.Label(window, text= "Difference = ", font=("Arial", 25))
        self.label_diff.place(x=10, y=750)
        
        
        self.label1 = tk.Label(text="Camera: Off", font=("Arial", 20))
        self.label1.place(x=640, y=700)

        self.label2 = tk.Label(text="Detection: Off", font=("Arial", 20))
        self.label2.place(x=640, y=750)
        

        self.update()
        self.update_clock()
        self.window.mainloop()
    def update_clock(self):
        str_current_time = datetime.now().strftime("%H:%M:%S")
        self.label_clock.config(text=str_current_time)
        self.window.after(1000, self.update_clock)

    def toggle_camera(self):
        if self.cap is None:
            self.frame_rate_calc = 1
            self.freq = cv2.getTickFrequency()
            self.cap = VideoStream(resolution=(1290,670),framerate=30).start()
            self.frame_width = 1290  # Set your desired width
            self.frame_height = 670  # Set your desired height
            self.btn_toggle_camera.config(image=self.notcam_icon, width=85, height=68)
            self.label1.config(text="Camera: On")
            # initialize line
            self.line_up = int((2*(self.frame_height/5)) - 200)
            self.line_down = int((3*(self.frame_height/5)) + 200)
            self.up_limit = int((self.frame_height/2))
            self.down_limit = int((self.frame_height/2))
            self.line_down_color = (255,0,0)
            self.line_up_color = (0,0,255)
            self.pt1 = [0,self.line_down]
            self.pt2 = [self.frame_width,self.line_down]
            self.pts_L1 = np.array([self.pt1,self.pt2], np.int32)
            self.pts_L1 = self.pts_L1.reshape((-1,1,2))
            self.pt3 = [0,self.line_up]
            self.pt4 = [self.frame_width,self.line_up]
            self.pts_L2 = np.array([self.pt3,self.pt4],np.int32)
            self.pts_L2 = self.pts_L2.reshape((-1,1,2))
            self.pt5 =  [0, self.up_limit]
            self.pt6 =  [self.frame_width, self.up_limit]
            self.pts_L3 = np.array([self.pt5,self.pt6], np.int32)
            self.pts_L3 = self.pts_L3.reshape((-1,1,2))
            self.pt7 =  [0, self.down_limit]
            self.pt8 =  [self.frame_width, self.down_limit]
            self.pts_L4 = np.array([self.pt7,self.pt8], np.int32)
            self.pts_L4 = self.pts_L4.reshape((-1,1,2))
        else:
            self.cap.stop()
            self.cap = None
            self.label.config(image=self.image_icon, width= 1200, height=720)
            self.btn_toggle_camera.config(image=self.cam_icon, width=85, height=68)
            self.label1.config(text="Camera: Off")
            


    def toggle_detecting(self):
        self.detecting_enabled = not self.detecting_enabled # เปลี่ยนสถานะการตรวจจับวัตถุ
        if self.detecting_enabled:
            self.btn_toggle_detecting.config(image=self.notdetect_icon, width=85, height=68) # เปลี่ยนข้อความปุ่มเป็น "ปิดตรวจจับ"
            self.label2.config(text="Detection: On")
        else:
            self.btn_toggle_detecting.config(image=self.detect_icon, width=85, height=68) # เปลี่ยนข้อความปุ่มเป็น "เปิดตรวจจับ"
            self.label2.config(text="Detection: Off")

    def update(self):
        if self.cap is not None: # ตรวจสอบว่ากล้องถูกเปิดหรือไม่
            t1 = cv2.getTickCount()
            frame = self.cap.read()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imH, imW, _ = frame.shape
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)
            if self.detecting_enabled: # ตรวจสอบว่าต้องการทำการตรวจจับหรือไม่
                #if self.detection_thread is None or not self.detection_thread.is_alive():
                    #self.detection_thread = Thread(target=self.process_detection,args=(frame,imH,imW,input_data))
                    #self.detection_thread.start()
                self.process_detection(frame,imH,imW,input_data)
                #self.display_detections(frame)
            cv2.polylines(frame, [self.pts_L1], False, self.line_down_color, 2)
            cv2.polylines(frame, [self.pts_L2], False, self.line_up_color, 2)
            cv2.polylines(frame, [self.pts_L3], False, (255,255,255), 2)
            cv2.polylines(frame, [self.pts_L4], False, (0,0,0), 2)
            cv2.putText(frame,'FPS: {0:.2f}'.format(self.frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.label.config(image=photo)
            self.label.image = photo
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/self.freq
            self.frame_rate_calc = 1/time1
            if self.video_writer is not None: # ตรวจสอบว่ากล้องถูกเปิดหรือไม่
                self.video_writer.write(frame)
        self.window.after(15, self.update)
    def process_detection(self,frame,imH,imW,input_data):
        if self.float_input:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        detections = []
        for i in range(len(scores)):
            if ((scores[i] > 0.3) and (scores[i] <= 1.0)):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                detections.append([[xmin,ymin,xmax - xmin,ymax - ymin],scores[i],int(classes[i])])
                #cv2.rectangle(frame,(int(xmin),int(ymin)), (int(xmax),int(ymax)),(0,255,0), 2)
        tracks = self.deepsort.update_tracks(detections,frame=frame)
        self.display_detections(frame,tracks)
        #with self.detection_lock:
            #self.detection_results = tracks
    def display_detections(self,frame,tracks):
        #with self.detection_lock:
            #tracks = self.detection_results
        for track in tracks:
                if not track.is_confirmed():
                    continue
                bbox = track.to_tlbr()
                track_id = track.track_id
                state = ""
                going = ""
                new = True
                for p in self.multi_person.persons:
                    if p.getId() == track_id:
                        state = p.getMstate()
                        going = p.getGoing()
                        label = f"ID: {track_id} : {state} , {going}"
                        labelSize , baseLine = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
                        label_ymin = max(int(bbox[1]),labelSize[1] +10)
                        cv2.rectangle(frame , (int(bbox[0]),label_ymin - labelSize[1] - 10) , (int(bbox[0]) + labelSize[0] , label_ymin + baseLine -10) , (255,255,255) , cv2.FILLED)
                        text_position = (int(bbox[0]), label_ymin - 7)  # Position above the box
                        cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])),(0,255,0), 2)
                        if abs(int(bbox[0])) <= self.frame_width and abs(int(bbox[1])) <= self.frame_height:
                            new = False
                            p.updateCoords(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
                            if p.getGstate() == False:
                                p.setState()
                                temp = p.getTracks()
                                if temp[0][1] <= self.down_limit:
                                    p.set_GoingDown()
                                elif temp[0][1] >= self.up_limit:
                                    p.set_GoingUp()
                            if track_id not in self.already_counted:
                                if p.going_UP(self.line_down,self.line_up):
                                    self.already_counted.add(track_id)
                                    self.count_out+=1
                                    # print(f"ID: {track_id} crossed going up at {time.strftime('%c')} : count out [{count_out}]")
                                elif p.going_DOWN(self.line_down,self.line_up):
                                    self.already_counted.add(track_id)
                                    self.count_in+=1
                                    # print(f"ID: {track_id} crossed going down at {time.strftime('%c')} : count in [{count_in}]")
                            reversed_dir = p.reversed_direction(self.line_down, self.line_up)
                            if reversed_dir == 'up' and track_id in self.already_counted:
                                self.count_out += 1
                                # print(f"ID: {track_id} reversed direction to down at {time.strftime('%c')} : count out [{count_in}] s:{state}")
                            elif reversed_dir == 'down' and track_id in self.already_counted:
                                self.count_in += 1
                                # print(f"ID: {track_id} reversed direction to up at {time.strftime('%c')} : count in [{count_out}] s:{state}")
                            if p.getDir() == 'down' and p.getY() > self.down_limit:
                                p.setDone()
                            elif p.getDir() == 'up' and p.getY() < self.up_limit:
                                p.setDone()
                        if p.timedOut():
                            index = self.multi_person.persons.index(p)
                            self.multi_person.persons.pop(index)
                            del p
                if new:
                    new_tracker = MyPerson(track_id,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),self.max_p_age)
                    self.multi_person.persons.append(new_tracker)
               
        self.update_count_label()
    def update_count_label(self):
        incount_label = f"Count IN: {self.count_in}"
        outcount_label = f"Count OUT: {self.count_out}"
        self.label_countin.config(text=incount_label, font=("Arial", 25))
        self.label_countout.config(text=outcount_label, font=("Arial", 25))
        diff = (self.count_in - self.count_out)
        diff_label = f"Difference = {diff}"
        self.label_diff.config(text=diff_label, font=("Arial", 25))
        if diff > 0:
            self.label_light.config(image=self.light_icon, width=85, height=68)
        else:
            self.label_light.config(image=self.nolight_icon, width=85, height=68)
# Create Tkinter window
root = tk.Tk()
# Center the window
window_width = 1024
window_height = 768
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width / 2) - (window_width / 2)
y_coordinate = (screen_height / 2) - (window_height / 2)
root.geometry(f'{window_width}x{window_height}+{int(x_coordinate)}+{int(y_coordinate)}')

"""root.attributes('-fullscreen', True)"""



app = CameraApp(root, "กล้อง")



