import cv2
import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from datetime import datetime
from tkinter import messagebox, filedialog
import os
import numpy as np
import mediapipe as mp


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_center(self):
        return self.x + self.w // 2, self.y + self.h // 2
    
    def get_area(self):
        return self.w * self.h

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_confidence)
        
    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bboxes = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, ic = image.shape
                bbox = Rect(int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih))
                bboxes.append(bbox)
                self.mp_drawing.draw_detection(image, detection)
        return image, bboxes
    

    def fancyDrawing(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(imgRGB)
        if results.detections:
            for id, detection in enumerate(results.detections):
                # mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, bbox, (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 0, 255), 2)
        return img


detector = FaceDetector()

root = tk.Tk()

class App:
    def __init__(self,root,detector):
        self.root = root
        self.detector = detector
        self.root.title("Capture Image")
        self.root.geometry("1340x700")
        self.root.resizable(True, True)
        self.root.configure(background = "sky blue")
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.dest_path = os.path.join(self.current_dir, "images")
        self.cap = None
    
    def run(self):
        self.create_widgets()
        self.show_frame()
        self.root.mainloop()
    
    def create_widgets(self):
        root = self.root
        root.feedlabel = Label(root, text = "Feed", font = ("arial", 20, "bold"), bg = "sky blue")
        root.feedlabel.grid(row = 0, column = 0, padx = 10, pady = 10)
        
        root.cameralabel = Label(root, text = "Camera", font = ("arial", 20, "bold"), bg = "sky blue")
        root.cameralabel.grid(row = 0, column = 1, padx = 10, pady = 10)
        
        root.capturelabel = Label(root, text = "Capture", font = ("arial", 20, "bold"), bg = "sky blue")
        root.capturelabel.grid(row = 0, column = 2, padx = 10, pady = 10)
        
        
        root.previewlabel = Label(root, text = "Preview", font = ("arial", 20, "bold"), bg = "sky blue")
        root.previewlabel.grid(row = 0, column = 3, padx = 10, pady = 10)

        root.feedframe = Frame(root, width = 400, height = 400, bg = "white")
        root.feedframe.grid(row = 1, column = 0, padx = 10, pady = 10)

    def show_frame(self):
        _, frame = self.cap.read()
        frame = self.detector.fancyDrawing(frame)
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        self.root.feedframe.imgtk = frame
        self.root.feedframe.configure(image=frame)
        self.root.feedframe.after(10, self.show_frame)
    
    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
    
    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()
    
    def capture_image(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        self.root.captureframe.imgtk = frame
        self.root.captureframe.configure(image=frame)
    
    def save_image(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.dest_path, "image.jpg"), frame)
    
    def preview_image(self):
        image = cv2.imread(os.path.join(self.dest_path, "image.jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image=image)
        self.root.previewframe.imgtk = image
        self.root.previewframe.configure(image=image)
        
        
        
        
        


if __name__ == "__main__":
    app = App(root,detector)
    app.open_camera()
    app.run()