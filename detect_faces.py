import cv2
import pickle
import face_recognition as fr
import os
import numpy as np
import argparse
import sys
import threading
import json
import time

temp_dir = "temp"
exit_file = "exit.json"
img_dir = "imgs"

ID = [None]

class SharedSpace:
    def __init__(self, filename):
        self.data = {}
        self.file_name = filename
    
    def add(self, key, val):
        self.data[key] = val
    
    def get(self, key):
        return self.data[key]

    def save(self):
        with open(self.file_name, "wb") as f:
            pickle.dump(self.data, f)
    
    def load(self):
        with open(self.file_name, "rb") as f:
            self.data = pickle.load(f)

def face_recognition(data_path,data_id,shared_space_path):
    with open(data_path, "rb") as f:
        try:
            data = pickle.load(f) # id : [img]
        except Exception as e:
            sys.exit()
    
    print("data", data.keys())
    print("data_id", data_id)
    faces = []
    known_faces = []
    known_names = []

    # imgs\known_faces_encoding.pickle
    # imgs\known_faces_names.pickle
    
    
    print("known_faces", known_names)
    
    name = "Unknown"
    face_locations = fr.face_locations(data[data_id])

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_img = data[data_id][top:bottom, left:right]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        faces.append(face_img)

    face_encodings = fr.face_encodings(data[data_id], face_locations)
    match_names = []
    got_encodings = False
    for face_encoding in face_encodings:
        got_encodings = True
        matches = fr.compare_faces(known_faces, face_encoding)
        distance = fr.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(distance)
        if matches[best_match_index]:
            name = known_names[best_match_index]
        else:
            name = "Unknown"
        match_names.append(name)

    shared_space = SharedSpace(shared_space_path)
    shared_space.add("names", match_names)
    shared_space.add("got_encodings", got_encodings)
    shared_space.save()
    
    sys.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="0")
    parser.add_argument("--data_path", type=str, default="data.pkl")
    parser.add_argument("--data_id", type=str, default="0")
    parser.add_argument("--shared_space_path", type=str, default="shared_space.pkl")
    args = parser.parse_args()

    ID[0] = args.id
    face_recognition(args.data_path, args.data_id, args.shared_space_path)

def wait_for_exit():
    while True:
        with open(exit_file, "r") as f:
            exit_data = json.load(f)["exit"]
        
        if ID[0] in exit_data:
            sys.exit()
        time.sleep(1)

if __name__ == "__main__":
    t = threading.Thread(target=main)
    t.start()
    
    t2 = threading.Thread(target=wait_for_exit)
    t2.start()
    
    t.join()
    t2.join()