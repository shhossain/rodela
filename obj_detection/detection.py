import numpy as np
import cv2
import os

thres = 0.5  # Threshold to detect object
nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress , 0.1 means high suppress
classNames = []
current_file_dir = os.path.dirname(__file__)
coco_file_path = os.path.join(current_file_dir,"coco.names")
with open(coco_file_path, 'r') as f:
    classNames = f.read().splitlines()



# font = cv2.FONT_HERSHEY_PLAIN
font = cv2.FONT_HERSHEY_COMPLEX
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = os.path.join(current_file_dir,"frozen_inference_graph.pb")
configPath = os.path.join(current_file_dir,"ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")

# now create a class
class ObjectModel:
    def __init__(self, classNames=classNames, weightsPath=weightsPath, configPath=configPath, inputSize=(320, 320), scale=1.0/127.5, mean=(127.5, 127.5, 127.5), swapRB=True):
        self.classNames = classNames
        self.weightsPath = weightsPath
        self.configPath = configPath

        self.model = cv2.dnn_DetectionModel(self.weightsPath, self.configPath)
        self.model.setInputSize(inputSize[0], inputSize[1])
        self.model.setInputScale(scale)
        self.model.setInputMean(mean)
        self.model.setInputSwapRB(swapRB)

    def detect(self, img, confThreshold=0.5, nmsThreshold=0.2):
        return self.model.detect(img, confThreshold, nmsThreshold)


def detect_video(video, model):
    cap = cv2.VideoCapture(video)
    while True:
        success, img = cap.read()
        classIndex, confidence, bbox = model.detect(img, confThreshold=thres)
        if len(classIndex) != 0:
            for classInd, boxes in zip(classIndex.flatten(), bbox):
                cv2.rectangle(img, boxes, Colors[classInd - 1], 2)
                cv2.putText(img, classNames[classInd - 1].upper(), (boxes[0] + 10, boxes[1] + 40), font, 1, Colors[classInd - 1], 2)
        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def detect_image(image, model):
    img = image
    if img is str:
        img = cv2.imread(image)
    # img = cv2.imread(image)
    classIndex, confidence, bbox = model.detect(img, confThreshold=thres)
    if len(classIndex) != 0:
        for classInd, boxes in zip(classIndex.flatten(), bbox):
            cv2.rectangle(img, boxes, Colors[classInd - 1], 2)
            cv2.putText(img, classNames[classInd - 1].upper(), (boxes[0] + 10, boxes[1] + 40), font, 1, Colors[classInd - 1], 2)
    cv2.imshow('Output', img)
    


if __name__ == "__main__":
    model = ObjectModel(classNames, weightsPath, configPath)
    print(model)
    # print(weightsPath,configPath)
    
