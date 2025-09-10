import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os
import shutil

img = cv2.imread("W:/vscode/Machine-Learning/MLPROJECTClassification/model/dataset/test_img/17038722333525.jpg")
#print(img.shape)

#plt.imshow(img)
#plt.show()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#plt.imshow(gray, cmap="gray")
#plt.show()

face_cascade = cv2.CascadeClassifier("W:/vscode/Machine-Learning/MLPROJECTClassification/model/opencv/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("W:/vscode/Machine-Learning/MLPROJECTClassification/model/opencv/haarcascade_eye.xml")

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#print(faces)

(x,y,w,h) = faces[0]
#print(x,y,w,h)

face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0))
#plt.imshow(face_img)
#plt.show()

# for (x,y,w,h) in faces:
#     face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = face_img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

# plt.figure()
# # plt.imshow(roi_color, cmap='gray')
# # plt.show()

def get_croppes_img(imagie_path):
    img = cv2.imread(imagie_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = face_img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        if len(eyes) >= 2:
            return roi_color
    else:
        return None

#get_croppes_img("W:/vscode/Machine-Learning/MLPROJECTClassification/model/dataset/MESSI/16192430-f5ef-11ef-bd6e-cd71c2e1454a.jpg")
#cropped_img = get_croppes_img("W:/vscode/Machine-Learning/MLPROJECTClassification/model/dataset/MESSI/notgood.jpg")
# plt.imshow(cropped_img, cmap="gray")
# plt.show()

current_path = "W:/vscode/Machine-Learning/MLPROJECTClassification/model/dataset/"
cr_path = "W:/vscode/Machine-Learning/MLPROJECTClassification/model/dataset/cropped"
img_dirs = []

for entry in os.scandir(current_path):
    if entry.is_dir():
        img_dirs.append(entry.path)

print(img_dirs)

if os.path.exists(cr_path):
    shutil.rmtree(cr_path)
os.mkdir(cr_path)

cropped_img_dir = []
celeb_name_dict = {}

for img_dir in img_dirs:
    celeb_name = img_dir.split("/")[-1]

    for entry in os.scandir(img_dir):
        roi_color = get_croppes_img(entry.path)
        if roi_color is not None:
            cropped_img_dir.append(roi_color)
            cropped_folder = os.path.join(cr_path,celeb_name)
            if not os.path.exists(cropped_folder):
                os.mkdir(cropped_folder)
                print(f"We're generating the {cropped_folder} folder...")
    


