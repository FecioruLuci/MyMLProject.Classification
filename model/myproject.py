import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

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
plt.imshow(face_img)
plt.show()