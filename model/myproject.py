import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import os
import shutil
import pywt
import json
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sb
import joblib

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

def get_croppes_img(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

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
    if entry.is_dir() and entry.name != "test_img":
        img_dirs.append(entry.path)

print(img_dirs)

if os.path.exists(cr_path):
    shutil.rmtree(cr_path)
os.mkdir(cr_path)

cropped_img_dir = []
celeb_name_dict = {}

for img_dir in img_dirs:
    count = 1
    celeb_name = img_dir.split("/")[-1]
    celeb_name_dict[celeb_name] = []
    for entry in os.scandir(img_dir):
        roi_color = get_croppes_img(entry.path)
        if roi_color is not None:
            cropped_img_dir.append(roi_color)
            cropped_folder = os.path.join(cr_path,celeb_name)
            if not os.path.exists(cropped_folder):
                os.mkdir(cropped_folder)
                cropped_img_dir.append(cropped_folder)
                print(f"We're generating the {cropped_folder} folder...")
            cropped_name = celeb_name + str(count) + ".png"
            cropped_file_path = cropped_folder + "/" + cropped_name

            cv2.imwrite(cropped_file_path,roi_color)
            celeb_name_dict[celeb_name].append(cropped_file_path)
            count = count + 1
#source stackoverflow
def w2d(img, mode='haar', level=1):
    imArray = img

    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )

    imArray =  np.float32(imArray)   
    imArray /= 255

    coeffs=pywt.wavedec2(imArray, mode, level=level)

    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0; 

    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    return imArray_H
img_har = w2d(img)

# plt.imshow(img_har,cmap="gray")
# plt.show()

celebrity_file_names_dict = {}
for entry in os.scandir(cr_path):
    if entry.is_dir():
        celebrity_name = entry.name
        file_list = [f.path for f in os.scandir(entry.path) if f.is_file()]
        celebrity_file_names_dict[celebrity_name] = file_list

for celeb, files in celebrity_file_names_dict.items():
    print(f"{celeb}: {len(files)} imagini")

class_dict = {}
countul = 0
for celeb_names in celebrity_file_names_dict.keys():
    class_dict[celeb_names] = countul
    countul = countul + 1

x = []
y = []
for celeb_name, train_files in celebrity_file_names_dict.items():
    for train_file in train_files:
        img = cv2.imread(train_file)
        scale_img = cv2.resize(img,(32,32))
        img_har = w2d(img, "db1",5)
        scale_img_har = cv2.resize(img_har,(32,32))
        combined_img = np.vstack((scale_img.reshape(32*32*3,1), scale_img_har.reshape(32*32,1)))
        x.append(combined_img)
        y.append(class_dict[celeb_name])

#print(class_dict)
#print(len(x[0]))
x = np.array(x).reshape(len(x),4096).astype(float)
# print(x.shape)
# print(x[0])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,stratify=y, random_state=42)
# model = SVC(kernel="rbf", C=10)
# model.fit(x_train,y_train)
# print(model.score(x_test,y_test))
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf",C=10))])
pipe.fit(x_train,y_train)
print(pipe.score(x_test,y_test))
print(len(x_test))
print(classification_report(y_test, pipe.predict(x_test)))
for celeb_namee, label in class_dict.items():
    print(f"{celeb_namee} -- {label}")


model_params = {
    "svm":{
        "model": SVC(gamma="auto",probability=True),
        "params": {
            "svc__C": [1,10,100,1000],
            "svc__kernel": ["rbf","linear"],
        }
    
    },
    "random forest":{
        "model": RandomForestClassifier(),
        "params":{
            "randomforestclassifier__n_estimators": [1,5,10],
        }
    },
    "logistic regresion":{
        "model": LogisticRegression(solver="liblinear", multi_class="auto"),
        "params":{
            "logisticregression__C": [1,5,10],
        }
    },
}
scores = []
best_estimator = {}
for modell, param in model_params.items():
    pipe = make_pipeline(StandardScaler(), param["model"])
    model2 = GridSearchCV(pipe,param["params"],cv=5, return_train_score=False)
    model2.fit(x_train,y_train)
    scores.append({
        "model": modell,
        "best_score": model2.best_score_,
        "best_params": model2.best_params_
    })
    best_estimator[modell] = model2.best_estimator_

df = pd.DataFrame(scores, columns=["model","best_score","best_params"])
print(df.head(5))
print(best_estimator["logistic regresion"].score(x_train,y_train))
print(best_estimator["svm"].score(x_train,y_train))
print(best_estimator["random forest"].score(x_train,y_train))

best = best_estimator["logistic regresion"]
cm = confusion_matrix(y_test, best.predict(x_test))
plt.figure(figsize=(10,7))
sb.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

joblib.dump(best,"saved.model.pkl")
with open("class_dict","w") as f:
    f.write(json.dumps(class_dict))








