# importing libraries
import tkinter as tk
from tkinter import messagebox
from tkinter import Message, Text

import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from pathlib import Path
import os
import pickle
import  pickle
import shutil

from PIL.ImageTk import PhotoImage
from  sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score
from tkinter import BOTH, END, LEFT

import  numpy as np

import mediapipe as mp
import matplotlib.pyplot as plt


window = tk.Tk()
window.geometry("1720x1560")

window.title("Sign-Language")
#bg = PhotoImage(file = "Group 13.png")
# Show image using label
#label1 = tk.Label(window, image = bg)
#label1.place(x = 0, y = 0)

window.configure(background='white')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(window, text="Sign-Language",bg="#0099FF", fg="black", width=50,height=3, font=('times', 30, 'bold'))
message.place(x=200, y=20)

label = tk.Label(window)
label.grid(row=0, column=0)

lbl2 = tk.Label(window, text="Name", width=20, fg="Black", bg="white",height=2, font=('times', 15, ' bold '))
lbl2.place(x=430, y=235)

txt2 = tk.Entry(window, width=20, bg="white", fg="green", font=('times', 15, ' bold '), bd =2)
txt2.place(x=620, y=250)

# ======================= functions ===============
delete_success = False
sentence = []

def clear_text():
   txt2.delete(0, END)

def collect_imgs():
    name = txt2.get()
    if len(name) == 0:
        messagebox.showerror("showerror", "Enter name of sign .")
    else:
        DATA_DIR = './data'
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        number_of_classes = 1
        dataset_size = 100
        cap = cv2.VideoCapture(0)
        # cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        name = (txt2.get())

        if not os.path.exists(os.path.join(DATA_DIR, name)):
            os.makedirs(os.path.join(DATA_DIR, name))

        # print('Collecting data for class {}'.name)
        done = False
        counter = 0
        while True:
            ret, img = cap.read()
            # success,img = cap.read()
            # cv2.imshow("image",img)
            cv2.putText(img, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow("frame", img)
            if cv2.waitKey(25) == ord('q'):
                break
        print(counter)
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, name, '{}.jpg'.format(counter)), frame)
            print("str(j)", name)
            print('format(counter) : ', format(counter))
            counter += 1
        txt2.delete(0, END)
        cap.release()
        cv2.destroyWindow('frame')

def createDataset():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    DATA_DIR = './data'

    data = []
    labels = []
    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            # vgr input image to rgb
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

    f = open('data.pickle', 'wb')
    try:
        pickle.dump({'data': data, 'labels': labels}, f)
        messagebox.showinfo("showinfo", "Dataset Created Successfully !")
        print("Dataset Created")
    except:
        messagebox.showerror("showerror", "Error in creating dataset !")
        print("Some error in creating dataset")

    f.close()
# Training the images saved in training image folder

def trainDataset():
    data_dict = pickle.load(open('./data.pickle', 'rb'))

    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, train_size=0.75, shuffle=True,
                                                        stratify=labels)

    model = RandomForestClassifier()

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = accuracy_score(y_predict, y_test)
    print('{}% of samples were classified correctly : '.format(score * 100))

    print(data_dict.keys())
    print(data_dict)

    f = open('model.p', 'wb')
    try:
        pickle.dump({'model': model}, f)
        messagebox.showinfo("showinfo","Dataset Trained Successfully ")
        print("Dataset Trained Successfully")
    except:
        messagebox.showerror("showerror","Some error in training dataset ")
        print("Some Error in training Dataset :( ")
    f.close()

def testDataset():
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    while True:

        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    xa = hand_landmarks.landmark[i].x
                    ya = hand_landmarks.landmark[i].y

                    x_.append(xa)
                    y_.append(ya)

                for j in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[j].x
                    y = hand_landmarks.landmark[j].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                    # data_aux.append(x)
                    # data_aux.append
                    # x_.append(x)
                    # y_.append

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # def predict(data_aux):
            #     return
            print(data_aux)
            print(np.asarray(data_aux))
            prediction = model.predict([np.asarray(data_aux)])
            print(prediction, " : Model Prediction")
            print(prediction[0], " custome")

            # predicted_character = labels_dict[(prediction[0])]
            # print(predicted_character , " : Label Predicted")
            predicted_character = prediction[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            sentence.append(predicted_character)
            print("Sentence : ", sentence)

        if cv2.waitKey(25) == ord('q'):
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(1)


    cap.release()
    cv2.destroyWindow('frame')

def delete_dataset():
    global delete_success
    # getting the folder path from the user
    DATA_DIR = './data'
    name = txt2.get()
    if len(name) == 0:
        messagebox.showerror("showerror", "Enter a name of sign")
    else:
        path_name = os.path.join(DATA_DIR, name)
        # checking whether folder exists or not
        if os.path.exists(path_name):
            shutil.rmtree(path_name)
            messagebox.showinfo("showinfo", "File deleted successfully ")
            delete_success = True
            txt2.delete(0, END)

            def createTrainDataset():
                mp_hands = mp.solutions.hands
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles

                hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

                DATA_DIR = './data'

                data = []
                labels = []
                for dir_ in os.listdir(DATA_DIR):
                    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
                        data_aux = []

                        x_ = []
                        y_ = []

                        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
                        # vgr input image to rgb
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        results = hands.process(img_rgb)
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                for i in range(len(hand_landmarks.landmark)):
                                    x = hand_landmarks.landmark[i].x
                                    y = hand_landmarks.landmark[i].y

                                    x_.append(x)
                                    y_.append(y)

                                for i in range(len(hand_landmarks.landmark)):
                                    x = hand_landmarks.landmark[i].x
                                    y = hand_landmarks.landmark[i].y
                                    data_aux.append(x - min(x_))
                                    data_aux.append(y - min(y_))

                            data.append(data_aux)
                            labels.append(dir_)

                f = open('data.pickle', 'wb')
                try:
                    pickle.dump({'data': data, 'labels': labels}, f)
                    print("Dataset Created")

                    data_dict = pickle.load(open('./data.pickle', 'rb'))
                    data = np.asarray(data_dict['data'])
                    labels = np.asarray(data_dict['labels'])
                    # split data into train and test
                    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, train_size=0.75,
                                                                        shuffle=True,
                                                                        stratify=labels)
                    model = RandomForestClassifier()
                    model.fit(x_train, y_train)
                    y_predict = model.predict(x_test)
                    score = accuracy_score(y_predict, y_test)
                    print('{}% of samples were classified correctly : '.format(score * 100))
                    print(data_dict.keys())
                    print(data_dict)
                    f = open('model.p', 'wb')
                    try:
                        pickle.dump({'model': model}, f)
                        messagebox.showinfo("showinfo", "Dataset Created & Trained Successfully")
                        print("Dataset Created & Trained Successfully")
                    except:
                        messagebox.showerror("showerror", "Some error in training dataset ")
                        print("Some Error in training Dataset :( ")
                    f.close()
                except:
                    messagebox.showerror("showerror", "Error in creating dataset !")
                    print("Some error in creating dataset")

                f.close()


            return  createTrainDataset()

        else:
            # file not found message
            messagebox.showerror("showerror", "File not found in the directory")


createDataset = tk.Button(window, text="Create Dataset",command=createDataset, fg="white", bg="DodgerBlue2",width=20, height=3, activebackground="Red",
                    font=('times', 15, ' bold '))
createDataset.place(x=200, y=500)
trainImg = tk.Button(window, text="Training",command=trainDataset, fg="white", bg="DodgerBlue2",width=20, height=3, activebackground="Red",
                     font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Testing",command=testDataset,fg="white", bg="DodgerBlue2",width=20, height=3, activebackground="Red",
                     font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit",command=window.destroy, fg="white", bg="DodgerBlue2",width=20, height=3, activebackground="Red",
                       font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)



takeImg = tk.Button(window, text="Take Images",command=collect_imgs, fg="white", bg="DodgerBlue2",width=15, height=3, activebackground="Red",
                    font=('times', 15, ' bold '))
takeImg.place(x=950, y=230)

deletedata = tk.Button(window, text="Delete data",command=delete_dataset, fg="white", bg="DodgerBlue2",width=15, height=3, activebackground="Red",
                       font=('times', 15, ' bold '))
deletedata.place(x=1210, y=230)


window.mainloop()
