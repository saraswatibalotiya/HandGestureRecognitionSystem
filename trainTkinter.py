import cv2
import numpy as np
from  tkinter import  *
from PIL import  Image,ImageTk
import datetime
import pickle
import cv2
import mediapipe as mp
import numpy as np

root = Tk()
root.geometry("700x640")
root.configure(bg="black")
Label(root,text="Saras Camera",font=("times new roman",30,"bold"),bg="black",fg="red").pack()
f1 = LabelFrame(root,bg="red")
f1.pack()
L1 = Label(f1,bg="red")
L1.pack()

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
Button(root,text="Take Snapshot",font=("times new roman",30,"bold"),bg="black",fg="red").pack()

# while True:
#     img=cap.read()[1]
#     img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img = ImageTk.PhotoImage(Image.fromarray(img1))
#     L1['image'] = img
#     root.update()

while True:
    ret , img=cap.read()
    H, W, _ = img.shape
    # img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # img = ImageTk.PhotoImage(Image.fromarray(img1))
    # # H, W, _ = img.shape
    # L1['image'] = img


    data_aux = []
    x_ = []
    y_ = []

    # ret, frame = cap.read()



    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,  # image to draw
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
                # data_aux.append(y)
                # x_.append(x)
                # y_.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # def predict(data_aux):
        #     return
        print(data_aux)
        print(np.asarray(data_aux))
        prediction = model.predict([np.asarray(data_aux)])
        print(prediction," : Model Prediction")
        print(prediction[0] ," custome")

        # predicted_character = labels_dict[(prediction[0])]
        # print(predicted_character , " : Label Predicted")
        predicted_character = prediction[0]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # cv2.imshow('frame', img)
    # cv2.waitKey(1)
    root.update()

cap.release()