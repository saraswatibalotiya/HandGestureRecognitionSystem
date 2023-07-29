import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1
dataset_size = 100

cap = cv2.VideoCapture(0)

name = input("Enter Character")

if not os.path.exists(os.path.join(DATA_DIR, name)):
    os.makedirs(os.path.join(DATA_DIR, name))

# print('Collecting data for class {}'.name)
done = False
while True:
    ret, img = cap.read()
    # success,img = cap.read()
    # cv2.imshow("image",img)
    cv2.putText(img, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,cv2.LINE_AA)
    cv2.imshow("frame", img)
    if cv2.waitKey(25) == ord('q'):
        break
counter = 0
print(counter)
while counter < dataset_size:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(DATA_DIR, name, '{}.jpg'.format(counter)), frame)
    print("str(j)",name)
    print('format(counter) : ',format(counter))
    counter += 1

cap.release()
cv2.destroyAllWindows()

#
# for j in range(number_of_classes):
#     if not os.path.exists(os.path.join(DATA_DIR, str(j))):
#         os.makedirs(os.path.join(DATA_DIR, str(j)))
#
#     print('Collecting data for class {}'.format(j))
#
#     done = False
#     while True:
#         ret, img = cap.read()
#         # success,img = cap.read()
#         # cv2.imshow("image",img)
#         cv2.putText(img, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,cv2.LINE_AA)
#         cv2.imshow("frame", img)
#         if cv2.waitKey(25) == ord('q'):
#             break
#
#     counter = 0
#     print(counter)
#     while counter < dataset_size:
#         ret, frame = cap.read()
#         cv2.imshow('frame', frame)
#         cv2.waitKey(25)
#         cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
#         print("str(j)",str(j))
#         print('format(counter) : ',format(counter))
#         counter += 1
#
# cap.release()
# cv2.destroyAllWindows()