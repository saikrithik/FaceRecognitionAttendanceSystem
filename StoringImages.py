import cv2,os              
import numpy as np          
WHITE = [255, 255, 255]


face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

name=input('Enter Name')
roll_no=input('Enter Roll no for recognition')
Count = 0
cap = cv2.VideoCapture(0) 
path="images/"
#os.mkdir(path)

while Count < 4:
    ret, img = cap.read()
                                                                     
    #cv2.imwrite("images/"+name+"."+roll_no+'.'+str(Count)+".jpg", img)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        temp=input("Enter any Key and press enter to capture: ")
        cv2.waitKey(500)
        cv2.imwrite("images/"+name+'.'+str(Count)+".jpg", img)
        cv2.imshow("CAPTURED PHOTO", img)                                                     
        Count = Count + 1
        cv2.imshow('Face Recognition System Capture Faces', img)
               
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print ('FACE CAPTURE FOR THE SUBJECT IS COMPLETE')
cap.release()
cv2.destroyAllWindows()
