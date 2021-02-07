#!/usr/bin/env python
# coding: utf-8

# In[3]:


import face_recognition
from PIL import Image
import cv2
import os

ask=input('Are the GroupImages of Class are saved at "./groupphotos/example.jpg"?:\nType y for yes and n for no')
if ask=='n' or ask=='N':
    exit()

i=0
cropimage_filenames = filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), os.listdir('groupphotos/'))
cropimage_filenames = sorted(cropimage_filenames)
paths_to_cropimages = ['images/' + x for x in cropimage_filenames]


for paths_to_cropimages in cropimage_filenames:
    image=face_recognition.load_image_file("groupphotos/"+paths_to_cropimages)
    print(paths_to_cropimages)
    face_locations= face_recognition.face_locations(image)
    print("no. of faces={}".format(len(face_locations)))

    for face_location in face_locations:

        top,right,bottom,left = face_location
        face_image = image[top-50:bottom+35, left-35:right+35]
        face_image = cv2.resize(face_image,(500,500))
        pil_image = Image.fromarray(face_image)
        pil_image.save("./test/classface{}.jpg".format(i))
        i=i+1


# In[109]:


import dlib
import numpy as np
import os
import pickle
import face_recognition
import cv2

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('Shape_Predictor_trainedData.dat')
face_recognition_model=dlib.face_recognition_model_v1('Face_Recognition_trainedData.dat')
scale_factor = 0.45

with open ('test_encodes.dat', 'rb') as fp:
    face_encodings2 = pickle.load(fp)

def compare_face_encodings(known_faces, face):
    thres_hold = np.linalg.norm(known_faces - face, axis=1)
    #print(type(thres_hold))
    return (thres_hold <= TOLERANCE)

def find_match(known_faces, names, face):
    matches = face_recognition.compare_faces(known_faces, face)
    
    face_distances = face_recognition.face_distance(known_faces, face)
    best_match_index = np.argmin(face_distances)
    #print(best_match_index)
    if matches[best_match_index]:
        if(face_distances[best_match_index]<scale_factor):
            name = names[best_match_index]
            return(name)
    return 'Not Found'



image_filenames = filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), os.listdir('images/'))
image_filenames = sorted(image_filenames)
paths_to_images = ['images/' + x for x in image_filenames]
face_encodings = []
test_filenames = filter(lambda x: x.endswith('.jpg')or x.endswith('.png'), os.listdir('test/'))
paths_to_test_images = ['test/' + x for x in test_filenames]
names = [x[:-6] for x in image_filenames]
present=[]


for path_to_image in paths_to_test_images:
    try:
        #print(path_to_image)
        # Get face encodings from the test image
        testimage =face_recognition.load_image_file(path_to_image)
        face_encodings_in_image = [np.array(face_recognition.face_encodings(testimage))]
        match = find_match(face_encodings2, names, face_encodings_in_image[0])

        if match!='Not Found' and match not in present:
            present.append(match)
        
        print(path_to_image, match)
    except:
        pass

print("The presenties are\n {}".format(present))
#try:
#    present = sorted(present)
#except BaseException:
#    print("sorted")


# In[108]:





# In[110]:

import pandas as pd
df2 = pd.read_excel(r'attendance.xlsx')
new=pd.DataFrame(columns=['Attendance'])
df2.append(new)
df2["Attendance"]="Absent"
allnames= df2['STUDENT NAMES']
for i in range(len(df2['STUDENT NAMES'])):

    for j in range(len(present)):

        if allnames[i] == present[j]:
            df2["Attendance"][i]='Present'
print (df2)


# In[113]:


import datetime
now= datetime.datetime.now()
today=now.day
month=now.month
year=now.year
classname=input('Enter Section Name')
df2.to_csv(r'C:\Users\sai krithik\Desktop\Attendance\Attendance.'+classname+'.'+str(today)+'.'+str(month)+'.'+str(year)+'.csv', sep=',', index = None, header=True)

