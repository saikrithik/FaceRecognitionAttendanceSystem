
################################TEAM Y_AXIS#####################################
import face_recognition
import cv2
import numpy as np
import os
import pickle

video_capture = cv2.VideoCapture(0)#0
video_capture.set(cv2.CAP_PROP_FPS, 24)
scale_factor=0.5

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


with open ('test_encodes.dat', 'rb') as fp:
    known_face_encodings = pickle.load(fp)

image_filenames = filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), os.listdir('images/'))
image_filenames = sorted(image_filenames)
known_face_names = [x[:-6] for x in image_filenames]
final_list = []
for num in known_face_names: 
    if num not in final_list: 
        final_list.append(num)
print(final_list)



face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)#, fx=0.25, fy=0.25
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        
        face_names = []
        for face_encoding in face_encodings:
            name = find_match(known_face_encodings, known_face_names, face_encodings[0])
            face_names.append(name)
            #print(face_names)

    process_this_frame = not process_this_frame


    for (x, y, w, h), name in zip(face_locations, face_names):#Rescaling
        x *= 4
        y *= 4
        w *= 4
        h *= 4

        cv2.rectangle(frame, (h, x), (y, w), (235, 206, 135), 2)
        cv2.rectangle(frame, (h, w - 35), (y, w), (235, 206, 135), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (h + 6, w - 6), font, 1.0, (0, 0, 0), 1)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

















        
