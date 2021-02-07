import face_recognition
import cv2
import numpy as np
import os
import pickle

video_capture = cv2.VideoCapture(0)#0
video_capture.set(cv2.CAP_PROP_FPS, 24)


with open ('test_encodes.dat', 'rb') as fp:
    known_face_encodings = pickle.load(fp)

image_filenames = filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), os.listdir('images/'))
image_filenames = sorted(image_filenames)
known_face_names = [x[:-6] for x in image_filenames]
#print(known_face_names)
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
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.5)
            #name = "Unknown"
            #print(matches)
            name= 'Unknown'
            #face_distances = face_recognition.compare_faces(known_face_encodings, face_encoding)
            count=0
            #print(face_distances)
            for match in matches:
               if match:
                   name= known_face_names[count]
               count+=1
            
                
            #print(face_distances)
            
            #best_match_index = np.argmin(face_distances)
            #if matches[best_match_index]:
             #   name = known_face_names[best_match_index]

            face_names.append(name)
            print(face_names)

    process_this_frame = not process_this_frame


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (235, 206, 135), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (235, 206, 135), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

















        
