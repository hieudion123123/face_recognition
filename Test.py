from annoy import AnnoyIndex
import numpy as np
from imutils import paths
from tqdm import tqdm
import urllib2
import face_recognition
import cv2
from time import sleep

vector_len = 128
u = AnnoyIndex(vector_len, 'angular')
u.load('images.ann')
imagePaths = list(paths.list_images('path_of_you'))

for i, imagePath in tqdm(enumerate(imagePaths)):
    name = imagePath.split(os.path.sep)[-2]
    known_face_names.append(name)
    face_names = []
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

   
    face_names = []
        
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches_id = u.get_nns_by_vector(face_encoding, 1)[0]
        known_face_encoding = u.get_item_vector(matches_id)
        compare_faces = face_recognition.compare_faces([known_face_encoding], face_encoding)
        name = "unknown"

        if compare_faces[0]:
            name = known_face_names[matches_id]
        face_names.append(name)
        print(face_names)
        
    for (top, right, bottom, left), name in zip(face_locations, face_names):
         top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        output_names.append(name)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

 video_capture.release()
 cv2.destroyAllWindows()