import cv2 
import numpy as np
import pickle
import dlib
from keras.preprocessing import image

pickle_in=open("model_trained_64.p","rb")
model=pickle.load(pickle_in)

cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)

    for face in faces:
        landmarks=predictor(gray,face)
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        roi_gray=gray[y1:y2,x1:x2]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        
        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if key==27:
          break
cap.release()
cv2.destroyAllWindows()