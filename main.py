import cv2
import numpy as np
from keras.models import load_model

model = load_model('model.h5')

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    height, width = frame.shape[:2]

    # add green rectangle
    cv2.rectangle(frame, (int(width/2-125), int(height/2+125)), (int(width/2+125), int(height/2-125)), (0, 255, 0), 3)

    # cut ROI
    roi = frame[int(height/2-125):int(height/2+125), int(width/2-125):int(width/2+125)]

    # convert to LAB
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    # split LAB to only use A channel
    L, A, B = cv2.split(lab)

    # apply image preprocessing
    img = cv2.medianBlur(A,5)
    img = cv2.GaussianBlur(img,(5,5), cv2.BORDER_DEFAULT)
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # get binary image
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('test', img)

    # predict digit
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img)

    # show result
    cv2.putText(frame, str(np.argmax(prediction)), (int(width/2-200), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3) 
    cv2.imshow('frame', frame)
    
    # quit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()