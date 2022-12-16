import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    # convert to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    L, A, B = cv2.split(lab)

    # increase image contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    A2 = clahe.apply(A)
    test = cv2.merge((L, A2, B))


    img = cv2.blur(A2,(5,5))

    img = cv2.medianBlur(img,5)

    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # get binary image
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # detect circle
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)

    mask = np.zeros_like(img)
    # draw circle on image
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

    result = cv2.bitwise_and(img, mask)
    cv2.imshow('test', result)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()