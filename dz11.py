import cv2
import numpy as np

image = cv2.imread('image_to_search.jpg') 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

patterns = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

overlay_image = cv2.imread('image_for_overlay.png')  

for (x, y, w, h) in patterns:
    overlay_resized = cv2.resize(overlay_image, (w, h))  
    roi = image[y:y+h, x:x+w]

    if overlay_resized.shape[2] == 4:  
        alpha = overlay_resized[:, :, 3] / 255.0
        for c in range(0, 3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay_resized[:, :, c]
    else:
        roi[:, :] = overlay_resized

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output_image.jpg', image)
