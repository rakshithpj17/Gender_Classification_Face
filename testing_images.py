from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import argparse
                    
model = load_model('training_face.model')

#Argument Parses to Input Images
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

#Reading an image
image = cv2.imread(args["image"])
cv2.imshow("Original", image)
cv2.waitKey(0)
    
classes = ['male','female']

face, confidence = cv.detect_face(image)
# #idx, f = enumerate(face)
# #print(face)

# #Geting the starting and ending points of face rectangle
# (startX, startY) = f[0], f[1]
# (endX, endY) = f[2], f[3]

for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

#Pre-Processing to predict Gender
face_crop = np.copy(image[startY:endY, startX:endX])
face_crop = cv2.resize(face_crop, (96, 96))
face_crop = img_to_array(face_crop)
face_crop = np.expand_dims(face_crop, axis=0)

prediction = model.predict(face_crop)[0]

idx = np.argmax(prediction)
label = classes[idx]

label = "{}: {:.2f}%".format(label, prediction[idx] * 100)

Y = startY - 10 if startY - 10 > 10 else startY + 10

cv2.putText(image, label, (startX, Y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Gender Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()