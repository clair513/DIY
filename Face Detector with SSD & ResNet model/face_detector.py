# Importing required libraries:
import argparse
import numpy as np
import cv2


# Command line argument parser:
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-i', '--image', required=True)
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-p', '--prototxt', required=True)
parser.add_argument('-c', '--confidence_interval', type=float, default=0.5)
# Creating Python dictionary from parsed arguments:
args = vars(parser.parse_args())

# Loading our pre-trained model:
network = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
# Loading input Image:
img = cv2.imread(args['image'])
# Get image dimensions:
(h,w) = img.shape[:2]
# Pre-processing image [Mean subtraction(To control illumination changes by computing avg pixel intensity for RGB & subtracting that fro Input image) and manual Scaling (for Normalization)] to 4-Dimensional image:
img_blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))

# Feed image blob into model for detection:
network.setInput(img_blob)
face= network.forward()

# Loop over detections to filter weak detections based on minimum C.I, then add bounding box:
for i in range(0, face.shape[2]):
    ci= face[0,0,i,2]
    if ci>args['confidence_interval']:
        box = face[0,0,i,3:7] * np.array([w,h,w,h])
        (tl,tr,bl,br) = box.astype('int')
        text = '{:.2f}%'.format(ci*100)

        if tr - 10 > 10:
            y = tr - 10
        else:
            y = tr + 10

        cv2.rectangle(img, (tl,tr), (bl,br), (0,128,0), 2)
        cv2.putText(img, text, (tl,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (252,21,0), 2)

# Image Output:
print(cv2.__version__)
cv2.imshow('Output Image',img)
cv2.waitKey(0)

# TO EXECUTE:
# python face_detector.py --image family_1_input.jpg --model res10_300x300_ssd_iter_140000.caffemodel --prototxt deploy.prototxt.txt
