# VideoProcessor

import boto3
import requests
import cv2
from numpy import genfromtxt
import numpy as np

#  ----------Paths and artifacts used by script -------------------------------------------------
# The script uses a private bucket: s3bucketnamepost to store the csv file coming from the parser script.
# The script uses a public bucket to store the input video and reference image: s3bucketpublicpost
# Replace with your own paths (URL or local files) to your input video, reference image and the generated csv file.


# First we take the faceoutput csv from S3 using Boto3 and store it locally
# Later, we willload it into NumPy.


s3c=boto3.client('s3')
s3c.download_file(‘s3bucketname','faceoutput.csv','faceoutput.csv')

# video input in S3,replace with your own path
Videof = " http://s3bucketpublicname/vaivideo.mp4"

# reference image in our face collection used for detection in S3, replace with your own path

imgdown = cv2.VideoCapture("http://s3bucketpublicpost/vai.jpg")
if( imgdown.isOpened() ) :
    ret,Imagef = imgdown.read()


# Other variables:
permanence = 7  # permanence factor of the bounding box. see comments on the blog post below for deeper explanation
wkey = 1  # OpenCV wait key, needs to be modified according to the speed of the input video, see comments on this post below 

# Reference image rescaling factor for visualization. The reference image will be displayed in the upper left corner of the video
# We need to rescale if it is too big, in order to have a proper visualization. Adjust according to your image.

Refimagescale = 0.4  # Reference image rescaling factor for visualization. 

# -------------------------------------------------------------------------------------------------


# Load csv into Numpy
my_data = genfromtxt(‘faceoutput.csv’, delimiter=',', dtype=float)

# Load video and extract data

cap = cv2.VideoCapture(Videof)
width = cap.get(3)
height = cap.get(4)
vfps = cap.get(cv2.CAP_PROP_FPS)

# Recalculate NumPy data to frames per second
my_data[:, 0] = my_data[:, 0] * vfps / 1000
my_data[:, 0] = my_data[:, 0].astype(int)  # convert to  int type

# index we are going to need, indice is the frame counter, our frames’ index
indice = 0
perm = 0  # permanence counter

# Read the reference image used for detection to be displayed in the contextualization
# resizing it with the resizing factor – it will be place in the upper left of the #video when person is detected.


res_img = cv2.resize(Imagef, (0, 0), fx=Refimagescale, fy=Refimagescale)

# loop frame by frame
while True:

    ret, img = cap.read()
    # if video stops, we are out
    if ret == False:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    gray = cv2.equalizeHist(gray)  # equalize

    # Check if current frame is a frame where Rekognition detected a person
    # Return NumPy array of dimensions (1,6) with the parameters

    matchidx = my_data[np.where((my_data[:, 0]) == indice)]

    # Parameters are: Frame per second, Left, Top, Width, Height, Confidence
    if matchidx.size != 0:
        # If indice = frame start the visualization / contextualization
        my_data[:, 0] = my_data[:, 0]
        cx = int((matchidx[0, 1]) * width)
        cy = int((matchidx[0, 2]) * height)
        cx2 = int((matchidx[0, 3]) * width + cx)
        cy2 = int((matchidx[0, 4]) * height + cy)
        conf = int(matchidx[0, 5])

        # Display reference image in upper left part of the frame
        x_offset = y_offset = 1
        img[y_offset:y_offset + res_img.shape[0], x_offset:x_offset + res_img.shape[1]] = res_img

        # Display bounding box, lines and text
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, 'Match', (cx, cy2), font, 1, (255, 255, 255), 1, cv2.LINE_AA)  # texto con match
        cv2.putText(img, str(conf) + "%", (cx, cy - 10), font, 1, (255, 255, 255), 1, cv2.LINE_AA)  # confianza
        cv2.putText(img, str(indice) + "," + str(cx) + "," + str(cy), (cx, cy2 + 15), font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)  # footer con frame number tambien x,y coordenates
        cv2.line(img, (cx - 40, cy + 15), (cx, cy + 15), (255, 255, 255), 1, 8)  # linea horizontal

        cv2.rectangle(img, (cx, cy), (cx2, cy2), (255, 255, 255), 1)
        # Now we activate permanence for smoother bounding box transitions
        perm = permanence

    # Are we in permanence state?, If so, display everything again
    if perm > 0:
        x_offset = y_offset = 1
        img[y_offset:y_offset + res_img.shape[0], x_offset:x_offset + res_img.shape[1]] = res_img
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, 'Match', (cx, cy2), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(conf) + "%", (cx, cy - 10), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(img, str(indice) + "," + str(cx) + "," + str(cy), (cx, cy2 + 20), font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.line(img, (cx - 40, cy + 15), (cx, cy + 15), (255, 255, 255), 1, 8)
        cv2.rectangle(img, (cx, cy), (cx2, cy2), (255, 255, 255), 1)
        perm = perm - 1

    # Show the frame and prepare for next loop iteration

    cv2.imshow('Image', img)
    k = cv2.waitKey(wkey) & 0xFF  # OpenCV wait to process next frame, value needs to be >0
    indice = indice + 1  # increment the frame index, loop over next frame

cap.release()
cv2.destroyAllWindows()
