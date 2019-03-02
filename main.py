import numpy as np
import cv2
import io
import os
import numpy as np
import cv2
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Instantiates a client
client = vision.ImageAnnotatorClient()
cap = cv2.VideoCapture('http://18.30.62.49:8080//video')
# cap.set(CV_CAP_PROP_BUFFERSIZE, 3)
while(True):
     # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
    # Our operations on the frame come here
    gray = frame
    # bgr = cv2.imread(image_path)
    lab = cv2.cvtColor(gray, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    gray = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite('ocr.png',gray)
    # Display the resulting frame
    cv2.imshow('After',gray)
    cv2.imshow("before",frame)

    k = cv2.waitKey(1) & 0xFF
    if k  == ord('q'):
        break

    if k == ord('w'):

        # with io.open(frame, 'rb') as image_file:
        #     content = image_file.read()

        # image = types.Image(content=content)
        img = frame
        image = types.Image(content=cv2.imencode('.png', img)[1].tostring())
        resp = client.document_text_detection(image=image)


        # print('\n'.join([d.description for d in resp.text_annotations]))
        img = cv2.imread("ocr.png")
        for word in resp.text_annotations[1:]:
        # print(resp.text_annotations[2])
        # word = resp.text_annotations[2]
        # print(resp)

            text = word.description
            

            print(word.bounding_poly.vertices)
            # Write some Text
            x = word.bounding_poly.vertices[0].x
            y = word.bounding_poly.vertices[0].y
            print("Left corner :", x,"   ",y)
            font  = cv2.FONT_HERSHEY_SIMPLEX
            # bottomLeftCornerOfText = (10,500)
            fontScale              = 0.5
            fontColor              = (255,255,255)
            lineType               = 2
            bottomLeftCornerOfText = (x,y)

            length = word.bounding_poly.vertices[1].x - word.bounding_poly.vertices[0].x
            width = word.bounding_poly.vertices[2].y - word.bounding_poly.vertices[1].y

            # fontScale = (img.shape[0]*img.shape[1]) /( length*width) / 6
            print(fontScale)

            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX = int((length - textsize[0]) / 2) + x
            textY = int((width + textsize[1]) / 2) + y

            color = (0,0,0)
            cv2.rectangle(img, (word.bounding_poly.vertices[0].x,word.bounding_poly.vertices[0].y), (word.bounding_poly.vertices[2].x,word.bounding_poly.vertices[2].y), color,-1)

            cv2.putText(img,text, 
                (textX, textY ), 
                font, 
                fontScale,
                fontColor,
                lineType)

            #Display the image
        print("OCR DONE")
        cv2.imshow("img",img)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()