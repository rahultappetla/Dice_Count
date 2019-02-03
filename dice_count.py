import cv2
import numpy as np
import os

#setting up image input stream directory
names = ['dice{}.png'.format(i) for i in range(1, 7)]
input_dir = r'C:\Users\RAHUL TAPPETLA\Desktop\input'
m=0

for name in names:
    
    src = os.path.join(input_dir,name)
    
    img = cv2.imread(src)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    values=[]
    for i in range(len(contours)):
        
        contour_area = cv2.contourArea(contours[i])
        
        
        if contour_area>10000:
            
            (x, y, w, h) = cv2.boundingRect(contours[i])
            dice_ROI = thresh[y:y+h, x:x+w]
            
            blobdetector = cv2.SimpleBlobDetector_create()
            key_points = blobdetector.detect(dice_ROI)
            value = len(key_points)
            values.append(value)
        
            
            marked = cv2.drawKeypoints(dice_ROI, key_points, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('Marked',marked)
            cv2.waitKey()
            cv2.destroyAllWindows()
            
        
    message="Sum is {}".format(sum(values))        
    final=cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.putText(final,message, (10,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),4, cv2.LINE_AA)
    cv2.imwrite('output_{}.png'.format(name),final)
    cv2.imshow('image after contours',final)
    cv2.waitKey()
    cv2.destroyAllWindows()
