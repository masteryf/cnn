import cv2 as cv
import numpy as np
import os
import time
drawing = False
resolution = 128
mode = False
big = 25
start = (-1, -1)
def mouse_event(event, x, y, flags, param):
    global start, drawing, mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv.circle(img, (x, y), big, (0, 0, 0), -1)
            else:
                cv.circle(img, (x, y),big, (255, 255, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv.circle(img, (x, y), big, (0, 0, 0), -1)
        else:
            cv.circle(img, (x, y), big, (255, 255, 255), -1)
#img = np.zeros((512, 512, 1), np.uint8)
img = cv.imread("GUI.png")
cv.namedWindow('image')
cv.setMouseCallback('image', mouse_event)

while (True):
    cv.imshow('image', img)
    if cv.waitKey(1) == ord('p'):
        img = cv.imread("GUI.png")
    if cv.waitKey(1) == ord('m'):
        mode = not mode
    if cv.waitKey(1) == 27:
        break
    if cv.waitKey(1) == ord('s'):
        #img = cv.imread("GUI.png")

        cv.putText(img,str(resolution)+ "*" +str(resolution), (960,310),cv.FONT_HERSHEY_SIMPLEX, 1, (112,164,203), 2, 4, 0)
        cv.putText(img, "start...", (960,460),cv.FONT_HERSHEY_SIMPLEX, 1, (112,164,203), 2, 4, 0)
        cv.imshow('image',img)
        cv.waitKey(10)
        print("start...")
        pho = img[115:731,175:792]
        frame = cv.resize(pho, (resolution, resolution), interpolation=cv.INTER_AREA)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #cv.imshow("aaa",pho)
        #cv.waitKey(1000)
        with open('photo.txt', 'w') as file:
            np.savetxt(file, frame, fmt='%d', delimiter='\t')
        os.system("C:/Users/64783/CLionProjects/untitled/cmake-build-debug/untitled.exe")
        time.sleep(2)
        cv.waitKey(10)
        cv.putText(img, "complete!", (960,610),cv.FONT_HERSHEY_SIMPLEX, 1, (112,164,203), 2, 4, 0)
        with open("C:\\Users\\64783\\CLionProjects\\untitled\\cmake-build-debug\\ans.txt","r")as ans:
            cv.putText(img, ans.read(), (950,740),cv.FONT_HERSHEY_SIMPLEX, 2, (112,164,203), 2, 4, 0)
            print(ans.read())