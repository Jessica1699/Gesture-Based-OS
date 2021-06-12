import cv2
import numpy as np
import HandTrackingModule as htm
import time
import os

brushThickness = 15
eraserThickness = 100

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for impath in myList:
    image = cv2.imread(f'{folderPath}/{impath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]

drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detect_conf=0.85)

xp,yp = 0,0

imgCanvas = np.zeros((720,1280,3),np.uint8)

while True:
    # 1. import image
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flipping the image

    # 2. find hand landmarks using hand tracking module
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        # print(lmlist)

        # tip of index and middle finger
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

    # 3. check which fingers are up:if 1 finger(index finger) then draw...if 2 fingers are up move around canvas
    # without drawing

        fingers = detector.fingersUp()
        # print(fingers)

    # 4. if selection mode - 2 fingers are up then select not draw
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawColor, cv2.FILLED)

    # 5. if drawing mode - index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1),15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp,yp = x1,y1      # draw a point 1st time instead of line

            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            cv2.line(img, (xp,yp),(x1,y1),drawColor,brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp,yp = x1,y1

    # Clear Canvas when all fingers are up
        if all(x >= 1 for x in fingers):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    # Setting the header image
    img[0:125, 0:1280] = header

    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Canvas",imgCanvas)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
