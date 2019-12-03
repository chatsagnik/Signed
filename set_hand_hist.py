import cv2
import numpy as np
import pickle

def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(15):
        for j in range(5):
            if np.any(imgCrop == None):
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
            #print(imgCrop.shape)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
            x+=w+d
        
        if np.any(crop == None):
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop)) 
        imgCrop = None
        x = 420
        y+=h+d
    return crop

def get_hist():
    # Try capturing with attached webcam. If unavailable, use default webcam.
    cam = cv2.VideoCapture(1)
    if cam.read()[0]==False:
        cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 300, 300
    flagPressedC, flagPressedS = False, False
    imgCrop = None

    while True:

        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        #Captures the skin tone values in hand histogram using HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        keypress = cv2.waitKey(1)
        
        if keypress == ord('c'):        
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        elif keypress == ord('s'):
            flagPressedS = True 
            break

        if flagPressedC:    
            proj_img = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            cv2.filter2D(proj_img,-1,disc,proj_img)
            
            blur = cv2.GaussianBlur(proj_img, (11,11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh,thresh,thresh))
            #cv2.imshow("res", res)
            cv2.imshow("Thresh", thresh)

        # if not flagPressedS:
        imgCrop = build_squares(img)
        #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("Set hand histogram", img)
    cam.release()
    cv2.destroyAllWindows()

    # Store the Histogram as a pickle
    with open("hist", "wb") as f:
        pickle.dump(hist, f)


get_hist()
