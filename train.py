#!/usr/bin/env python3  

# Imports
#------------------------------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import pickle, os, random

#------------------------------------------------------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------------------------------------------------------
def set_hist_spot(img):
    x, y, w, h = 420, 140, 10, 10
    step = 10
    def_crop = None
    spot = None
    for i in range(15):
        for j in range(8):
            if np.any(def_crop == None):
                def_crop = img[y:y+h, x:x+w]
            else:
                def_crop = np.hstack((def_crop, img[y:y+h, x:x+w]))
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
            x += w+step
        
        if np.any(spot == None):
            # set spot as the default spot
            spot = def_crop
        else:
            spot = np.vstack((spot, def_crop)) 
        def_crop = None
        x = 420
        y += h+step
    return spot

#------------------------------------------------------------------------------------------------------------------------------
def get_hist():
    # Try capturing with attached webcam. If unavailable, use default webcam.
    cam = cv2.VideoCapture(1)
    if cam.read()[0]==False:
        cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 300, 300
    flag_start_capturing = False

    imgCrop = None

    while True:

        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        #Captures the skin tone values in hand histogram using HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        keypress = cv2.waitKey(1)
        
        if keypress == ord('c'):        
            HSV_img = cv2.cvtColor(hist_spot, cv2.COLOR_BGR2HSV)
            flag_start_capturing = True
            hist = cv2.calcHist([HSV_img], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        elif keypress == ord('s'):
            # flagPressedS = True 
            break

        if flag_start_capturing:    
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
        hist_spot = set_hist_spot(img)
        #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("Hand histogram", img)
    cam.release()
    cv2.destroyAllWindows()

    # Store the Histogram as a pickle
    with open("hist", "wb") as f:
        pickle.dump(hist, f)

    return hist

#------------------------------------------------------------------------------------------------------------------------------
def make_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)
    return

#------------------------------------------------------------------------------------------------------------------------------
def store_images(hist, label, no_of_pics):

    # Try capturing with attached webcam. If unavailable, use default webcam.
    cam = cv2.VideoCapture(1)
    if cam.read()[0]==False:
        cam = cv2.VideoCapture(0) # default webcam

    # Set coordinates for hand box in the capture feed.
    x, y, w, h = 310, 50, 300, 300

    # Create the folder to store the images.
    folder = './signs/train/'+str(label)
    make_folder(folder)
    
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    # If camera doesn't start, force it to capture frames.
    if not cam.isOpened():
                cam.open()

    # Start capturing frame by frame until we have captured required no of images
    while(pic_no < no_of_pics):
        
        img = cam.read()[1]
        # The hand coordinates are defined according to the flipped image
        img = cv2.flip(img, 1)
        
        # The Hue Saturation Value model used to capture the skin color
        HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Back Projection is used to  measure how well the pixels of a given image fit the distribution of pixels in a particular histogram model.
        # If you have a histogram of flesh color (say, a Hue-Saturation histogram ), then you can use it to find flesh color areas in an image.
        proj_img = cv2.calcBackProject([HSV_img], [0, 1], hist, [0, 180, 0, 256], 1)
        
        # Apply Morphological Transformation to detect the edges better.
        morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

        # Filter2D operation convolves an image with the kernel.
        cv2.filter2D(proj_img,-1,morph,proj_img)

        # Smoothing the  input
        smooth_img = cv2.GaussianBlur(proj_img, (11,11), 0)
        smooth_img = cv2.medianBlur(smooth_img, 15)

        # Otsu's thresholding after Gaussian filtering
        # If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value.
        thresh_img = cv2.threshold(smooth_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        
        # Creating a 3-channeled image
        thresh_img = cv2.merge((thresh_img,thresh_img,thresh_img))

        # Creating grayscale images
        thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2GRAY)

        # Cropping required part of the image
        thresh_img_copy = thresh_img
        thresh_img = thresh_img[y:y+h, x:x+w]

        # Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity.
        # Contours are most accurate for binary images. Hence we find contours on thresholded image.
        result = cv2.findContours(thresh_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = result if len(result) == 2 else result[1:3]
        
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh_img[y1:y1+h1, x1:x1+w1]
                
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))

                save_img = cv2.resize(save_img, (50, 50))

                # Flip around 50% of the images
                rand = random.randint(0, 10)
                if rand % 2 == 0:
                    save_img = cv2.flip(save_img, 1)
                
                cv2.putText(img, "Capturing Images...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite("signs/train/"+str(label)+"/"+str(pic_no)+".jpg", save_img)

        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        
        # Display the rectangular area which will be used to capture the hand sign
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

        result = cv2.findContours(thresh_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours2, hierarchy = result if len(result) == 2 else result[1:3]

        cv2.imshow("Input",cv2.drawContours(img,contours2,-1,(0,255,0), 3))

        cv2.imshow("Image after Thresholding", thresh_img)

        # Handle the keyboard commands
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if flag_start_capturing == False:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames = 0
        if flag_start_capturing == True:
            frames += 1
        if keypress == ord('s'): #  stop  capturing altogether
            break
        
    cam.release()
    cv2.destroyAllWindows()
    return frames

#------------------------------------------------------------------------------------------------------------------------------
# Script
#------------------------------------------------------------------------------------------------------------------------------
hist = get_hist()
label = input("Enter Hand Sign Label: ")
no_of_pics = int(input("Enter no of training images: "))
store_images(hist, label,no_of_pics)
#------------------------------------------------------------------------------------------------------------------------------