#!/usr/bin/env python3  
# Imports
#------------------------------------------------------------------------------------------------------------------------------ 
import numpy as np
import tensorflow as tf
import cv2
import os
# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import pickle, os, random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# compatibility issues with tensorflow
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TensorFlow warning is removed
train_path = ['./signs/train']

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
# load the dataset and populate xtrain and ytrain
def load_data_set(paths):
    X_train = []
    y_train = []
    for path in paths:
        idx = -1
        for root, directories, filenames in os.walk(path):
            if (len(directories)!=0):
                directs = directories
            #print("Hi")
            if (len(filenames)==0):
                continue
            else:
                idx +=1
                char = directs[idx]
                y_train.append([char]*len(filenames))
                
        for root, directories, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".jpg"):
                    fullpath = os.path.join(root, filename)
                    img = cv2.imread(fullpath)
                    X_train.append(img)
    new_Arr = []
    y_train = np.array(y_train)
    for i in range(y_train.shape[0]):
        arr = np.array(y_train[i])
        for j in range(arr.shape[0]):
            new_Arr.append(y_train[i][j])
    
    y_train = new_Arr
    return X_train,y_train
 
 #------------------------------------------------------------------------------------------------------------------------------
def buildNN(classes):

    # Initialising the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    
    # Change units = num_classes
    classifier.add(Dense(units = classes, activation = 'softmax'))
    
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

#------------------------------------------------------------------------------------------------------------------------------
def accuracy(y_train, y_pred, n_classes):
    length = y_train.shape[0]
    correct = 0.0
    incorrect = 0.0
    for i in range(length):
        if (np.argmax(y_train[i])==np.argmax(y_pred[i])):
            correct += 1.0
        else:
            incorrect += 1.0
    acc = correct/(correct+incorrect)
    print("Accuracy is:%f"%(acc))
    return acc

#------------------------------------------------------------------------------------------------------------------------------
def predict_images(hist, clf, prediction_dict):

    # Try capturing with attached webcam. If unavailable, use default webcam.
    cam = cv2.VideoCapture(1)
    if cam.read()[0]==False:
        cam = cv2.VideoCapture(0) # default webcam

    # Set coordinates for hand box in the capture feed.
    x, y, w, h = 310, 50, 300, 300
    flag_start_predicting = False
    frames = 0

    # If camera doesn't start, force it to capture frames.
    if not cam.isOpened():
                cam.open()
    
    # Keep predicting until User exits
    while (True):

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
        result = cv2.findContours(thresh_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = result if len(result) == 2 else result[1:3]

        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 20:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pred_img = thresh_img[y1:y1+h1, x1:x1+w1]
                
                if w1 > h1:
                    pred_img = cv2.copyMakeBorder(pred_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    pred_img = cv2.copyMakeBorder(pred_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))

                pred_img = cv2.resize(pred_img, (50, 50))
                
                # create 3 channels from 1 channel
                pred_img = cv2.merge((pred_img,pred_img,pred_img))
                
                # convert into tensor
                tensor = tf.convert_to_tensor(pred_img)
                
                # create an extra dimension to simulate a batch image
                tensor_reshape = tf.reshape(tensor, [1, tensor.get_shape().as_list()[0],tensor.get_shape().as_list()[1],3])
                
                # convert tensor back to numpy array
                with tf.compat.v1.Session().as_default() as sess:
                    pred_img = tensor_reshape.eval()
                               
                # creating a single batch image
                inp = []
                inp.append(pred_img)

                # get the decision_function for the input image
                y_pred = clf.predict(inp)
                
                # Find the predicted class
                idx = np.argmax(y_pred)
                prediction = prediction_dict.get(idx)
                cv2.putText(img, "Predicted Class is "+str(prediction), (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))
                print(prediction)
        
        # Display the rectangular area which will be used to capture the hand sign
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        
        result = cv2.findContours(thresh_img_copy.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours2, hierarchy = result if len(result) == 2 else result[1:3]

        cv2.imshow("Input",cv2.drawContours(img,contours2,-1,(0,255,0), 3))

        cv2.imshow("Image after Thresholding", thresh_img)

        # Handle the keyboard commands
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if flag_start_predicting == False:
                flag_start_predicting = True
            else:
                flag_start_predicting = False
                frames = 0
        if flag_start_predicting == True:
            frames += 1
        if keypress == ord('s'): #  stop  capturing altogether
            break        
    cam.release()
    cv2.destroyAllWindows()
    return frames

#------------------------------------------------------------------------------------------------------------------------------
# Script
#------------------------------------------------------------------------------------------------------------------------------
X_train, y_train = load_data_set(train_path)

X_train = np.array(X_train)
y_train = np.array(y_train)


class_labels = np.unique(y_train)
classes = len(np.unique(y_train))

test_keys = [*range(classes)]

prediction_dict = {test_keys[i]: class_labels[i] for i in range(len(test_keys))}

print("There are %d classes. The dictionary is as follows: "%classes)
print(prediction_dict)

# print(y_train.shape)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_train = onehot_encoder.fit_transform(integer_encoded)

clf = buildNN(classes)
clf.fit(X_train,y_train,batch_size=800,epochs=6)
#y_pred = clf.predict(X_train)
#accuracy(y_train, y_pred, classes)

hist = get_hist()
predict_images(hist, clf, prediction_dict)
#------------------------------------------------------------------------------------------------------------------------------