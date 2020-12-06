# Signed
Real Time Translation of Sign Language to English.
\\[ x = {-b \pm \sqrt{b^2-4ac} \over 2a} \\]

## Setting Up the environment.
* Set up an anaconda environment as `Python 3.5.6`, and activate this environment.
* Use command prompt or anaconda prompt to install libraries by using `install_packages.txt` using `pip`.
**Note:** Use command prompt or anaconda prompt to run `dependencies.py`. If this runs without any trouble we can now start running our code. This step is necessary because on a lot of systems tensorflow is not installed properly with `pip` installation on the first go. For my particular environment I used ```pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-2.0.0-cp35-cp35m-win_amd64.whl``` as specified on the Tensorflow website.

## Creating your own custom dataset. 
* Run `train.py` to create your own dataset. 
* When the Histogram screen pops up, cover the small green boxes with your palm completely and press `c`. Your palm should be white and the background should be black.
![Histogram](https://github.com/chatsagnik/Signed/blob/master/Histogram.PNG)
* Press `s` to save the histogram and stop recording.
* Input the Hand Sign Label you want the system to store, and input the no of training images. 
* If the label of your dataset is `Ex`, the images are stored under `.\signs\train\Ex\`. Some example thresholded images are shown below.
<br>

![Sign A](https://github.com/chatsagnik/Signed/blob/master/1.jpg) ![Sign B](https://github.com/chatsagnik/Signed/blob/master/12.jpg) 
![Sign C](https://github.com/chatsagnik/Signed/blob/master/3.jpg) ![Sign D](https://github.com/chatsagnik/Signed/blob/master/36.jpg) 
![Sign E](https://github.com/chatsagnik/Signed/blob/master/82.jpg)

## Predict hand-signs in real time.
* Run `predict_hand_sign.py` to start predicting handsigns.
* When the Histogram screen pops up, cover the small green boxes with your palm completely and press `c`. Your palm should be white (in the foreground) and the background should be black.
* Press `s` to save the histogram and stop recording.
* Put your palm inside the green box and start 'signing'. The prediction will be shown on the screen.
![Predicted Y](https://github.com/chatsagnik/Signed/blob/master/Predicted_Y.PNG)
![Predicted C](https://github.com/chatsagnik/Signed/blob/master/Predicted_C.PNG)
![Predicted F](https://github.com/chatsagnik/Signed/blob/master/Predicted_F.PNG)
