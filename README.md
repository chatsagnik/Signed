# Signed
Real Time Translation of Sign Language to English

1. 
..a. Set up an anaconda environment as `Python 3.5.6`.
..b. Activate the said environment.

2. Use command prompt or anaconda prompt to setup environment by using `install_packages.txt`.
```pip3 install_packages.txt``` 

3. Use command prompt or anaconda prompt to run `dependencies.py`. If this runs without any trouble we can now start running our code. This step is necessary because on a lot of systems tensorflow is not installed properly on `pip` installation on the first go.

4. 
..a. Run `train.py` to create your own dataset. If the label of your dataset is `Ex`, the images are stored under `.\signs\train\Ex\`
..b. When the Histogram screen pops up, cover the small green boxes with your palm completely and press `c`. Your palm should be white and the background should be black.
..c. Press `s` to save the histogram and stop recording.
..d. Input the Hand Sign Label you want the system to store, and input the no of training images.

5.
..a. Run `predict_hand_sign.py` to start predicting handsigns.
..b. When the Histogram screen pops up, cover the small green boxes with your palm completely and press `c`. Your palm should be white (in the foreground) and the background should be black.
..c. Press `s` to save the histogram and stop recording.
..d. Put your palm inside the green box and start 'signing'. The prediction will be shown on the screen.
