# **Behavioral Cloning**

## Writeup

### Udacity Course, October 2017 cohort

**Self-Driving Car Engineer Nanodegree Program**

**Project 'Behavioral Cloning', December 2017**
---

**Train a deep neural network to drive a car Project**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[Ten_epochs]: Result_ten_epochs.png "10 epochs, not using a Generator"
[Three_epochs]: Result_three_epochs.png "3 epochs, using a Generator"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to load images via a generator and to create and train the model
* drive.py for driving the car in autonomous mode (this file has not been modified)
* model.h5 containing a trained convolution neural network (all augmentations etc. takes place here)
* run1.mp4 containing a video of a succesful run using the trained CNN
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used the Nvidia CNN model, using RELU layers to introduce nonlinearity (model.py lines 113-137)

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64.

The data is normalized in the model using a Keras lambda layer (code line 116).

#### 2. Attempts to reduce overfitting in the model

The model contains two Dropout layers in order to reduce overfitting (model.py lines 133 and 135).

The model was trained and validated on three different data sets to ensure that the model was not overfitting (code line 28-55).
* data set 1 : two laps of forward, slow center driving.
* data set 2 : reverse drive from start to bridge and back again
* data set 3 : same as no 2, but with a lot of recovery driving to learn how to handle recovery.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. When the model finally was finished, the the car went on test driving without errors for several hours on the easy track, that had been used to train the model.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 140).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The data set are described in section 2 (above).

I spend a fair amount of time in the first place just to learn to drive the car with the keyboard. I never succeed using the mouse to handle the car. It also took many recording sessions and failures before I had the three data sets, that were used to train this model. My lesson learned from the Traffic sign Classfication task was to create credible data, that were as unbiased as possible.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the LeNet CNN model, but I switched to the more complex Nvidia CNN model, mostly to get some experience with this model. And it turned out that it could manage the job of driving the car running only three epochs of training data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. This was the general picture when I ran the model for ten epochs (with images loaded into memory):

Training ten epochs
![alt text][Ten_epochs]

The Validation Loss is constantly high, which may be caused by overfitting.
To combat the overfitting, I therefore modified the model by adding two Dropout layers between two of the connected layers.

The final model was created using only three epochs and loading the images via a Generator (which took an extraordinary amount of extra processing time) _and_ removing half of the images where the steering was < 0.75. Note to my self: as of writing this, I now see that I only looked at positive angles, should have processed negative low angles as well :-( Bummer, but it all went well even so :-)

Training three epochs
![alt text][Three_epochs]

The output from the model training is shown here - this took about 18 hours (**!**):

(IntroToTensorFlow) claus-h-rasmussens-mac-pro:CarND-Behavioral-Cloning-P3 claushrasmussen$ python model.py
Using TensorFlow backend.
Number of observations: 14965
Sun Dec 10 21:14:14 2017
Number of observations: 17055
Sun Dec 10 21:14:14 2017
Number of observations: 18486
Sun Dec 10 21:14:14 2017
14788/14788 [==============================] - 21738s - loss: 0.0325 - val_loss: 0.0554
Epoch 2/3
14788/14788 [==============================] - 21567s - loss: 0.0111 - val_loss: 0.0522
Epoch 3/3
14788/14788 [==============================] - 21313s - loss: 0.0084 - val_loss: 0.0576
Mon Dec 11 15:11:14 2017
model saved

It can bee seen that both the Loss and the Validation Loss are low, so the overfitting has been avoided in this model.

**At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.**


#### 2. Final Model Architecture

The final model architecture (model.py lines 113-137) consisted of a implementation of the Nvidia convolution neural network with the following layers and layer sizes
160x320x3-C24-C36-C48-C64-C64-100N-DO-50N-DO-10N-1N, with Relu activations.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 RGB image   							|
| Convolution 5x5     	| 2x2 stride, VALID padding, 24 layers 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, VALID padding, 36 layers 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, VALID padding, 48 layers 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, VALID padding, 64 layers 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, VALID padding, 64 layers 	|
| RELU					|												|
| Flatten       |
| Fully connected		| Output 100        									|
| DropOut	      	|   				|
| Fully connected		| Output 50        									|
| DropOut	      	|   				|
| Fully connected		| Output 10        									|
| Fully connected		| Output 1        									|


#### 3. Creation of the Training Set & Training Process

The creation of the data sets used here is described in the text above.

To augment the data sat, I also flipped images and angles i order to multiply the dataset with reversed driving data - this must be the first and most obvious augmentation method (model.py line 91 and 97). This resulted in a total of 36972 images (from the 18486 samples).

The little rock on the road caused by using cv2.imread, that the color space changes from RGB to GBR gave me a lot of headache, resulting in the car driving into sands right after the brigde. Using cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB) (model.py line 81) did the trick. Btw, I have not changes anything in drive.py, data augmentation etc. only takes place in model.py.

The model seemed biased by too much 'driving straight' data, so I removed half of the images with steering angles < 0.75 (model.py line 85).
From an inital amount of data 36972 I ended up training and validating on a total of 18485 samples, effectively reducing the data set by one third.

I finally randomly shuffled the data set and put 20% of the data into a validation set.


#### 4. Final thoughts

I have spend too much time learning how to do the task - I started out using Jupyter Notebook, which was in the beginning, but ended up making if more difficult than necessary to create the code in model.py, I think. I also should have read the discussions at the Udacity forum earlier in the process - I ended up reading almost every discussion in the search of specific answers instead of using the discussions as guide lines into solving this project.
Anyway, I learned a lot along the way, and once again that it's almost all about the data, not so much the model. Well, it kind of has something to do with the model, but model building is actually the easy part here. Besides sleepless nights trying to figure out why I couldn't get past the sand barrier and the curves, I had a lot of fun. And learned a ton :-)


Kind regards, Claus H. Rasmussen.
