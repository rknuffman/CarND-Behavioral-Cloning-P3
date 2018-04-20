# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[orig]: ./examples/original.jpg "Original Image"
[crop]: ./examples/cropped.jpg "Cropped Image"
[flip]: ./examples/flipped.jpg "Flipped Image"
[viz]: ./examples/model.jpeg "Model Visualization"
[ctr]: ./examples/center.jpg "Center Image"
[rec1]: ./examples/recovery_1.jpg "Recovery Image 1"
[rec2]: ./examples/recovery_2.jpg "Recovery Image 2"
[rec3]: ./examples/recovery_3.jpg "Recovery Image 3"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
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

My model consists of a convolutional neural network with 5x5 filter sizes and depths between 24 and 48, 3x3 filter sizes and depths of 64, and multiple densely connected layers (model.py lines 90-102) 

The model includes RELU layers to introduce nonlinearity (code line 93-97), and the data is normalized in the model using a Keras lambda layer (code line 92). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 107-111). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and recovering from the left and right sides of the road.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start simple and increase model complexity where appropriate.

My first step was to use a simple convolutional neural network model similar to the LeNet architecture. I didn't expect incredible performance from this architecture, but knew I would get a decent baseline to compare against.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I observed training and validation loss, and halted training after validation loss stopped improving.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and drove onto dirt sections, only to re-emerge further down the track. To improve the driving behavior in these cases, I found recording recovery laps to be beneficial.

Ultimately, switching to a more powerful network presented in the lecture, training on recovery data, and continuing to monitor and halt training after validation loss stopped improving was my final architecture. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 90-102) consisted of a convolutional neural network with the following layers and layer sizes:

![alt text][viz]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. I found controlling the steering angle with the mouse to provide a much more consistent target for my model to learn from.  Here is an example image of center lane driving:

![alt text][ctr]

Additionally, I included both left and right camera perspectives while center driving in my training set.  I manually adjusted the steering angle for these perspectives based on a correction factor.  This allowed the model to learn how to adjust if the center camera begins to see perspectives similar to a left or right camera.  

I then recorded the vehicle recovering from the left and right sides of the road back to center so that the vehicle would learn to recover after a more extreme approach of the edge of the lane. These images show what a recovery looks like starting from various positions :

![alt text][rec1]
![alt text][rec2]
![alt text][rec3]

To augment the data sat, I also flipped images and angles thinking that this would provide a balanced amount of left and right turns for the model to learn from. For example, here is an image that has then been flipped:

![alt text][flip]
![alt text][orig]

After the collection process, I had 12,045 number of data points. I then preprocessed this data by cropping unnecessary portions of the image, and normalizing pixel intensities.  

![alt text][crop]

I finally randomly shuffled the data set and put 20% of the data into a validation set. The remaining 80% was used for training the model (code line 27).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by a lack of improvement in validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.