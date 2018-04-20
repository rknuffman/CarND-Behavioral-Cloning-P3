import csv
import cv2
import numpy as np
import os

from sklearn.model_selection import train_test_split
import sklearn

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, Cropping2D

# directory with recorded training data
base_path = '/hdd/datasets/udacity_sim_data/recorded_data/'

# separately recorded center laps and recovery laps
samples = []
with open(base_path + 'train/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open(base_path + 'recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# split 
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# correction factor for left / right camera perspectives
c_factor = 0.3

# read / correct / flip
def process_image(sample, angle, correction=0.0, flip=False):
    img = cv2.imread(sample)
    corrected_angle = angle + correction
    if flip:
        return np.fliplr(img), -corrected_angle
    else:
        return img, corrected_angle

# generator for feeding training data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                
                # center
                processed_image, processed_angle = process_image(batch_sample[0], angle)
                images.append(processed_image)
                angles.append(processed_angle)
                processed_image, processed_angle = process_image(batch_sample[0], angle, flip=True)
                images.append(processed_image)
                angles.append(processed_angle)

                # left
                processed_image, processed_angle = process_image(batch_sample[1], angle, correction=c_factor)
                images.append(processed_image)
                angles.append(processed_angle)
                processed_image, processed_angle = process_image(batch_sample[1], angle, correction=c_factor, flip=True)
                images.append(processed_image)
                angles.append(processed_angle)

                # right
                processed_image, processed_angle = process_image(batch_sample[2], angle, correction=-c_factor)
                images.append(processed_image)
                angles.append(processed_angle)
                processed_image, processed_angle = process_image(batch_sample[2], angle, correction=-c_factor, flip=True)
                images.append(processed_image)
                angles.append(processed_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# train and validation generators with batch of 32
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# using NVIDIA architecture from lecture
input_shape=(160,320,3)

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=input_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# early stopping (3 epochs) to prevent overfitting
model.fit_generator(train_generator, 
                    samples_per_epoch= len(train_samples), 
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), 
                    nb_epoch=3)
model.save('model.h5')
