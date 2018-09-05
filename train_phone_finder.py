from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys
import glob
import cv2
import random
import numpy as np
import os
import matplotlib.pyplot as plt
# Keras modules
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th')
# Get the prediction function to find accuracy
from find_phone import predict_phone_position

half_crop_size = 22
crop_size = 44

def crop_images(img, pos, num_sample_phones=50, num_sampe_background=50):
    height, width = img.shape
    phone_images = []
    background_images = []
    pos_pixel = np.array((int(pos[0] * width), int(pos[1] * height)))
    # left boundary of box
    box_lb = pos_pixel[0] - half_crop_size
    # right boundary of box
    box_rb = pos_pixel[0] + half_crop_size
    # upper boundary of box
    box_ub = pos_pixel[1] - half_crop_size
    # bottom boundary of box
    box_bb = pos_pixel[1] + half_crop_size
    # crop the phone from the image
    phone_crop = img[box_ub:box_bb, box_lb:box_rb]
    # randomly rotate 90 degree of cropped phone
    for i in range(num_sample_phones):
        random.seed(i)
        pi = random.random()
        if pi > 0.75:
            t = random.choice([1, 2, 3, 4])
            phone_images.append(np.rot90(phone_crop, t))
        else:
            phone_images.append(phone_crop)

    # randomly crop background images
    for i in range(num_sampe_background):
        # coordinate of the left up corner of cropped background
        random.seed(i)
        start_x = box_lb - 60 if (box_lb > 60) else 0
        start_y = box_ub - 60 if (box_ub > 60) else 0
        b_x = random.randint(start_x, width - crop_size)
        b_y = random.randint(start_y, height - crop_size)
        # in case there would be overlap between the background crop and phone crop
        while b_x in range(start_x, box_rb) and b_y in range(start_y, box_bb):
            b_x = random.randint(0, width - crop_size)
            b_y = random.randint(0, height - crop_size)
        back_crop = img[b_y: b_y + crop_size, b_x: b_x + crop_size]
        background_images.append(back_crop)

    return phone_images, background_images


def prepare_data(image_dir, label_dir):
    # read in label and stored into list
    f = open(label_dir)
    iter_f = iter(f)
    list_f = []
    for line in iter_f:
        line = line.strip('\n')
        list_f.append(line.split(" "))
    # convert list to dict
    dict_f = {x[0]: np.array([round(float(x[1]), 4), round(float(x[2]), 4)]) for x in list_f}

    data_phone = []
    data_background = []
    for filename in os.listdir(image_dir):
        if filename != "labels.txt":
            image = cv2.imread(image_dir + '/' + filename)
            #print(filename)
            image_G = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            phone_images, background_images = crop_images(image_G, dict_f[filename])
            data_phone.extend(phone_images)
            data_background.extend(background_images)
    data_phone = np.array(data_phone)
    data_background = np.array(data_background)
    data = np.vstack((data_phone, data_background))
    label = np.hstack((np.ones(len(data_phone)), np.zeros(len(data_background))))

    data, label = shuffle(data, label, random_state=42)

    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=42)

    # Reshape data to match input format of CNN
    train_data = train_data.reshape(train_data.shape[0], 1, crop_size, crop_size).astype('float32')
    test_data = test_data.reshape(test_data.shape[0], 1, crop_size, crop_size).astype('float32')
    # normalize input data
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    return train_data, test_data, train_label, test_label


def create_model(X_train, X_test, y_train, y_test):
    # to get reproducible results
    np.random.seed(0)
    tf.set_random_seed(0)

    # create model
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(1, 44, 44), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    sgd = optimizers.SGD(lr=0.1, decay=1e-2)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # Earlystopping
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
    callbacks_list = [earlystop]
    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=callbacks_list, epochs=50,
                        batch_size=128)
    # save model in HDF5 format
    model.save("model.h5")
    return model


def accuracy(image_dir, label_dir):
    f = open(label_dir)
    iter_f = iter(f)
    list_f = []
    for line in iter_f:
        line = line.strip('\n')
        list_f.append(line.split(" "))
    # convert list to dict
    dict_f = {x[0]: np.array([round(float(x[1]), 4), round(float(x[2]), 4)]) for x in list_f}

    model = load_model('model.h5')
    accuracy = 0
    total = 0
    for filename in os.listdir(image_dir):
        total = total + 1
        image = image_dir + '/' + filename
        pos = predict_phone_position(image, model)
        res = np.sqrt(np.sum(np.power(pos - dict_f[filename], 2)))
        if res <= 0.05:
            accuracy = accuracy + 1
        else:
            print(filename, " ", pos, " ", dict_f[filename])
    accuracy = accuracy / total
    print(accuracy)
    return accuracy

def main():
    path = sys.argv[1]
    print(path)
    train_data, test_data, train_label, test_label= prepare_data(path, os.path.join(path, 'labels.txt'))
    #accuracy(path, os.path.join(path, 'labels.txt'))
    model = create_model(train_data, test_data, train_label, test_label)
    print("model trained")

if __name__ == "__main__":
    main()