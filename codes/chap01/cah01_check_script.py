#-*- coding:utf-8 -*-

"""docstring - my own code for a simple perceptron """

import numpy as np
#import csv
import pandas as pd


RND_MEAN = 0
RND_STD = 0.0030
LEARNING_RATE = 0.001


def load_dataset():
    global df
    df = pd.read_csv('../../data/chap01/abalone.csv')

    # one-hot vectorization
    df['Male'] = np.where(df['Sex'] == 'M', 1, 0)
    df['Female'] = np.where(df['Sex'] == 'F', 1, 0)
    df['Indetermined'] = np.where(df['Sex'] == 'I', 1, 0)
    del df['Sex']

def model_init():
    global weight, bias
    weight = np.random.normal(RND_MEAN,RND_STD,[10,1])
    bias = np.zeros([1])

def dataset_division():
    global weight, bias
    shuffle_map = np.arange(len(df))
    np.random.shuffle(shuffle_map)
    step_count = int(len(df) * 0.8) // 10

    test_begin_idx = step_count * 10
    test_data = df.loc[shuffle_map[test_begin_idx:]]
    test_x = test_data.drop('Rings', axis=1)
    test_y = test_data['Rings']
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    for n in range(step_count):
        train_data = df.loc[shuffle_map[10*0:10*1]]
        train_x = train_data.drop('Rings', axis=1)
        train_x = train_x.to_numpy()
        train_y = train_data['Rings']
        train_y = train_y.to_numpy()
        y = np.array([train_y]).T
        loss, acc = training(train_x, y)
        if n % 100 == 0:
            print("Trainingset ~ Loss: {:5.3f}, Accuracy: {:5.7f}".format(loss, acc))

    loss, acc = testing(test_x, test_y)
    print("Testset ~ Loss: {:5.3f}, Accuracy: {:5.7f}".format(loss, acc))

def testing(test_x, test_y):
    global weight, bias

    y = np.array([test_y]).T

    # Y = X x W + B
    output = np.matmul(test_x, weight) + bias

    # Loss = MSE (Y-Y')
    diff = output - y
    square = np.square(diff)
    loss = np.mean(square)

    # Accuracy
    mdiff = np.mean(np.abs((output - y) / y))

    return (loss, 1 - mdiff)

def training(train_x, y):
    global weight, bias
    #Forward
    # Y = X x W + B
    output = np.matmul(train_x, weight) + bias

    # Loss = MSE (Y-Y')
    diff = output - y
    square = np.square(diff)
    loss = np.mean(square)

    # Accuracy
    mdiff = np.mean(np.abs((output- y)/y))

    #Backward
    # dL/dw = transposeX x G
    # dL/db = np.sum(G, axis = 0)

    # G_output = dL/dY = d/dY ( MSE(Y-Y') )

    shape = diff.shape
    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff

    G_output = g_square_diff * g_loss_square

    g_output_w = train_x.T

    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis =0)

    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b

    return(loss, 1-mdiff)


if __name__ == "__main__":
    load_dataset()
    model_init()
    for epoch_n in range(20):
        dataset_division()