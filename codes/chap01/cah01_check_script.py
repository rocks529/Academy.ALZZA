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
    shuffle_map = np.arange(len(df))
    np.random.shuffle(shuffle_map)
    step_count = int(len(df) * 0.8) // 10

    test_begin_idx = step_count * 10
    test_data = df.loc[shuffle_map[test_begin_idx:]]
    test_x = test_data.drop('Rings', axis=1)
    test_y = test_data['Rings']

    train_data = df.loc[shuffle_map[10*0:10*1]]
    train_x = train_data.drop('Rings', axis=1)
    train_x = train_x.to_numpy()
    train_y = train_data['Rings']
    train_y = train_y.to_numpy()

    output = np.matmul(train_x, weight) + bias
    diff = output - np.array([train_y]).T
    square = np.square(diff)
    loss = np.mean(square)
    print(diff, loss)
    print(np.shape(output))


if __name__ == "__main__":
    load_dataset()
    model_init()
    dataset_division()