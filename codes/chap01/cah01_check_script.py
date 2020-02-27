#-*- coding:utf-8 -*-

import numpy as np
import csv
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
    weight = np.random.normal(RND_MEAN,RND_STD,[10,1])
    bias = np.zeros([10])

load_dataset()
model_init()

global df
shuffle_map = np.arange(len(df))
np.random.shuffle(shuffle_map)
step_count = int(len(df) * 0.8) // 10
test_begin_idx = step_count * 10

test_data = df.loc[shuffle_map[test_begin_idx:]]
test_x = test_data.drop('Rings',axis=1)
test_y = test_data['Rings']


