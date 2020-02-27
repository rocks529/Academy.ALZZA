import numpy as np
import csv
import pandas as pd

RND_MEAN = 0
RND_STD = 0.0030
LEARNING_RATE = 0.001

with open('../../data/chap01/abalone.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None)
    rows = []
    for row in csvreader:
        rows.append(row)

rows = pd.read_csv('../../data/chap01/abalone.csv')

print(rows[1,1])