import tensorflow as tf
import numpy as np
from abmf import Abmf
import matplotlib.pyplot as plt
import json

def zero_padding(sentence_list, maxlen):
    result = np.zeros((len(sentence_list), maxlen), dtype=np.int)
    for i, row in enumerate(sentence_list):
        for j, val in enumerate(row):
            result[i][j] = val
    return result

def load_data(filepath):
    result = []
    with open("./pro_data/{}".format(filepath)) as file_in:
        for line in file_in:
            data = json.loads(line)
            result.append(data)
    return result

train_data = load_data("train")
def generate_batch_data(batch_size, data):
    while(1):
        sentence = []
        sentenlen = []
        usridx = []
        proidx = []
        rate = []
        curr_num = 0
        np.random.shuffle(data)
        for sample in data:
            curr_num += 1
            sentence.append(sample['sentence'])
            sentenlen.append(sample['sentenlen'])
            usridx.append(int(sample['usridx']))
            proidx.append(int(sample['proidx']))
            rate.append(sample['rate'])
            if curr_num == batch_size:
                curr_num = 0
                sentence = zero_padding(sentence, 300)
                batch_data = {'sentence':sentence, 'sentenlen':sentenlen, 'usridx':usridx, 'proidx':proidx, 'rate':rate}
                yield batch_data
                sentence = []
                sentenlen = []
                usridx = []
                proidx = []
                rate = []

for i in range(5):
    batch_data = generate_batch_data(128, train_data)
    print(next(batch_data))
    print(next(batch_data))
    print(len(next(batch_data)['sentence']))