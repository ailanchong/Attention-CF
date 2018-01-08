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

def generate_test_data(data):
    sentence = []
    sentenlen = []
    usridx = []
    proidx = []
    rate = []
    for sample in data:
        sentence.append(sample['sentence'])
        sentenlen.append(sample['sentenlen'])
        usridx.append(int(sample['usridx']))
        proidx.append(int(sample['proidx']))
        rate.append(sample['rate'])
    sentence = zero_padding(sentence, 300)
    batch_data = {'sentence':sentence, 'sentenlen':sentenlen, 'usridx':usridx, 'proidx':proidx, 'rate':rate}
    return batch_data 


def load_data(filepath):
    result = []
    with open("./pro_data/{}".format(filepath)) as file_in:
        for line in file_in:
            data = json.loads(line)
            result.append(data)
    return result

def train(batch_size, train_data, test_data,iter_nums=5):
    test_data = generate_test_data(test_data)
    abmfer = Abmf(maxseqlen=300, word_dim = 200, rnnstate_size = 50, attention_size = 20, usr_num=6040, rank_dim=50, pro_num=3544, word_num=8000)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(abmfer.loss)
    saver = tf.train.Saver()
    step = tf.Variable(0,trainable=False)
    step_list = []
    loss_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_gendata = generate_batch_data(batch_size,train_data)
        for i in range(iter_nums):
            for j in range(len(train_data) // batch_size):
                batch_data = next(train_gendata)
                _, step_num, loss = sess.run([optimizer, step, abmfer.loss], feed_dict={
                    abmfer.input_seq : batch_data['sentence'],
                    abmfer.input_seqlen : batch_data['sentenlen'],
                    abmfer.usr : batch_data['usridx'],
                    abmfer.pro : batch_data['proidx'],
                    abmfer.rate : batch_data['rate']
                })
                print("batch_step: %s, train_rmse: %s" %(step_num, loss))
                step_list.append(step_num)
                loss_list.append(loss)

            #plot 画图！！！
            plt.plot(step_list, loss_list)
            plt.savefig("abmf")
            plt.show()
            #验证集损失
            test_loss = sess.run([abmfer.loss], feed_dict={
                abmfer.input_seq : test_data['sentence'],
                abmfer.input_seqlen : test_data['sentenlen'],
                abmfer.usr : test_data['usridx'],
                abmfer.pro : test_data['proidx'],
                abmfer.rate : test_data['rate']
            })
            print("iter_step: %s, test_rmse: %s" %(i, test_loss))
            #模型保存
            saver.save(sess, "./model/abmf.ckpt")
    
if __name__ == "__main__":
    train_data = load_data("train")
    test_data = load_data("val")
    train(128, train_data, test_data)
        

         
        