import tensorflow as tf
import numpy as np
from abmf import Abmf
import matplotlib.pyplot as plt


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
            sentence.append(data['sentence'])
            
            sentenlen.append(data['sentenlen'])
            usridx.append(int(data['usridx']))
            proidx.append(int(data['proidx']))
            rate.append(data['rate'])
            if curr_num == batch_size:
                curr_num = 0
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
        sentence.append(data['sentence'])
        sentenlen.append(data['sentenlen'])
        usridx.append(data['usridx'])
        proidx.append(data['proidx'])
        rate.append(data['rate'])
    return batch_data = {'sentence':sentence, 'sentenlen':sentenlen, 'usridx':usridx, 'proidx':proidx, 'rate':rate}       


def train(batch_size, train_data, test_data):
    test_data = generate_test_data(test_data)
    abmfer = Abmf(maxseqlen=300, word_dim = 200, rnnstate_size = 50, attention_size = 20, usr_num=6040, rank_dim=50, pro_num=3544, word_num=8000)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(abmfer.loss)
    saver = tf.train.Saver()
    step = tf.Variable(trainable=False)
    step_list = []
    loss_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iter_nums):
            for j in range(len(data) // batch_size):
                batch_data = generate_batch_data(batch_size,train_data)
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
    

        

         
        