import tensorflow as tf
import numpy
from abmf import Abmf
import matplotlib.pyplot as plt
def prepare_data():
    """
    """


def train():
    abmfer = Abmf()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(abmfer.loss)
    saver = tf.train.Saver()
    step = tf.Variable(trainable=False)
    step_list = []
    loss_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iter_num):
            #generate batch_size
                _, step_num, loss = sess.run([optimizer, step, abmfer.loss], feed_dict={
                    abmfer.input_seq:,
                    abmfer.input_seqlen:,
                    abmfer.usr:,
                    abmfer.pro:,
                    abmfer.rate:
                })
                print("batch_step: %s, train_rmse: %s" %(step_num, loss))
                step_list.append(step_num)
                loss_list.append(loss)

            #plot 画图！！！
            plt.plot(step_list, loss_list)
            plt.savefig("abmf")
            #验证集损失
            test_loss = sess.run([abmfer.loss], feed_dict={
                abmfer.input_seq:,
                abmfer.input_seqlen:,
                abmfer.usr:,
                abmfer.pro:,
                abmfer.rate:
            })
            print("iter_step: %s, test_rmse: %s" %(i, test_loss))
            #模型保存
            saver.save(sess, "./model/abmf.ckpt")
    

        

         
        