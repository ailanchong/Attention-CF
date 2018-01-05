import tensorflow
import numpy
from tf_utils import alstm_layer

class Abmf(object):
    def __init__(self, maxseqlen, word_dim, rnnstate_size, attention_size, usr_num, rank_dim, pro_num, word_num):
        """
        rnnstate_size should be equal with rank_dim
        """
        self.input_seq = tf.placeholder(dtype=tf.float32, shape=[None, maxseqlen])
        self.input_seqlen = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.usr = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.pro = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.rate = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        with tf.name_scope("embedding_layer"):
            self.word_matrix = tf.Variable(tf.truncated_normal([word_num+1, word_dim]), name="word_matrix", dtype=tf.float32)
            self.input_setence = tf.nn.embedding_lookup(self.word_matrix, self.input_seq)

        with tf.name_scope("attention_rnn"):
            rnn_output = alstm_layer(self.input_setence, self.input_seqlen, rnnstate_size, attention_size)
        with tf.name_scope("matrix_factorization"):
            self.usr_matrix = tf.Variable(tf.truncated_normal([usr_num+1, rank_dim]), name="usr_matrix", dtype=tf.float32)
            self.usr_bias = tf.Variable(tf.truncated_normal([usr_num+1]), name="usr_bias", dtype=tf.float32)
            self.pro_matrix = tf.Variable(tf.truncated_normal([pro_num+1, rank_dim]), name="pro_matrix", dtype=tf.float32)
            self.pro_bias = tf.Variable(tf.truncated_normal([pro_num+1]), name="pro_bias", dtype=tf.float32)
            self.global_mean = tf.Variable(0.0, name="global_mean", dtype=tf.float32)
            currusr_matrix = tf.nn.embedding_lookup(self.usr_matrix, self.usr)
            currusr_bias = tf.nn.embedding_lookup(self.usr_bias, self.usr)
            currpro_matrix = tf.nn.embedding_lookup(self.pro_matrix, self.pro)
            currpro_bias = tf.nn.embedding_lookup(self.pro_bias, self.pro)
            
        with tf.name_scope("predict"):
            currpro_matrix = tf.add(currpro_matrix, rnn_output)
            interaction = tf.reduce_mean(tf.multiply(currpro_matrix, currusr_matrix), 1)
            self.predicts = interaction + currusr_bias + currpro_bias + self.global_mean
        
        with tf.name_scope("loss"):
            self.loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.predicts, self.rate)))





