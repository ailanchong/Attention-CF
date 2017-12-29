import tensorflow
import numpy
from utils import attention
class Arnn(object):
    def __init__(self):
        self.attention_size = 50

    def alstm_layer(self, inputs, lengths, state_size, keep_prob=1.0,
         scope = 'lstm-layer', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            cell = tf.contirb.rnn.DropoutWrapper(
                tf.contirb.rnn.LSTMCell(
                    state_size,
                    reuse=reuse
                ),
                output_keep_prob=keep_prob
            )
            outputs, output_state = tf.nn.dynamic_rnn(
                inputs=inputs,
                cell=cell,
                sequence_length=lengths,
                dtype=tf.float32
            )
            outputs = attention(outputs, self.attention_size, time_major=False, return_alphas=False):
            return outputs


    def abmf_layer(self, inputs, )

    



    
    
