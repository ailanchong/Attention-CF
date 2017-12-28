import tensorflow
import numpy

class Arnn(object):
    def __init__(self):
        self

    def alstm_layer(inputs, lengths, state_size, keep_prob=1.0,
         scope = 'lstm-layer', reuse=False, return_final_state = False):
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
            if return_final_state:
                return outputs, output_state
            else:
                return outputs
    
