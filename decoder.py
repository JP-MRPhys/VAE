from layers import *
import argparse


class PixelCNN(object):
    def __init__(self, X, conf, full_horizontal=True, h=None):
        self.X = X
        if conf.data == "mnist":
            self.X_norm = X
        else:
            '''
                Image normalization for CIFAR-10 was supposed to be done here
            '''
            self.X_norm = X

        v_stack_in, h_stack_in = self.X_norm, self.X_norm

        if conf.conditional is True:
            if h is not None:
                self.h = h
            else:
                self.h = tf.placeholder(tf.float32, shape=[None, conf.num_classes])
        else:
            self.h = None

        for i in range(conf.layers):


            filter_size = 3 if i > 0 else 7
            mask = 'b' if i > 0 else 'a'
            residual = True if i > 0 else False
            i = str(i)

            print("Layer  :" + i)
            with tf.variable_scope("v_stack"+i):
                v_stack = GatedCNN([filter_size, filter_size, conf.f_map], v_stack_in, False, mask=mask, conditional=self.h).output()
                v_stack_in = v_stack

            with tf.variable_scope("v_stack_1"+i):
                v_stack_1 = GatedCNN([1, 1, conf.f_map], v_stack_in, False, gated=False, mask=None).output()

            with tf.variable_scope("h_stack"+i):
                h_stack = GatedCNN([filter_size if full_horizontal else 1, filter_size, conf.f_map], h_stack_in, True, payload=v_stack_1, mask=mask, conditional=self.h).output()

            with tf.variable_scope("h_stack_1"+i):
                h_stack_1 = GatedCNN([1, 1, conf.f_map], h_stack, True, gated=False, mask=None).output()
                if residual:
                    h_stack_1 += h_stack_in # Residual connection
                h_stack_in = h_stack_1

        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, conf.f_map], h_stack_in, True, gated=False, mask='b').output()

        if conf.data == "mnist":
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, 1], fc1, True, gated=False, mask='b', activation=False).output()
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2, labels=self.X))
            self.pred = tf.nn.sigmoid(self.fc2)
        else:
            color_dim = 256
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, conf.channel * color_dim], fc1, True, gated=False, mask='b', activation=False).output()
                self.fc2 = tf.reshape(self.fc2, (-1, color_dim))

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.fc2, tf.cast(tf.reshape(self.X, [-1]), dtype=tf.int32)))

            '''
                Since this code was not run on CIFAR-10, I'm not sure which 
                would be a suitable way to generate 3-channel images. Below are
                the 2 methods which may be used, with the first one (self.pred)
                being more likely.
            '''
            self.pred_sampling = tf.reshape(tf.multinomial(tf.nn.softmax(self.fc2), num_samples=1, seed=100), tf.shape(self.X))
            self.pred = tf.reshape(tf.argmax(tf.nn.softmax(self.fc2), dimension=tf.rank(self.fc2) - 1), tf.shape(self.X))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--f_map', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--grad_clip', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--samples_path', type=str, default='samples')
    parser.add_argument('--summary_path', type=str, default='logs')
    parser.add_argument('conditional', type=str, default='False')

    conf = parser.parse_args()



