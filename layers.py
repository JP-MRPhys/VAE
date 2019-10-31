import tensorflow as tf
import numpy as np


def get_weights(shape, name, horizontal, mask_mode='noblind', mask=None):
    weights_initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name, shape, tf.float32, weights_initializer)

    '''
        Use of masking to hide subsequent pixel values 
    '''
    if mask:
        filter_mid_y = shape[0] // 2
        filter_mid_x = shape[1] // 2
        mask_filter = np.ones(shape, dtype=np.float32)
        if mask_mode == 'noblind':
            if horizontal:
                # All rows after center must be zero
                mask_filter[filter_mid_y + 1:, :, :, :] = 0.0
                # All columns after center in center row must be zero
                mask_filter[filter_mid_y, filter_mid_x + 1:, :, :] = 0.0
            else:
                if mask == 'a':
                    # In the first layer, can ONLY access pixels above it
                    mask_filter[filter_mid_y:, :, :, :] = 0.0
                else:
                    # In the second layer, can access pixels above or even with it.
                    # Reason being that the pixels to the right or left of the current pixel
                    #  only have a receptive field of the layer above the current layer and up.
                    mask_filter[filter_mid_y + 1:, :, :, :] = 0.0

            if mask == 'a':
                # Center must be zero in first layer
                mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.0
        else:
            mask_filter[filter_mid_y, filter_mid_x + 1:, :, :] = 0.
            mask_filter[filter_mid_y + 1:, :, :, :] = 0.

            if mask == 'a':
                mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.

        W *= mask_filter
    return W


def get_bias(shape, name):
    return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer)


def conv_op(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class GatedCNN():

    def __init__(self, W_shape, fan_in, horizontal, gated=True, payload=None, mask=None, activation=True,
                 conditional=None, conditional_image=None):

        self.fan_in = fan_in
        in_dim = self.fan_in.get_shape()[-1]
        self.W_shape = [W_shape[0], W_shape[1], in_dim, W_shape[2]]
        self.b_shape = W_shape[2]

        self.in_dim = in_dim
        self.payload = payload
        self.mask = mask
        self.activation = activation
        self.conditional = conditional
        self.conditional_image = conditional_image
        self.horizontal = horizontal

        if gated:
            self.gated_conv()
        else:
            self.simple_conv()

    def gated_conv(self):
        W_f = get_weights(self.W_shape, "v_W", self.horizontal, mask=self.mask)
        W_g = get_weights(self.W_shape, "h_W", self.horizontal, mask=self.mask)

        b_f_total = get_bias(self.b_shape, "v_b")
        b_g_total = get_bias(self.b_shape, "h_b")
        if self.conditional is not None:
            h_shape = int(self.conditional.get_shape()[1])
            V_f = get_weights([h_shape, self.W_shape[3]], "v_V", self.horizontal)
            b_f = tf.matmul(self.conditional, V_f)
            V_g = get_weights([h_shape, self.W_shape[3]], "h_V", self.horizontal)
            b_g = tf.matmul(self.conditional, V_g)

            b_f_shape = tf.shape(b_f)
            b_f = tf.reshape(b_f, (b_f_shape[0], 1, 1, b_f_shape[1]))
            b_g_shape = tf.shape(b_g)
            b_g = tf.reshape(b_g, (b_g_shape[0], 1, 1, b_g_shape[1]))

            b_f_total = b_f_total + b_f
            b_g_total = b_g_total + b_g
        if self.conditional_image is not None:
            b_f_total = b_f_total + tf.layers.conv2d(self.conditional_image, self.in_dim, 1, use_bias=False,
                                                     name="ci_f")
            b_g_total = b_g_total + tf.layers.conv2d(self.conditional_image, self.in_dim, 1, use_bias=False,
                                                     name="ci_g")

        conv_f = conv_op(self.fan_in, W_f)
        conv_g = conv_op(self.fan_in, W_g)

        if self.payload is not None:
            conv_f += self.payload
            conv_g += self.payload

        self.fan_out = tf.multiply(tf.tanh(conv_f + b_f_total), tf.sigmoid(conv_g + b_g_total))

    def simple_conv(self):
        W = get_weights(self.W_shape, "W", self.horizontal, mask_mode="standard", mask=self.mask)
        b = get_bias(self.b_shape, "b")
        conv = conv_op(self.fan_in, W)
        if self.activation:
            self.fan_out = tf.nn.relu(tf.add(conv, b))
        else:
            self.fan_out = tf.add(conv, b)

    def output(self):
        return self.fan_out


class ConvolutionalEncoder(object):
    def __init__(self, X, conf):
        '''
            This is the 6-layer architecture for Convolutional Autoencoder
            mentioned in the original paper:
            Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction
            Note that only the encoder part is implemented as PixelCNN is taken
            as the decoder.
        '''

        W_conv1 = get_weights([5, 5, conf.channel, 100], "W_conv1")
        b_conv1 = get_bias([100], "b_conv1")
        conv1 = tf.nn.relu(conv_op(X, W_conv1) + b_conv1)
        pool1 = max_pool_2x2(conv1)

        W_conv2 = get_weights([5, 5, 100, 150], "W_conv2")
        b_conv2 = get_bias([150], "b_conv2")
        conv2 = tf.nn.relu(conv_op(pool1, W_conv2) + b_conv2)
        pool2 = max_pool_2x2(conv2)

        W_conv3 = get_weights([3, 3, 150, 200], "W_conv3")
        b_conv3 = get_bias([200], "b_conv3")
        conv3 = tf.nn.relu(conv_op(pool2, W_conv3) + b_conv3)
        conv3_reshape = tf.reshape(conv3, (-1, 7*7*200))

        W_fc = get_weights([7*7*200, 10], "W_fc")
        b_fc = get_bias([10], "b_fc")
        self.pred = tf.nn.softmax(tf.add(tf.matmul(conv3_reshape, W_fc), b_fc))



# Invertible 1x1 conv
def invertible_1x1_conv(z, logdet, forward=True):
    # Shape
    h,w,c = z.shape[1:]
    # Sample a random orthogonal matrix to initialise weights
    w_init = np.linalg.qr(np.random.randn(c,c))[0]
    w = tf.get_variable("W", initializer=w_init)
    # Compute log determinant
    dlogdet = h * w * tf.log(abs(tf.matrix_determinant(w)))
    if forward:
    # Forward computation
        _w = tf.reshape(w, [1,1,c,c])
        z = tf.nn.conv2d(z, _w, [1,1,1,1], 'SAME')
        logdet += dlogdet
        #11
        return z, logdet
    else:
        # Reverse computation
        _w = tf.matrix_inverse(w)
        _w = tf.reshape(_w, [1,1,c,c])
        z = tf.nn.conv2d(z, _w, [1,1,1,1], 'SAME')
        logdet -= dlogdet
        return z, logdet
