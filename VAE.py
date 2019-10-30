import tensorflow as tf
import numpy as np
import pathlib
import time
import os, os.path
import matplotlib.pyplot as plt

tf.enable_eager_execution()


def get_filenames_jpg(directory):
    results = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            fullpath = os.path.join(root, f)
            if os.path.splitext(fullpath)[1] == '.jpg':
                results.append(fullpath)

    return results


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)

    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])  # 0 for rgb
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('./images/VAE_medium_image_at_epoch_{:04d}.png'.format(epoch))
    plt.show(block=False)

    # plt.close()


def parse_jpg(filename):
    print(filename)

    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string)
    image_resize = tf.image.resize_images(image, size=[128, 128])

    # split0, split1, split2 = tf.split(image_resize, num_or_size_splits=3, axis=2)
    image_resize = tf.image.per_image_standardization(image_resize)

    return image_resize


def parse_mnits(data):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string)
    image_resize = tf.image.resize_images(image, size=[28, 28])

    # split0, split1, split2 = tf.split(image_resize, num_or_size_splits=3, axis=2)

    return image_resize


def resize_mnist(image):
    return tf.image.resize(image, [128, 128])


class CVAE(tf.keras.Model):

    def __init__(self, latent_dim):

        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.channels = 3
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 128, self.channels)),  # 224,224,1
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),  # 112,112,32
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),  # 56,56,64
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=(2, 2), activation='relu'),  # 28,28,128
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=8 * 8 * 128, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(8, 8, 128)),

                tf.keras.layers.Conv2DTranspose(  # 56,56,64
                    filters=256,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),

                tf.keras.layers.Conv2DTranspose(  # 56,56,64
                    filters=128,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),

                tf.keras.layers.Conv2DTranspose(  # 56,56,64
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),

                tf.keras.layers.Conv2DTranspose(  # 56,56,64
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),

                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=self.channels, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

    # @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits


if __name__ == '__main__':

    epochs = 100
    latent_dim = 1000
    num_examples_to_generate = 16

    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])

    model = CVAE(latent_dim)

    print("Inference net")
    for layer in model.inference_net.layers:
        # print("Input: " + str(layer.input))
        print("Output: " + str(layer.output))

    print("Generative model")
    for layer in model.generative_net.layers:
        # print("Input: " + str(layer.input))
        print("Output: " + str(layer.output))

    datadir = '/media/jehill/DATA/ML_data/celebA/celeba-dataset/img_align_celeba/'
    filenames = get_filenames_jpg(datadir)
    print(len(filenames))

    train_filenames = filenames[0:20000]
    test_filenames = filenames[20000:25000]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_filenames).shuffle((len(train_filenames))).map(
        parse_jpg).batch(100)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_filenames).shuffle((len(test_filenames))).map(
        parse_jpg).batch(100)

    """

    #generate_and_save_images(model, 0, random_vector_for_generation)

    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()



    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')


    print(train_images.shape)

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    TRAIN_BUF = 60000
    BATCH_SIZE = 50
    TEST_BUF = 10000

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).map(resize_mnist).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).map(resize_mnist).shuffle(TEST_BUF).batch(BATCH_SIZE)

    """

    for epoch in range(1, epochs + 1):

        start_time = time.time()
        print(str(start_time))
        for train_x in train_dataset:
            compute_apply_gradients(model, train_x, optimizer)

        end_time = time.time()
        print('Epoch: {} time elapse for current epoch {}'.format(epoch, end_time - start_time))

        if epoch % 1 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(model, test_x))
            elbo = -loss.result()
            # display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, '
                  'time elapse for current epoch {}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))

        if epoch % 3 == 0:
            generate_and_save_images(
                model, epoch, random_vector_for_generation)



            












