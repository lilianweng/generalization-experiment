import logging
import time

import numpy as np
import tensorflow as tf
import sys
from utils import prepare_mnist_dataset, make_session, configure_logging

configure_logging()
MAX_EPOCHS = 100


class IntrinsicDimensionExperiment:
    """Measure intrinsic dimension in model for MNIST.
    """

    def __init__(self, d, layer_sizes, clip_norm=None):
        self.sess = make_session()
        self.clip_norm = clip_norm

        dataset, (_, _), (self.x_test, self.y_test) = prepare_mnist_dataset(
            batch_size=128, train_sample_size=5000, test_sample_size=1000)
        self.iterator = dataset.make_initializable_iterator()
        self.batch_input, self.batch_label = self.iterator.get_next()

        self.hidden_layers = layer_sizes
        self.n_layers = len(self.hidden_layers) + 1  # plus the output layer
        self.shape_per_layer = [(28 * 28, self.hidden_layers[0])] + [
            (self.hidden_layers[i], self.hidden_layers[i + 1])
            for i in range(len(self.hidden_layers) - 1)
        ] + [(self.hidden_layers[-1], 10)]
        logging.info(f"shape per layer: {self.shape_per_layer}")

        self.d = d
        self.D = np.sum((h + 1) * w for h, w in self.shape_per_layer)
        logging.info(f"Experiment config: d={d} D={self.D}")

        self.loss, self.accuracy = self.build_network(self.batch_input, self.batch_label)

        for t in tf.trainable_variables():
            print(t)

        np.random.seed(int(time.time()))

    def build_network(self, input_ph, label_ph):
        self.transforms = self.sample_transform_matrices()
        self.subspace = tf.get_variable("subspace", shape=(self.d, 1), trainable=True,
                                        initializer=tf.zeros_initializer())

        with tf.variable_scope("mnist_dense_nn", reuse=False):
            label_ohe = tf.one_hot(label_ph, 10, dtype=tf.float32)

            out = tf.reshape(tf.cast(input_ph, tf.float32), (-1, 28 * 28))

            for i, (h, w) in enumerate(self.shape_per_layer):

                if i > 0:
                    # No dropout on the input layer.
                    out = tf.nn.dropout(out, keep_prob=0.9)

                weights = tf.get_variable(f'w_{i}', shape=(h, w), trainable=False,
                                          initializer=tf.glorot_uniform_initializer())
                new_weights = tf.stop_gradient(weights) + tf.reshape(tf.matmul(
                    tf.stop_gradient(self.transforms[i][0]), self.subspace), weights.shape)

                biases = tf.get_variable(f'b_{i}', shape=(w,), trainable=False,
                                         initializer=tf.zeros_initializer())
                new_biases = tf.stop_gradient(biases) + tf.reshape(tf.matmul(
                    tf.stop_gradient(self.transforms[i][1]), self.subspace), biases.shape)

                out = tf.matmul(out, new_weights) + new_biases

                if i < self.n_layers - 1:
                    out = tf.nn.relu(out)

            logits = out
            pred_probas = tf.nn.softmax(logits)
            pred_labels = tf.cast(tf.argmax(pred_probas, 1), tf.uint8)

            loss = tf.losses.softmax_cross_entropy(label_ohe, logits)
            # loss = tf.losses.mean_squared_error(label_ohe, preds)

            accuracy = tf.reduce_sum(
                tf.cast(tf.equal(pred_labels, label_ph), tf.float32)
            ) / tf.cast(tf.shape(label_ph)[0], tf.float32)

        return loss, accuracy

    def sample_transform_matrices(self):
        """Matrix P in the paper of size (D, d)
        Columns of P are normalized to unit length.
        """
        matrix = np.random.randn(self.D, self.d).astype(np.float32)
        for i in range(self.d):
            # each column is normalized to have unit 1.
            matrix[:, i] /= np.linalg.norm(matrix[:, i])

        # split P according to num. params per layer.
        w_matrices = []
        b_matrices = []
        offset = 0
        for i, (h, w) in enumerate(self.shape_per_layer):
            w_matrix_values = matrix[offset:offset + h * w]
            w_matrix = tf.Variable(w_matrix_values, dtype=tf.float32, name=f'w_matrix_{i}',
                                   trainable=False)
            w_matrices.append(w_matrix)  # for weights
            offset += h * w

            b_matrix_values = matrix[offset:offset + w]
            b_matrix = tf.Variable(b_matrix_values, dtype=tf.float32, name=f'b_matrix_{i}',
                                   trainable=False)
            b_matrices.append(b_matrix)  # for weights
            offset += w

        logging.info(f"shape of transform matrices for weights: {[m.shape for m in w_matrices]}")
        logging.info(f"shape of transform matrices for biases: {[m.shape for m in b_matrices]}")
        return list(zip(w_matrices, b_matrices))

    def _initialize(self):
        self.sess.run(self.iterator.initializer)
        self.sess.run(tf.global_variables_initializer())

    def _get_eval_results(self):
        feed_dict = {self.batch_input: self.x_test, self.batch_label: self.y_test}
        return self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

    def _get_train_op(self, lr):
        optimizer = tf.train.AdamOptimizer(lr)
        grads_tvars = optimizer.compute_gradients(self.loss, var_list=[self.subspace])

        if self.clip_norm:
            grads_tvars = [(tf.clip_by_norm(g, self.clip_norm), v) for g, v in grads_tvars]

        train_op = optimizer.apply_gradients(grads_tvars)
        return train_op, grads_tvars

    def train(self, lr):
        lr_ph = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        train_op, grads_tvars = self._get_train_op(lr_ph)
        self._initialize()

        step = 0
        epoch = 0
        while epoch <= MAX_EPOCHS:

            while True:
                try:
                    _, loss, acc = self.sess.run([train_op, self.loss, self.accuracy],
                                                 feed_dict={lr_ph: lr})
                    step += 1
                    if step % 100 == 0:
                        logging.info(f"[step:{step}|epoch:{epoch}] loss={loss} acc={acc}")

                except tf.errors.OutOfRangeError:
                    # complete one training epoch.
                    self.sess.run(self.iterator.initializer)
                    break

            epoch += 1
            if epoch % 50 == 0:
                lr *= 0.5

        _, final_eval_acc = self._get_eval_results()
        logging.info(f"[final] d={self.d} eval_acc={final_eval_acc}")
        return epoch, final_eval_acc


if __name__ == '__main__':
    net_layers = list(map(int, sys.argv[1].split(',')))
    d = int(sys.argv[2])
    logging.info(f"Testing net_layers={net_layers} d={d} ")

    exp = IntrinsicDimensionExperiment(d, net_layers)
    _, final_eval_acc = exp.train(0.001)
    print(final_eval_acc)
