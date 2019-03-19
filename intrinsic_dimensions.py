import logging
import time

import numpy as np
import tensorflow as tf

from utils import prepare_mnist_dataset, make_session, configure_logging

configure_logging()


class IntrinsicDimensionExperiment:

    def __init__(self, d, layer_sizes):
        self.sess = make_session()

        dataset, (_, _), (self.x_test, self.y_test) = prepare_mnist_dataset()
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

        np.random.seed(int(time.time()))

    def build_network(self, input_ph, label_ph):
        self.transforms = self.sample_transform_matrices()
        self.subspace = tf.get_variable("subspace", shape=(self.d,1),
                                        initializer=tf.zeros_initializer())

        with tf.variable_scope("mnist_dense_nn", reuse=False):
            label_ohe = tf.one_hot(label_ph, 10, dtype=tf.float32)

            out = tf.reshape(tf.cast(input_ph, tf.float32), (-1, 28 * 28))

            for i, (h, w) in enumerate(self.shape_per_layer):
                weights = tf.get_variable(f'w{i}', shape=(h, w))
                weights = weights + tf.reshape(tf.matmul(
                    self.transforms[i][0], self.subspace), weights.shape)

                biases = tf.get_variable(f'b{i}', shape=(w,), initializer=tf.zeros_initializer())
                biases = biases + tf.reshape(tf.matmul(
                    self.transforms[i][1], self.subspace), biases.shape)

                out = tf.matmul(out, weights) + biases

                if i < self.n_layers - 1:
                    out = tf.nn.relu(out)

            preds = tf.nn.softmax(out)

            loss = tf.losses.softmax_cross_entropy(label_ohe, preds)
            accuracy = tf.reduce_sum(label_ohe * preds) / tf.cast(
                tf.shape(label_ohe)[0], tf.float32)

        return loss, accuracy

    def sample_transform_matrices(self):
        """Matrix P in the paper of size (D, d)
        Columns of P are normalized to unit length.
        """
        matrix = np.random.random((self.D, self.d)).astype(np.float32)
        for i in range(self.d):
            # each column is normalized to have unit 1.
            matrix[:, i] /= np.sum(matrix[:, i])

        # split P according to num. params per layer.
        w_matrices = []
        b_matrices = []
        offset = 0
        for h, w in self.shape_per_layer:
            w_matrices.append(matrix[offset:offset + h * w])  # for weights
            offset += h * w

            b_matrices.append(matrix[offset:offset + w])  # for weights
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
        return tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=self.subspace)

    def train(self, n_epoches=10, lr=0.005):
        train_op = self._get_train_op(lr)
        self._initialize()

        step = 0
        eval_acc = 0.0
        for epoch in range(n_epoches):
            while True:
                try:
                    _, loss = self.sess.run([train_op, self.loss])
                    step += 1
                    if step % 100 == 0:
                        logging.info(f"[step:{step}] loss={loss}")
                except tf.errors.OutOfRangeError:
                    eval_loss, eval_acc = self._get_eval_results()
                    logging.info(f"[epoch:{epoch}|step:{step}] eval_loss:{eval_loss} "
                                 f"eval_acc:{eval_acc}")
                    self.sess.run(self.iterator.initializer)
                    break

            lr *= 0.8

        return eval_acc


if __name__ == '__main__':
    d = 100
    exp = IntrinsicDimensionExperiment(d, [512])
    exp.train(lr=0.005)
