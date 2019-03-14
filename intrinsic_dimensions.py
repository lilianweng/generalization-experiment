import logging
import time

import numpy as np
import tensorflow as tf

from utils import (create_mnist_model, prepare_mnist_dataset,
                   make_session, configure_logging, tf_flat_vars)

configure_logging()


class IntrinsicDimensionExperiment:

    def __init__(self, d, layer_sizes):
        self.sess = make_session()

        dataset, (_, _), (self.x_test, self.y_test) = prepare_mnist_dataset()
        self.iterator = dataset.make_initializable_iterator()
        self.batch_input, self.batch_label = self.iterator.get_next()

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) + 1  # plus the output layer
        self.loss_ph, self.accuracy_ph = create_mnist_model(
            self.batch_input, self.batch_label, layer_sizes=layer_sizes)

        self.d = d
        self.D = np.sum(np.prod(vs.get_shape().as_list()) for vs in tf.trainable_variables())
        logging.info(f"Experiment config: d={d} D={self.D}")

        np.random.seed(int(time.time()))

        self._transform_matrix_ph = tf.placeholder(tf.float64, shape=(self.D, self.d))
        self._dimension_indices_ph = tf.placeholder(tf.int32, shape=(self.d,))

    def _initialize(self):
        self.sess.run(self.iterator.initializer)
        self.sess.run(tf.global_variables_initializer())

    def _get_eval_results(self, feed_dict):
        feed_dict.update({self.batch_input: self.x_test, self.batch_label: self.y_test})
        return self.sess.run([self.loss_ph, self.accuracy_ph], feed_dict=feed_dict)

    def sample_transform_matrix(self):
        """Matrix P in the paper of size (seld.D, d
        """
        return np.random.random((self.D, self.d))

    def sample_selected_dimensions(self):
        indices = np.random.choice(range(self.D), size=self.d, replace=False)
        indices.sort()
        return indices

    def flatten_variable_values(self):
        return self.sess.run(tf_flat_vars(tf.trainable_variables()))

    def gradient_update(self, lr):
        var_list = tf.trainable_variables()

        optimizer = tf.train.AdamOptimizer(lr)
        grads_and_tvars = optimizer.compute_gradients(self.loss_ph, var_list)
        grads, var_list = list(zip(*grads_and_tvars))

        # flatten gradients
        flat_grads = tf_flat_vars(grads)
        selected_grads = tf.gather(flat_grads, self._dimension_indices_ph)
        selected_grads = tf.expand_dims(selected_grads, 1)
        logging.info(f"transform_matrix.shape:{self._transform_matrix_ph.shape} "
                     f"selected_grads.shape:{selected_grads.shape}")
        flat_new_grads = tf.matmul(self._transform_matrix_ph, selected_grads)

        # assign flatten new gradients to the actual gradient variables
        offset = 0
        assign_ops = []
        new_grads = [tf.Variable(np.zeros(g.get_shape().as_list())) for g in grads]
        for g in new_grads:
            g_shape = g.get_shape().as_list()
            g_size = np.prod(g_shape)
            assign_op = tf.assign(g, tf.reshape(flat_new_grads[offset:offset + g_size], g_shape))
            offset += g_size
            assign_ops.append(assign_op)

        logging.info(f">>> assign_ops: {assign_ops}")

        with tf.control_dependencies(assign_ops):
            new_grads_tvars = zip(new_grads, var_list)
            train_op = optimizer.apply_gradients(new_grads_tvars)

        return train_op

    def train(self, n_epoches=10, lr=0.005):
        train_op = self.gradient_update(lr)
        self._initialize()

        matrix = self.sample_transform_matrix()
        indices = self.sample_selected_dimensions()
        logging.info(f"matrix[0,:]={matrix[0, :]} indices[:10]={indices[:10]}")

        feed_dict = {self._transform_matrix_ph: matrix, self._dimension_indices_ph: indices}

        step = 0
        eval_acc = 0.0
        for epoch in range(n_epoches):
            while True:
                try:
                    self.sess.run(train_op, feed_dict=feed_dict)
                    step += 1
                    if step % 100 == 0:
                        eval_loss, eval_acc = self._get_eval_results(feed_dict)
                        logging.info(f"epoch:{epoch} step:{step} eval_loss:{eval_loss} "
                                     f"eval_acc:{eval_acc}")
                except tf.errors.OutOfRangeError:
                    self.sess.run(self.iterator.initializer)
                    break

            lr *= 0.8

        return eval_acc


if __name__ == '__main__':
    d = 5000
    exp = IntrinsicDimensionExperiment(d, [512])
    exp.train()
