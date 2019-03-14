import tensorflow as tf
import numpy as np
import time
from utils import create_mnist_model, prepare_mnist_dataset, make_session, configure_logging
import json
import logging

configure_logging()


class LayerEqualityExperiment:

    def __init__(self, layer_sizes):
        self.output_filename = "data/layer_equality_results.json"
        self.sess = make_session()

        dataset, (_, _), (self.x_test, self.y_test) = prepare_mnist_dataset()
        self.iterator = dataset.make_initializable_iterator()
        self.batch_input, self.batch_label = self.iterator.get_next()

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) + 1  # plus the output layer
        self.loss_ph, self.accuracy_ph = create_mnist_model(
            self.batch_input, self.batch_label, layer_sizes=layer_sizes)

        self._vars_by_layer = [self._vars(f'mnist_dense_nn/mlp/mlp_l{i}')
                               for i in range(self.num_layers)]
        self._init_values_by_layer = None

        np.random.seed(int(time.time()))

    def _initialize(self, lr):
        # train_op = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(self.loss_ph)
        train_op = tf.train.AdamOptimizer(lr).minimize(self.loss_ph)

        self.sess.run(self.iterator.initializer)
        self.sess.run(tf.global_variables_initializer())

        # save the initialization values to be used later.
        self._init_values_by_layer = self.sess.run(self._vars_by_layer)

        return train_op

    def _vars(self, scope):
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        variables = sorted(variables, key=lambda v: v.name)
        return variables

    def get_layer_values(self, l):
        return self.sess.run(self._vars_by_layer[l])

    def _assign_layer_values(self, l, values):
        for i in range(len(self._vars_by_layer[l])):
            variable = self._vars_by_layer[l][i]
            value = values[i]
            assert variable.get_shape().as_list() == list(value.shape)
            self.sess.run(tf.assign(variable, value))

    def assign_init_values(self, l):
        self._assign_layer_values(l, self._init_values_by_layer[l])

    def assign_random_values(self, l):
        random_values = [np.random.random(vs.get_shape().as_list())
                         for vs in self._vars_by_layer[l]]
        self._assign_layer_values(l, random_values)

    def _get_eval_accuracy(self):
        feed_dict = {self.batch_input: self.x_test, self.batch_label: self.y_test}
        return self.sess.run(self.accuracy_ph, feed_dict=feed_dict)

    def measure_layer_robustness(self, eval_accuracy, epoch):
        results = []
        for l in range(self.num_layers):
            real_values = self.get_layer_values(l)

            # try assign initialization values to the l-th layer.
            self.assign_init_values(l)
            init_acc = self._get_eval_accuracy()
            # try assign rand values to the l-th layer.
            self.assign_random_values(l)
            rnd_acc = self._get_eval_accuracy()

            # reset the layer values to the real ones.
            self._assign_layer_values(l, real_values)

            result_dict = dict(
                epoch=epoch,
                layer=l,
                base_accuracy=eval_accuracy,
                init_accuracy=init_acc,
                random_accuracy=rnd_acc,
                diff_2norm=np.mean([
                    np.linalg.norm(np.array(x).flatten() - np.array(y).flatten())
                    for x, y in zip(real_values, self._init_values_by_layer[l])
                ])
            )
            logging.info(str(result_dict))
            results.append(result_dict)

        return results

    def train(self, n_epoches, lr):
        train_op = self._initialize(lr)
        print(">>>", tf.global_variables())

        results = []
        step = 0
        for epoch in range(n_epoches):
            while True:
                try:
                    _, train_loss, train_acc = self.sess.run(
                        [train_op, self.loss_ph, self.accuracy_ph])
                    step += 1
                except tf.errors.OutOfRangeError:
                    eval_accuracy = self._get_eval_accuracy()
                    logging.info(
                        f">>> epoch:{epoch} step:{step} train_loss:{train_loss:.4f} eval_accuracy:{eval_accuracy:.4f} lr:{lr}")
                    self.sess.run(self.iterator.initializer)
                    break

            if epoch > 0 and epoch % 10 == 0:
                lr *= 0.5
                results += self.measure_layer_robustness(eval_accuracy, epoch)
                # save to disk in every loop
                with open(self.output_filename, 'w') as fout:
                    json.dump(results, fout)


if __name__ == '__main__':
    exp = LayerEqualityExperiment([256, 256])
    exp.train(100, 0.002)

    # exp = LayerEqualityExperiment([256, 256, 256])
    # exp.train(100, ???)
