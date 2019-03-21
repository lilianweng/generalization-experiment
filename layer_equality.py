import logging
import time
import numpy as np
import tensorflow as tf
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import create_mnist_model, prepare_mnist_dataset, make_session, configure_logging

configure_logging()

DEFAULT_OUTPUT_FILENAME = "data/layer_equality_results.json"


class LayerEqualityExperiment:
    default_output_filename = "data/layer_equality_results.json"

    def __init__(self, layer_sizes: list, output_filename: str = DEFAULT_OUTPUT_FILENAME):
        self.output_filename = output_filename
        self.sess = make_session()

        dataset, (_, _), (self.x_test, self.y_test) = prepare_mnist_dataset(
            train_sample_size=6000, test_sample_size=1000)
        self.iterator = dataset.make_initializable_iterator()
        self.batch_input, self.batch_label = self.iterator.get_next()

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) + 1  # plus the output layer
        self.loss_ph, self.accuracy_ph = create_mnist_model(
            self.batch_input, self.batch_label, layer_sizes=layer_sizes,
            dropout_keep_prob=0.9)

        self._vars_by_layer = [self._vars(f'mnist_dense_nn/mlp/mlp_l{i}')
                               for i in range(self.num_layers)]
        self._init_values_by_layer = None

        np.random.seed(int(time.time()))

    def _initialize(self, lr):
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
                epoch=int(epoch),
                layer=int(l),
                base_accuracy=float(eval_accuracy),
                init_accuracy=float(init_acc),
                random_accuracy=float(rnd_acc),
                diff_2norm=float(np.mean([
                    np.linalg.norm(np.array(x).flatten() - np.array(y).flatten())
                    for x, y in zip(real_values, self._init_values_by_layer[l])
                ]))
            )
            logging.info(str(result_dict))
            results.append(result_dict)

        return results

    def train(self, n_epoches, lr):
        train_op = self._initialize(lr)

        results = []
        step = 0
        for epoch in range(1, n_epoches + 1):
            while True:
                try:
                    _, train_loss, train_acc = self.sess.run(
                        [train_op, self.loss_ph, self.accuracy_ph])
                    step += 1
                except tf.errors.OutOfRangeError:
                    eval_accuracy = self._get_eval_accuracy()
                    logging.info(f">>> epoch:{epoch} step:{step} train_loss:{train_loss:.4f} "
                                 f"eval_accuracy:{eval_accuracy:.4f} lr:{lr}")
                    self.sess.run(self.iterator.initializer)
                    break

            if epoch % 30 == 0:
                lr *= 0.5

            if epoch % 10 == 0:
                results += self.measure_layer_robustness(eval_accuracy, epoch)
                # save to disk in every loop
                print(results)
                with open(self.output_filename, 'w') as fout:
                    json.dump(results, fout)

    def plot(self):
        data = json.load(open(self.output_filename))
        max_layers = max(x['layer'] for x in data) + 1
        logging.info(f"max_layers: {max_layers}")

        epoch = [[x['epoch'] for x in data if x['layer'] == l] for l in range(max_layers)]
        init_acc = [[x['init_accuracy'] * 100.0 for x in data if x['layer'] == l] for l in
                    range(max_layers)]
        random_acc = [[x['random_accuracy'] * 100.0 for x in data if x['layer'] == l] for l in
                      range(max_layers)]
        norm2 = [[x['diff_2norm'] for x in data if x['layer'] == l] for l in range(max_layers)]

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))

        ax1.set_title("2-norm diff(init, current)")
        for i in range(max_layers):
            ax1.plot(epoch[i], norm2[i], '-', marker=i, label=f'Layer {i}')
        ax1.grid(ls='--', color='k', alpha=0.3)
        ax1.set_xlabel('num. training epoch')
        ax1.set_ylabel('2-norm distance to initial values')
        ax1.legend(frameon=False)

        ax2.set_title("Re-randomization robustness")
        for i in range(max_layers):
            ax2.plot(epoch[i], random_acc[i], marker=i, label=f'Layer {i}')
        ax2.set_xlabel('num. training epoch')
        ax2.set_ylabel('test accuracy (%)')
        ax2.set_ylim(0.0, 100.0)
        ax2.grid(ls='--', color='k', alpha=0.3)

        ax3.set_title("Re-initialization robustness")
        for i in range(max_layers):
            ax3.plot(epoch[i], init_acc[i], marker=i, label=f'Layer {i}')
        ax3.grid(ls='--', color='k', alpha=0.3)
        ax3.set_xlabel('num. training epoch')
        ax3.set_ylabel('test accuracy (%)')
        ax3.set_ylim(0.0, 100.0)
        ax3.legend(frameon=False)

        fig.savefig(self.output_filename.replace('.json', '.png'))


if __name__ == '__main__':
    # exp1 = LayerEqualityExperiment(
    #     [256, 256, 256], output_filename="data/layer_equality_256x3.json")
    # exp1.train(100, 0.0005)
    # exp1.plot()

    exp2 = LayerEqualityExperiment(
        [128, 128, 128, 128], output_filename="data/layer_equality_128x4.json")
    exp2.train(100, 0.0005)
    exp2.plot()
