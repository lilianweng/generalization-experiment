import json
import logging
import os
import subprocess
import time

import matplotlib.pyplot as plt
import tensorflow as tf

from intrinsic_dimensions_measurement import MAX_EPOCHS
from utils import configure_logging, make_session, prepare_mnist_dataset, create_mnist_model

os.makedirs('data', exist_ok=True)
configure_logging()


def report_full_performance(net_layers, lr=0.001, batch_size=128):
    sess = make_session()
    dataset, (_, _), (x_test, y_test) = prepare_mnist_dataset(
        batch_size=batch_size, train_sample_size=6000, test_sample_size=1000)
    iterator = dataset.make_initializable_iterator()
    batch_input, batch_label = iterator.get_next()

    lr_ph = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    loss, accuracy = create_mnist_model(batch_input, batch_label, net_layers,
                                        loss_type='cross_ent')
    train_op = tf.train.AdamOptimizer(lr_ph).minimize(loss)

    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())

    step = 0
    for epoch in range(MAX_EPOCHS):
        while True:
            try:
                sess.run(train_op, feed_dict={lr_ph: lr})
                step += 1
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
                break

        if epoch % 50 == 0:
            lr *= 0.5

    # evaluate t the end
    final_eval_acc = sess.run(accuracy, feed_dict={batch_input: x_test, batch_label: y_test})
    return final_eval_acc


def plot(filepath):
    data = json.load(open(filepath))
    full = data.pop('full')
    print("80% of full performance:", full * 0.8)

    xdata = sorted(map(int, data.keys()))
    ydata = [data[str(k)] for k in xdata]
    print(list(zip(xdata, ydata)))

    plt.figure(figsize=(8, 4))
    plt.plot(xdata, ydata, 'x-')
    plt.grid(ls='--', color='k', alpha=0.3)
    plt.axhline(y=0.8 * full, color='r', ls=':')
    plt.savefig(filepath.replace(".json", ".png"))


def explore(net_layers, max_d=1500):
    net_layers_str = '-'.join(map(str, net_layers))
    output_filename = f"data/intrinsic-dimension-net-{net_layers_str}-{int(time.time())}.json"
    logging.info(f"output_filename: {output_filename}")

    results = {}
    for d in range(1, max_d + 50, 50):
        proc = subprocess.run(
            ['python', 'intrinsic_dimensions_measurement.py',
             ','.join(map(str, net_layers)), str(d)],
            encoding='utf-8', stdout=subprocess.PIPE
        )
        last_line = proc.stdout.strip().split('\n')[-1]
        final_eval_acc = float(last_line)
        results[d] = final_eval_acc

        # back up the results in every loop
        with open(output_filename, 'w') as f:
            json.dump(results, f)

    full_acc = report_full_performance(net_layers)
    results['full'] = float(full_acc)

    print(results)
    with open(output_filename, 'w') as f:
        json.dump(results, f)

    return output_filename


if __name__ == '__main__':
    output_filename = explore([64, 64], max_d=1500)
    plot(output_filename)
