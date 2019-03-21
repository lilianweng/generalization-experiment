import click
import tensorflow as tf
import logging
from utils import configure_logging, make_session, prepare_mnist_dataset, create_mnist_model
import numpy as np
import os

configure_logging()
os.makedirs('tmp', exist_ok=True)  # for saving model weights


def load_weights(n_units):
    weights = np.load(f"tmp/mnist_unit{n_units}_weights.npy").item()
    shape_dict = {k: v.shape for k, v in weights.items()}
    logging.info(f"Loading weights from old model (n_units={n_units}): {shape_dict}")
    return weights


def dump_weights(sess, n_units):
    var_list = tf.trainable_variables()
    weights_dict = sess.run({v.name: v for v in var_list})
    np.save(f"tmp/mnist_unit{n_units}_weights.npy", weights_dict)


def assign_weights(sess, old_weights):
    var_list = tf.trainable_variables()
    assert sorted(v.name for v in var_list) == sorted(old_weights.keys())
    for v in var_list:
        w = old_weights[v.name]

        v_shape = v.shape.as_list()
        w_shape = list(w.shape)
        assert len(v_shape) == len(w_shape)

        if len(v_shape) == 2:

            if v_shape[0] == w_shape[0]:
                extra = sess.run(v)[:, w_shape[1]:]
                logging.info(f">>> {v.name} assign {w.shape} to {v} (extra: {extra.shape})")
                sess.run(tf.assign(v, np.concatenate([w, extra], axis=1)))

            elif v_shape[1] == w_shape[1]:
                extra = sess.run(v)[w_shape[0]:, :]
                logging.info(f">>> {v.name} assign {w.shape} to {v} (extra: {extra.shape})")
                sess.run(tf.assign(v, np.concatenate([w, extra], axis=0)))

        elif len(v_shape) == 1:

            if v_shape == w_shape:
                logging.info(f">>> {v.name} assign {w.shape} to {v}")
                sess.run(tf.assign(v, w))
            else:
                extra = sess.run(v)[w_shape[0]:]
                logging.info(f">>> {v.name} assign {w.shape} to {v} (extra: {extra.shape})")
                sess.run(tf.assign(v, np.concatenate([w, extra], axis=0)))


def report_performance(sess, n_units, old_n_units, max_epochs, loss_type, lr, batch_size,
                       train_sample_size):
    logging.info(
        f"Training new model (n_units={n_units}) based on old model (old_n_units={old_n_units}) ...")
    dataset, (x_train, y_train), (x_test, y_test) = prepare_mnist_dataset(
        batch_size=batch_size, train_sample_size=train_sample_size, seed=0)
    iterator = dataset.make_initializable_iterator()
    batch_input, batch_label = iterator.get_next()

    lr_ph = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    loss, accuracy = create_mnist_model(batch_input, batch_label, [n_units], loss_type=loss_type)
    train_op = tf.train.AdamOptimizer(lr_ph).minimize(loss)

    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())

    if old_n_units is not None:
        old_weights = load_weights(old_n_units)
        assign_weights(sess, old_weights)

    step = 0
    epoch = 0
    # for epoch in range(num_epoches):
    for epoch in range(max_epochs):
        while True:
            try:
                sess.run(train_op, feed_dict={lr_ph: lr})
                step += 1
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
                break

        if epoch % 50 == 0:
            lr *= 0.8

    # evaluate t the end
    eval_loss, eval_acc = sess.run(
        [loss, accuracy], feed_dict={batch_input: x_test, batch_label: y_test})

    train_loss, train_acc = sess.run([loss, accuracy], feed_dict={
        batch_input: x_train, batch_label: y_train, lr_ph: lr})

    # dump the parameters into files so that we can reuse the weights in the next round.
    dump_weights(sess, n_units)

    return epoch, step, train_loss, train_acc, eval_loss, eval_acc


@click.command()
@click.option('--n-units', default=50, type=int, help="num. hidden units in the middle layer.")
@click.option('--old-n-units', default=None, type=int, help="")
@click.option('--loss-type', default='mse', type=str, help="type of loss func.")
@click.option('--max-epochs', default=500, type=int, help="num. training epochs.")
@click.option('--n-train-samples', default=2500, type=int, help="num. training samples")
def main(n_units=1, old_n_units=None, loss_type='mse', max_epochs=500, n_train_samples=2500):
    assert old_n_units is None or old_n_units < n_units
    logging.info(f"n_units:{n_units} max_epochs:{max_epochs}")
    sess = make_session()
    lr = 0.005
    batch_size = 128

    epoch, step, train_loss, train_acc, eval_loss, eval_acc = report_performance(
        sess, n_units, old_n_units, max_epochs, loss_type, lr, batch_size, n_train_samples)
    print(n_units, epoch, step, train_loss, train_acc, eval_loss, eval_acc)


if __name__ == '__main__':
    main()
