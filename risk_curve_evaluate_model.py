import click
import tensorflow as tf
import logging
from utils import configure_logging, make_session, prepare_mnist_dataset, dense_nn

configure_logging()


def weigh_reuse():
    pass


def report_performance(sess, num_units, num_epoches, lr, batch_size, dataset_ratio):
    dataset, (x_train, y_train), (x_test, y_test) = prepare_mnist_dataset(
        batch_size=batch_size, ratio=dataset_ratio)
    iterator = dataset.make_initializable_iterator()
    batch_input, batch_label = iterator.get_next()

    loss_mse, accuracy = create_mnist_model(batch_input, batch_label, [num_units])
    train_op = tf.train.AdamOptimizer(lr).minimize(loss_mse)

    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())

    step = 0
    for epoch in range(num_epoches):
        while True:
            try:
                sess.run(train_op)
                step += 1
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
                break

        eval_mse_, eval_acc_, eval_mse_ = sess.run(
            [loss_mse, accuracy],
            feed_dict={batch_input: x_test, batch_label: y_test}
        )
        train_mse_, train_acc_, train_mse_ = sess.run(
            [loss_mse, accuracy],
            feed_dict={batch_input: x_train, batch_label: y_train}
        )
        print(f"Complete epoch:{epoch} step:{step} eval_accuracy:{eval_acc_}")

    return train_mse_, train_acc_, eval_mse_, eval_acc_


@click.command()
@click.option('--n-units', default=1, type=int, help="num. hidden units in the middle layer.")
def main(n_units=1, n_epoches=1):
    logging.info(f"n_units:{n_units} n_epoches:{n_epoches}")
    sess = make_session()
    lr = 0.005
    dataset_ratio = 0.1
    n_epoches = 10
    batch_size = 64
    train_mse, train_acc, eval_mse, eval_acc = report_performance(
        sess, n_units, n_epoches, lr, batch_size, dataset_ratio)
    print(n_units, train_mse, train_acc, eval_mse, eval_acc)


if __name__ == '__main__':
    main()
