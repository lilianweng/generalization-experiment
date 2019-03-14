import click
import tensorflow as tf
import logging
from utils import configure_logging, make_session, prepare_mnist_dataset, create_mnist_model

configure_logging()


def report_performance(sess, num_units, num_epoches, lr=0.005):
    dataset, (_, _), (x_test, y_test) = prepare_mnist_dataset()
    iterator = dataset.make_initializable_iterator()
    batch_input, batch_label = iterator.get_next()

    loss, accuracy = create_mnist_model(batch_input, batch_label, layer_sizes=[num_units])
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())

    step = 0
    for epoch in range(num_epoches):
        while True:
            try:
                _, train_loss_, train_acc_ = sess.run([train_op, loss, accuracy])
                # if step % 100 == 0:
                #    print(f"[{step}] train loss: {train_loss_} eval accuracy: {eval_acc_}")
                step += 1
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
                break

        eval_feed_dict = {batch_input: x_test, batch_label: y_test}
        eval_loss_, eval_acc_ = sess.run([loss, accuracy], feed_dict=eval_feed_dict)
        print(f"Complete epoch:{epoch} step:{step} eval_accuracy:{eval_acc_}")

    return train_loss_, train_acc_, eval_loss_, eval_acc_


@click.command()
@click.option('--n-units', default=1, type=int, help="num. hidden units in the middle layer.")
@click.option('--n-epoches', default=1, type=int, help="num. training epoches.")
def main(n_units=1, n_epoches=1):
    logging.info(f"n_units:{n_units} n_epoches:{n_epoches}")

    sess = make_session()
    train_loss, train_acc, eval_loss, eval_acc = report_performance(sess, n_units, n_epoches)
    print(n_units, train_loss, train_acc, eval_loss, eval_acc, 1.0 - eval_acc)


if __name__ == '__main__':
    main()
