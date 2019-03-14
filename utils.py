import logging
import multiprocessing
import subprocess
import tensorflow as tf
import numpy as np


def colorize(text, color):
    COLOR_MAPPINGS = dict(gray=30, red=31, green=32, yellow=33, blue=34,
                          magenta=35, cyan=36, white=37, crimson=38)
    if color not in COLOR_MAPPINGS:
        raise ValueError("invalid color '%s'" % color)
    return f'\x1b[{COLOR_MAPPINGS[color]}m{text}\x1b[0m'


def configure_logging(text_color='green'):
    import warnings
    warnings.filterwarnings("ignore")

    text = colorize("%(message)s", text_color)
    logging.basicConfig(level=logging.INFO, format=f"[%(asctime)s] " + text)


def tf_var_size(v):
    return np.prod(v.get_shape().as_list())


def tf_flat_vars(var_list):
    return tf.concat(axis=0, values=[tf.reshape(v, [tf_var_size(v)]) for v in var_list])


def dense_nn(inputs, layers_sizes, name="fc", reuse=False, output_fn=None,
             dropout_keep_prob=None, batch_norm=False, training=True):
    logging.info(f"Building mlp {name} | sizes: {[inputs.shape[0]] + layers_sizes}")
    with tf.variable_scope(name, reuse=reuse):
        out = inputs
        for i, size in enumerate(layers_sizes):
            logging.info(f"layer:{name}_l{i}, size:{size}")
            if i > 0 and dropout_keep_prob is not None and training:
                # No dropout on the input layer.
                out = tf.nn.dropout(out, keep_prob=dropout_keep_prob)

            out = tf.layers.dense(
                out,
                size,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.0),
                name=name + '_l' + str(i),
                reuse=reuse,
                # Add relu activation only for internal layers.
                activation=tf.nn.relu if i < len(layers_sizes) - 1 else None,
            )

            if batch_norm:
                out = tf.layers.batch_normalization(out, training=training)

        if output_fn:
            out = output_fn(out)

    return out


def get_available_gpus():
    gpus = []
    try:
        gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
        gpus = gpus.decode().strip().split('\n')
    except FileNotFoundError:
        pass
    logging.info(f"Found available gpus: {gpus}")
    return gpus


def make_session(num_cpus=None, use_gpu=True):
    """
    Returns a session with `num_cpus` CPUs + all the available GPUs if use_gpu = true.
    """
    if num_cpus is None:
        num_cpus = int(multiprocessing.cpu_count())

    gpu_options = None
    if use_gpu:
        num_gpus = len(get_available_gpus())
        if num_gpus > 0:
            visible_device_list = ','.join(list(map(str, range(num_gpus))))
            gpu_options = tf.GPUOptions(visible_device_list=visible_device_list)

    sess_config = dict(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpus,
        intra_op_parallelism_threads=num_cpus,
    )
    logging.info(f"tf session config: {sess_config}")
    tf_config = tf.ConfigProto(**sess_config)
    return tf.Session(config=tf_config)


def prepare_mnist_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    shuffle_buffer = 1000
    prefetch_buffer = 1000
    batch_size = 128
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(shuffle_buffer).prefetch(prefetch_buffer).batch(batch_size)
    return dataset, (x_train, y_train), (x_test, y_test)


def create_mnist_model(input_ph, label_ph, layer_sizes=[64], name='mnist_dense_nn'):
    assert list(input_ph.shape)[1:] == [28, 28]

    with tf.variable_scope(name, reuse=False):
        label_ohe = tf.one_hot(label_ph, 10, dtype=tf.float64)

        x = tf.reshape(input_ph, (-1, 28 * 28))
        # Toy model: 28x28 --> multiple fc layers --> ReLU --> fc 10 --> output logits
        logits = dense_nn(x, layer_sizes + [10], name="mlp", reuse=False, dropout_keep_prob=0.1)
        preds = tf.nn.softmax(logits)

        loss = tf.losses.softmax_cross_entropy(label_ohe, logits)
        accuracy = tf.reduce_sum(label_ohe * preds) / tf.cast(tf.shape(label_ohe)[0], tf.float64)

    return loss, accuracy
