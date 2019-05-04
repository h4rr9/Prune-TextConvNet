import tensorflow as tf


def conv1d_relu(inputs, filters, k_size, stride, padding, scope_name='conv'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        in_channels = inputs.shape[-1]

        kernel = tf.get_variable('kernal',
                                 [k_size, in_channels, filters],
                                 initializer=tf.truncated_normal_initializer())

        biases = tf.get_variable('biases',
                                 [filters],
                                 initializer=tf.random_normal_initializer())

        conv = tf.nn.conv1d(inputs,
                            kernel,
                            stride=stride,
                            padding=padding)

        output = tf.nn.relu(conv + biases, name=scope.name)

    return output


def one_maxpool(inputs, stride, padding='VALID', scope_name='pool'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        _inputs = tf.expand_dims(inputs, axis=1)
        height, width = _inputs.shape[-2:]

        _pool = tf.nn.max_pool(_inputs,
                               ksize=[1, 1, height, 1],
                               strides=[1, 1, stride, 1],
                               padding=padding,
                               name=scope.name)
        pool = tf.squeeze(_pool, axis=1)

    return pool


def flatten(inputs, scope_name='flatten'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        feature_dim = inputs.shape[1] * inputs.shape[2]

        flatten = tf.reshape(inputs, shape=[-1, feature_dim], name=scope.name)

    return flatten


def concatinate(inputs, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        concat = tf.concat(inputs, 1, name=scope.name)

    return concat


def fully_connected(inputs, out_dim, scope_name='fc'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights',
                            [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())

        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))

        out = tf.add(tf.matmul(inputs, w), b, name=scope.name)
    return out


def Dropout(inputs, rate, scope_name='dropout'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        dropout = tf.nn.dropout(inputs, keep_prob=1 - rate, name=scope.name)
    return dropout


def l2_norm(inputs, alpha, scope_name='l2_norm'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        norm = alpha * tf.divide(inputs,
                                 tf.norm(inputs, ord='euclidean'),
                                 name=scope.name)
    return norm
