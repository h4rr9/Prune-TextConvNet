import tensorflow as tf
import tcNet as T
import numpy as np
import time
import os
import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4


class Pruner(T.TextConvNet):

    def __init__(self):
        T.TextConvNet.__init__(self)
        self.prune_keep = 0.25

    def destroy(self):
        tf.reset_default_graph()

    def load_np_weights(self, sess):

        weights = {}
        for var in tf.trainable_variables():
            weights[var.name] = var.eval(session=sess)

        return weights

    def load_weights(self, sess):
        loader = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(T.PATH_BEST_WEIGHTS)

        loader.restore(sess, ckpt.model_checkpoint_path)

    def prune_initializer(self):
        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())

            self.load_weights(sess)

            np_weights = self.load_np_weights(sess)

            print('\nBefore pruning conv0 \n')

            self.prune_eval(sess)

        return np_weights

    def fine_tune(self, sess, step=0, epoch=0):
        start_time = time.time()
        sess.run(self.train_init)
        self.training = True
        total_loss = 0
        n_batches = 0
        total_correct_preds = 0

        try:
            while True:
                _, l, accuracy_batch = sess.run(
                    [self.opt,
                     self.loss,
                     self.accuracy,
                     ], feed_dict={self.keep_prob: self.train_prob})

                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step = step + 1
                total_correct_preds = total_correct_preds + accuracy_batch
                total_loss = total_loss + l
                n_batches = n_batches + 1
        except tf.errors.OutOfRangeError:
            pass

        print('\nAverage training loss at epoch {0} : {1}'.format(
            epoch, total_loss / n_batches))
        print('Training accuracy at epoch {0} : {1}'.format(
            epoch, total_correct_preds / self.n_train))
        print('Took: {0} seconds\n'.format(time.time() - start_time))

        return step

    def prune_eval(self, sess, epoch=0, best_acc=0):
        start_time = time.time()
        sess.run(self.val_init)
        self.training = False
        total_correct_preds = 0
        total_loss = 0
        n_batches = 0
        try:
            while True:
                l, accuracy_batch = sess.run(
                    [self.loss, self.accuracy], feed_dict={self.keep_prob: self.test_prob})

                total_correct_preds = total_correct_preds + accuracy_batch
                total_loss = total_loss + l
                n_batches = n_batches + 1
        except tf.errors.OutOfRangeError:
            pass

        if best_acc < total_correct_preds / self.n_test:
            best_acc = total_correct_preds / self.n_test

        print('\nBest validation accuracy : {0}'.format(best_acc))
        print('Average validation loss at epoch {0} : {1}'.format(
            epoch, total_loss / n_batches))
        print('Validation accuracy at epoch {0} : {1}'.format(
            epoch, total_correct_preds / self.n_test))
        print('Took: {0} seconds\n'.format(time.time() - start_time))

        return best_acc

    def prune_layer(self, weights, layer):

        kernel_key, bias_key = self.get_weights_keys_layer(weights, layer)

        kernel, bias = weights[kernel_key], weights[bias_key]

        top_m = int(self.prune_keep * kernel.shape[-1])

        req_indices = self.l2_filter(kernel, top_m)

        weights[kernel_key] = kernel[:, :, req_indices]
        weights[bias_key] = bias[req_indices]

        return weights

    def l2_filter(self, weight, top_m):

        assert len(weight.shape) == 3, "not covolution kernel"

        l2_norms = np.linalg.norm(weight, axis=(0, 1))

        req_indices = np.argsort(l2_norms)[::-1][:top_m]

        return req_indices

    def get_weights_keys_layer(self, weights, layer):

        weights_key, biases_key = [
            key for key in weights.keys() if layer in key]

        if 'biases' in weights_key:
            weights_key, biases_key = biases_key, weights_key

        return weights_key, biases_key

    def get_embedding_key(self, weights):

        [embedding_key] = [key for key in weights.keys() if 'embed' in key]

        return embedding_key

    def model_from_weights(self, weights):

        conv0_keys = self.get_weights_keys_layer(weights, 'conv0')
        conv1_keys = self.get_weights_keys_layer(weights, 'conv1')
        conv2_keys = self.get_weights_keys_layer(weights, 'conv2')
        fc0_keys = self.get_weights_keys_layer(weights, 'fc0')

        conv0_weights = (weights[conv0_keys[0]], weights[conv0_keys[1]])
        conv1_weights = (weights[conv1_keys[0]], weights[conv1_keys[1]])
        conv2_weights = (weights[conv2_keys[0]], weights[conv2_keys[1]])
        fc0_weights = (weights[fc0_keys[0]], weights[fc0_keys[1]])

        conv0 = layers.conv1d_relu(inputs=self.embed,
                                   filters=100,
                                   k_size=3,
                                   stride=1,
                                   padding='SAME',
                                   scope_name='conv0',
                                   _weights=conv0_weights)
        pool0 = layers.one_maxpool(inputs=conv0,
                                   padding='VALID', scope_name='pool0')

        flatten0 = layers.flatten(pool0, scope_name='flatten0')

        conv1 = layers.conv1d_relu(inputs=self.embed,
                                   filters=100,
                                   k_size=4,
                                   stride=1,
                                   padding='SAME',
                                   scope_name='conv1',
                                   _weights=conv1_weights)
        pool1 = layers.one_maxpool(inputs=conv1,
                                   padding='VALID', scope_name='pool1')

        flatten1 = layers.flatten(pool1, scope_name='flatten1')

        conv2 = layers.conv1d_relu(inputs=self.embed,
                                   filters=100,
                                   k_size=5,
                                   stride=1,
                                   padding='SAME',
                                   scope_name='conv2',
                                   _weights=conv2_weights)
        pool2 = layers.one_maxpool(inputs=conv2,
                                   padding='VALID', scope_name='pool2')

        flatten2 = layers.flatten(inputs=pool2, scope_name='flatten2')

        concat0 = layers.concatinate(
            inputs=[flatten0, flatten1, flatten2], scope_name='concat0')

        dropout0 = layers.Dropout(
            inputs=concat0, rate=1 - self.keep_prob, scope_name='dropout0')

        self.logits = layers.fully_connected(
            inputs=dropout0, out_dim=self.n_classes, scope_name='fc0')

    def build_from_weights(self, weights):

        embedding_key = self.get_embedding_key(weights)
        embedding_weights = weights[embedding_key]

        self.init()
        self.import_data()
        self.get_embedding(embedding_weights)
        self.model_from_weights(weights)
        self.init_loss()
        self.init_optimize()
        self.init_eval()

    def prune(self):

        self.weights = self.prune_initializer()

        self.destroy()

        self.weights = self.prune_layer(self.weights, 'conv0')

        self.build_from_weights(self.weights)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            print('\n\nBefore fine tuning\n\n')
            self.prune_eval(sess)

            print('Fine Tuning conv0')

            step = 0
            acc = 0
            for epoch in range(5):
                step = self.fine_tune(sess, step, epoch)
                acc = self.prune_eval(sess, epoch, acc)

            print('\n\nAfter Pruning conv0\n\n')

            self.prune_eval(sess)

            self.weights = self.load_np_weights(sess)

        self.destroy()

        self.weights = self.prune_layer(self.weights, 'conv1')

        self.build_from_weights(self.weights)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            print('\n\nBefore fine tuning\n\n')
            self.prune_eval(sess)

            print('Fine Tuning conv1')

            step = 0
            acc = 0
            for epoch in range(5):
                step = self.fine_tune(sess, step, epoch)
                acc = self.prune_eval(sess, epoch, acc)

            print('\n\nAfter Pruning conv1\n\n')

            self.prune_eval(sess)

            self.weights = self.load_np_weights(sess)

        self.destroy()

        self.weights = self.prune_layer(self.weights, 'conv2')

        self.build_from_weights(self.weights)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            print('\n\nBefore fine tuning\n\n')
            self.prune_eval(sess)

            print('Fine Tuning')

            step = 0
            for epoch in range(5):
                step = self.fine_tune(sess, step, epoch)
                acc = self.prune_eval(sess, epoch, acc)

            print('\n\nAfter Pruning conv2\n\n')

            self.prune_eval(sess)

            print('\nFinal fine tunning\n')
            step = 0
            acc = 0
            for epoch in range(50):
                step = self.fine_tune(sess, step, epoch)
                acc = self.prune_eval(sess, epoch, acc)

            self.weights = self.load_np_weights(sess)

    def write_graph(self):
        tf.summary.FileWriter('./pruned_graph', tf.get_default_graph())


if __name__ == '__main__':
    pruner = Pruner()
    pruner.build()
    pruner.prune()
