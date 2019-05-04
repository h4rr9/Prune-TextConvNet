import tensorflow as tf
import time
import utils
import os
import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PATH_CHECKPOINTS = './checkpointsc'
# PATH_CHECKPOINTS = '/scratch/scratch5/harig/tcNet_sub/checkpoints'
PATH_GRAPHS = './graphs'
# PATH_GRAPHS = '/scratch/scratch5/harig/tcNet_sub/graphs'


class TextConvNet:

    def __init__(self):
        self.batch_size = 50
        self.learning_rate = 1.0
        self.l2_constraint = 3
        self.gstep = tf.get_variable('global_step',
                                     initializer=tf.constant_initializer(0),
                                     dtype=tf.int32,
                                     trainable=False,
                                     shape=[])
        self.n_classes = 2
        self.skip_step = 20
        self.training = False
        self.keep_prob = tf.constant(0.5)

    def import_data(self):
        train, val = utils.load_subjectivity_data()

        self.max_sentence_size = train[0].shape[1]
        self.n_train = train[0].shape[0]
        self.n_test = val[0].shape[0]

        train_data = tf.data.Dataset.from_tensor_slices(train)
        train_data = train_data.shuffle(self.n_train)
        train_data = train_data.batch(self.batch_size)

        val_data = tf.data.Dataset.from_tensor_slices(val)
        val_data = val_data.batch(self.batch_size)

        iterator = tf.data.Iterator.from_structure(
            train_data.output_types, train_data.output_shapes)

        self.sentence, self.label = iterator.get_next()

        self.train_init = iterator.make_initializer(train_data)
        self.val_init = iterator.make_initializer(val_data)

    def get_embedding(self):
        with tf.name_scope('embed'):
            embedding_matrix = utils.load_embeddings_subjectivity()
            _embed = tf.constant(embedding_matrix)
            embed_matrix = tf.get_variable(
                'embed_matrix', initializer=_embed)

            self.embed = tf.nn.embedding_lookup(
                embed_matrix, self.sentence, name='embed')

            # self.embed = tf.reshape(
            #     embed, shape=[-1, self.max_sentence_size, 300, 1])

    def model(self):
        conv0 = layers.conv1d_relu(inputs=self.embed,
                                   filters=100,
                                   k_size=3,
                                   stride=1,
                                   padding='VALID',
                                   scope_name='conv0')
        pool0 = layers.one_maxpool(inputs=conv0, stride=1,
                                   padding='VALID', scope_name='pool0')
        flatten0 = layers.flatten(inputs=pool0, scope_name='flatten0')

        conv1 = layers.conv1d_relu(inputs=self.embed,
                                   filters=100,
                                   k_size=4,
                                   stride=1,
                                   padding='VALID',
                                   scope_name='conv1')
        pool1 = layers.one_maxpool(inputs=conv1, stride=1,
                                   padding='VALID', scope_name='pool1')
        flatten1 = layers.flatten(inputs=pool1, scope_name='flatten1')

        conv2 = layers.conv1d_relu(inputs=self.embed,
                                   filters=100,
                                   k_size=5,
                                   stride=1,
                                   padding='VALID',
                                   scope_name='conv2')
        pool2 = layers.one_maxpool(inputs=conv2, stride=1,
                                   padding='VALID', scope_name='pool2')
        flatten2 = layers.flatten(inputs=pool2, scope_name='flatten2')

        concat0 = layers.concatinate(
            inputs=[flatten0, flatten1, flatten2], scope_name='concat0')

        norm0 = layers.l2_norm(
            concat0, alpha=self.l2_constraint, scope_name='norm0')

        dropout0 = layers.Dropout(
            inputs=norm0, rate=1 - self.keep_prob, scope_name='dropout0')

        self.logits_train = layers.fully_connected(
            inputs=dropout0, out_dim=self.n_classes, scope_name='fc0')

        self.logits_test = layers.fully_connected(
            inputs=concat0, out_dim=self.n_classes, scope_name='fc0')

    def loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.label, logits=self.logits_train)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        with tf.name_scope('optimize'):
            _opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
            self.opt = _opt.minimize(self.loss, global_step=self.gstep)

    def summaries(self):
        with tf.name_scope('train_summaries'):
            train_loss = tf.summary.scalar('train_loss', self.loss)
            train_accuracy = tf.summary.scalar(
                'train_accuracy', self.accuracy / self.batch_size)
            hist_train_loss = tf.summary.histogram(
                'histogram_train_loss', self.loss)
            self.train_summary_op = tf.summary.merge(
                [train_loss, train_accuracy, hist_train_loss])

        with tf.name_scope('val_summaries'):
            val_loss = tf.summary.scalar('val_loss', self.loss)
            val_summary = tf.summary.scalar(
                'val_accuracy', self.accuracy / self.batch_size)
            hist_val_loss = tf.summary.histogram(
                'histogram_val_loss', self.loss)
            self.val_summary_op = tf.summary.merge(
                [val_loss, val_summary, hist_val_loss])

    def eval(self):
        with tf.name_scope('eval'):
            preds = tf.nn.softmax(self.logits_test)
            correct_preds = tf.equal(
                tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):

        self.import_data()
        self.get_embedding()
        self.model()
        self.loss()
        self.optimize()
        self.eval()
        self.summaries()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        total_correct_preds = 0

        try:
            while True:
                _, l, accuracy_batch, summaries = sess.run(
                    [self.opt,
                     self.loss,
                     self.accuracy,
                     self.train_summary_op])
                writer.add_summary(summaries, global_step=step)

                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step = step + 1
                total_correct_preds = total_correct_preds + accuracy_batch
                total_loss = total_loss + l
                n_batches = n_batches + 1
        except tf.errors.OutOfRangeError:
            pass

        saver.save(sess, PATH_CHECKPOINTS, step)

        print('\nAverage training loss at epoch {0}: {1}'.format(
            epoch, total_loss / n_batches))
        print('Training accuracy at epoch {0}: {1}'.format(
            epoch, total_correct_preds / self.n_train))
        print('Took: {0} seconds'.format(time.time() - start_time))

        return step

    def eval_once(self, sess, init, writer, epoch, val_step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        total_loss = 0
        n_batches = 0
        try:
            while True:
                l, accuracy_batch, summaries = sess.run(
                    [self.loss, self.accuracy, self.val_summary_op])
                writer.add_summary(summaries, global_step=val_step)
                total_correct_preds = total_correct_preds + accuracy_batch
                total_loss = total_loss + l
                n_batches = n_batches + 1
                val_step = val_step + 1
        except tf.errors.OutOfRangeError:
            pass

        print('Average validation loss at epoch {0}: {1}'.format(
            epoch, total_loss / n_batches))
        print('Validation accuracy at epoch {0}: {1}'.format(
            epoch, total_correct_preds / self.n_test))
        print('Took: {0} seconds\n'.format(time.time() - start_time))

        return val_step

    def train(self, n_epochs):
        utils.mkdir_safe(os.path.dirname(PATH_CHECKPOINTS))
        utils.mkdir_safe(PATH_CHECKPOINTS)

        train_writer = tf.summary.FileWriter(
            PATH_GRAPHS + '/train', tf.get_default_graph())

        val_writer = tf.summary.FileWriter(
            PATH_GRAPHS + '/val')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # ckpt = tf.train.get_checkpoint_state(PATH_CHECKPOINTS))

            # if ckpt and ckpt.model_checkpoint_path:
            #     saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()
            val_step = 0

            for epoch in range(n_epochs):
                step = self.train_one_epoch(
                    sess, saver, self.train_init, train_writer, epoch, step)
                val_step = self.eval_once(
                    sess, self.val_init, val_writer, epoch, val_step)

        train_writer.close()
        val_writer.close()


if __name__ == '__main__':
    model = TextConvNet()
    model.build()
    model.train(n_epochs=100)
