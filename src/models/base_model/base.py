import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from sklearn.utils import shuffle
from tqdm import tqdm


class BaseModel:

    def __init__(self, input_dim, target_dim, layers, activation, lr, scope):
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.layers = layers
        self.activation = activation
        self.lr = lr
        self.sess = tf.Session()
        self.scope = scope
        self.build()

    def _add_placeholders_op(self):
        self.inputs_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.input_dim],
            name="inputs_placeholder"
        )
        self.targets_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.target_dim],
            name="targets_placeholder"
        )

    def _add_forward_pass(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            state = self.inputs_placeholder
            for layer in self.layers:
                state = tf.layers.dense(
                    state,
                    layer,
                    activation=self.activation
                )
            return tf.layers.dense(state, 1)

    def _add_loss_op(self):
        self.loss = tf.reduce_mean(
            tf.squared_difference(
                self.targets_placeholder,
                self.forward_pass
            )
        )

    def _add_optimizer_op(self):
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def build(self):
        self._add_placeholders_op()
        self.forward_pass = self._add_forward_pass()
        self._add_loss_op()
        self._add_optimizer_op()

    def train(self, inputs, targets, epochs, batch_size=64):
        self.sess.run(tf.global_variables_initializer())
        batch_loss = float('inf')
        prog = tqdm(range(epochs),
                    postfix={'Batch Loss': batch_loss})
        for epoch in prog:
            for i in range(0, len(inputs), batch_size):
                inputs_batch = inputs[i: i+batch_size]
                targets_batch = targets[i: i+batch_size]
                forward_pass_out, batch_loss, _ = self.sess.run(
                    [self.forward_pass, self.loss, self.train_op],
                    feed_dict={
                        self.inputs_placeholder: inputs_batch,
                        self.targets_placeholder: targets_batch
                    }
                )

            prog.set_postfix({'Batch Loss': batch_loss})

    def dump(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, model_dir):
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    @property
    def num_weights(self):
        variables = tf.trainable_variables(scope=self.scope)
        return sum([np.prod(v.shape.as_list()) for v in variables])

    @property
    def get_weights(self):
        return tf.trainable_variables(scope=self.scope)
