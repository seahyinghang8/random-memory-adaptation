import numpy as np
import tensorflow as tf


class RandomMemory:
    def __init__(self, capacity, input_result, output_dimension):
        self.capacity = capacity
        self.states = np.zeros((capacity, input_result))
        self.values = np.zeros((capacity, output_dimension))
        # Pointer
        self.pointer = 0

    def nn(self, n_samples):
        # We seek and return n random memory locations
        idx = np.random.choice(np.arange(len(self.states)), n_samples, replace=False)
        embs = self.states[idx]
        values = self.values[idx]

        return embs, values

    def add(self, keys, values):
        # We add {k, v} pairs to the memory
        for i, _ in enumerate(keys):
            if self.pointer >= self.capacity:
                self.pointer = 0
            self.states[self.pointer] = keys[i]
            self.values[self.pointer] = values[i]
            self.pointer += 1

    def nearest_k(self, key, k=5):
        """
        get k nearest memory locations
        """
        unnormalized = np.matmul(self.states, key.T)
        cos_sim = unnormalized / (np.linalg.norm(self.states, axis=1) * np.linalg.norm(key))
        k_idx = np.argsort(cos_sim)[:k]
        return (self.states[k_idx], self.values[k_idx])


class RMA(object):
    def __init__(self, session, args):
        self.learning_rate = args.learning_rate
        self.session = session

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])


        self.w = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        # Memory Sampling
        self.memory_x_ = tf.placeholder(tf.float32, shape=[None, 784])
        self.memory_y_ = tf.placeholder(tf.float32, shape=[None, 10])

        # Network
        self.y = self.network(self.x, self.w, self.b)
        
        self.saver = tf.train.Saver()

        # Memory M
        self.M = RandomMemory(args.memory_size, self.x.get_shape()[-1], self.y.get_shape()[-1])

        # Loss function
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
#        w_update, b_update = tf.gradients(
#            tf.nn.softmax_cross_entropy_with_logits(labels=self.memory_y_, logits=self.network(self.memory_x_, self.w, self.b)),
#            [self.w, self.b]
#        )
#        test_preds = self.network(self.x, self.w - 0.01*w_update, self.b - 0.01*b_update)
        test_preds = self.network(self.x, self.w, self.b)
        self.correct_prediction = tf.equal(tf.argmax(test_preds, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Initialize the variables
        session.run(tf.global_variables_initializer())

    def get_memory_sample(self, batch_size):
        # Return the embeddings and values sample from memory
        x, y_ = self.M.nn(batch_size)
        return x, y_

    def add_to_memory(self, xs, ys):
        # Adds the given embedding to the memory
        self.M.add(xs, ys)

    def train(self, xs, ys, memory_sample_batch):
        self.session.run(self.optim, feed_dict={self.x: xs, self.y_: ys})

    def test(self, xs_test, ys_test):
        # assume all xs are from the same task
        accs = []
        ids = np.random.choice(xs_test.shape[0], size=100, replace=False)
        memx = []
        memy = []
        for idx in ids:
            x, y = self.M.nearest_k(xs_test[idx], 10)
            memx.append(x)
            memy.append(y)
        memx = np.concatenate(memx)
        memy = np.concatenate(memy)
        
        self.saver.save(self.session, "./model.ckpt")
        self.session.run(self.optim, feed_dict={self.x: memx, self.y_: memy})

        acc = self.session.run(
                self.accuracy,
                feed_dict={
                    self.x: xs_test,
                    self.y_: ys_test,
                    self.memory_x_: memx, self.memory_y_: memy,
                })
        
        self.saver.restore(self.session, "./model.ckpt")
        return acc

    @staticmethod
    def network(x, w, b):
        # Basic 2 layers MLP
        y = tf.matmul(x, w) + b
        return y
