import numpy as np
import tensorflow as tf


class RandomMemory:
    def __init__(self, capacity, input_result, output_dimension):
        self.capacity = capacity
        self.states = np.zeros((capacity, input_result))
        self.values = np.zeros((capacity, output_dimension))
        self.sample_weights = np.ones((capacity))
        # Pointer
        self.pointer = 0

    def nn(self, n_samples):
        # We seek and return n random memory locations
        idx = np.random.choice(
            np.arange(len(self.states)), n_samples,
            p=self.sample_weights / np.sum(self.sample_weights),
            replace=False)
        embs = self.states[idx]
        values = self.values[idx]
        self.sample_weights[idx] = 1
        mask = np.ones(len(self.states), np.bool)
        mask[idx] = 0
        self.sample_weights[mask] += 1        
        return embs, values

    # def add_old(self, keys, values):
    #     # We add {k, v} pairs to the memory
    #     for i, _ in enumerate(keys):
    #         if self.pointer >= self.capacity:
    #             self.pointer = 0
    #         self.states[self.pointer] = keys[i]
    #         self.values[self.pointer] = values[i]
    #         self.pointer += 1

    # def add(self, keys, values):
    #     for i, _ in enumerate(keys):
    #         if self.pointer >= self.capacity:
    #             unnormalized = np.matmul(self.states, keys[i].T)
    #             cos_sim = unnormalized / (np.linalg.norm(self.states, axis=1) * np.linalg.norm(keys[i]))
    #             insert_idx = np.argmin(cos_sim)
    #             self.states[insert_idx] = keys[i]
    #             self.values[insert_idx] = values[i]
    #         else:
    #             self.states[self.pointer] = keys[i]
    #             self.values[self.pointer] = values[i]
    #             self.pointer += 1

    def add(self, keys, values):
        for i, _ in enumerate(keys):
            if self.pointer >= self.capacity:
                unnormalized = np.matmul(self.states, keys[i].T)
                cos_sim = unnormalized / (np.linalg.norm(self.states, axis=1) * np.linalg.norm(keys[i]))
                sorted_ids = np.argsort(cos_sim)
                for insert_idx in sorted_ids:
                    if (self.values[insert_idx] != values[i]).any():
                        continue
                    else:
                        self.states[insert_idx] = keys[i]
                        self.values[insert_idx] = values[i]
                        self.sample_weights[insert_idx] = 1
                        break
            else:
                self.states[self.pointer] = keys[i]
                self.values[self.pointer] = values[i]
                self.pointer += 1

    # def add_random(self, keys, values):
    #     for i, _ in enumerate(keys):
    #         if self.pointer >= self.capacity:
    #             new_idx = np.random.randint(0, self.pointer)
    #             self.states[new_idx] = keys[i]
    #             self.states[new_idx] = values[i]
    #         else:
    #             self.states[self.pointer] = keys[i]
    #             self.values[self.pointer] = values[i]
    #             self.pointer += 1


class RMA(object):
    def __init__(self, session, args):
        self.learning_rate = args.learning_rate
        self.session = session

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.memory_sample_batch = tf.placeholder(tf.int16, shape=())

        # Memory Sampling
        embs_and_values = tf.py_func(self.get_memory_sample, [self.memory_sample_batch], [tf.float64, tf.float64])
        self.memory_batch_x = tf.to_float(embs_and_values[0])
        self.memory_batch_y = tf.to_float(embs_and_values[1])
        self.xa = tf.concat(values=[self.x, self.memory_batch_x], axis=0)
        self.ya_ = tf.concat(values=[self.y_, self.memory_batch_y], axis=0)

        # Network
        self.y = self.network(self.xa)

        # Memory M
        self.M = RandomMemory(args.memory_size, self.x.get_shape()[-1], self.y.get_shape()[-1])

        # Loss function
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ya_, logits=self.y))
        self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
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
        self.session.run(self.optim, feed_dict={self.x: xs, self.y_: ys, self.memory_sample_batch: memory_sample_batch})

    def test(self, xs_test, ys_test):
        acc = self.session.run(self.accuracy,
                               feed_dict={self.x: xs_test, self.y_: ys_test, self.memory_sample_batch: 0})
        return acc

    @staticmethod
    def network(x):
        # Basic 2 layers MLP
        w = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, w) + b
        return y
