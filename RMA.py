from sklearn.linear_model import LogisticRegression
import numpy as np
import tensorflow as tf


class RandomMemory:
    def __init__(self, capacity, input_result, output_dimension):
        self.capacity = capacity
        self.input_result = input_result
        self.output_dimension = output_dimension
#        self.states = np.zeros((capacity, input_result))
#        self.values = np.zeros((capacity, output_dimension))
        self.states = {}
        self.values = {}
        # Pointer
        self.pointer = {}

#    def nn(self, n_samples):
#        # We seek and return n random memory locations
#        idx = np.random.choice(np.arange(len(self.states)), n_samples, replace=False)
#        embs = self.states[idx]
#        values = self.values[idx]
#
#        return embs, values

    def add(self, keys, values, taskid):
        if taskid not in self.pointer:
            self.pointer[taskid] = 0
            self.states[taskid] = np.zeros((self.capacity, self.input_result))
            self.values[taskid] = np.zeros((self.capacity, self.output_dimension))
        pointer = self.pointer[taskid]
        # We add {k, v} pairs to the memory
        for i, _ in enumerate(keys):
            if pointer >= self.capacity:
                pointer = 0
            self.states[taskid][pointer] = keys[i]
            self.values[taskid][pointer] = values[i]
            pointer += 1

    def get(self, n_samples, taskid):
        idx = np.random.choice(np.arange(len(self.states[taskid])), n_samples, replace=False)
        embs = self.states[taskid][idx]
        values = self.values[taskid][idx]

        return embs, values


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

        # Memory M
        self.M = RandomMemory(args.memory_size, self.x.get_shape()[-1], self.y.get_shape()[-1])

        # Loss function
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
        self.updates = tf.gradients(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.memory_y_, logits=self.network(self.memory_x_, self.w, self.b)),
            [self.w, self.b]
        )
        w_update, b_update = self.updates
        test_preds = self.network(self.x, self.w - 0.01*w_update, self.b - 0.01*b_update)
        self.correct_prediction = tf.equal(tf.argmax(test_preds, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Initialize the variables
        session.run(tf.global_variables_initializer())
        
        # Logistic Preclassifier
        self.clf = LogisticRegression(random_state=0, solver='lbfgs',
                                      multi_class='multinomial')
        

    def get_memory_sample(self, batch_size):
        # Return the embeddings and values sample from memory
        x, y_ = self.M.nn(batch_size)
        return x, y_

    def add_to_memory(self, xs, ys, taskid):
        # Adds the given embedding to the memory
        self.M.add(xs, ys, taskid)

    def consolidate_memory(self):
        xs = []
        ys = []
        if len(self.M.pointer) > 1:
            for taskid in self.M.pointer:
                xs.append(self.M.states[taskid])
                ys.append([taskid]*self.M.capacity)
            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            self.clf.fit(xs, ys)

    def train(self, xs, ys, taskid, memory_sample_batch):
        self.session.run(self.optim, feed_dict={self.x: xs, self.y_: ys})

    def test(self, xs_test, ys_test):
        # assume all xs are from the same task
        if len(self.M.pointer) > 1:
            preds = self.clf.predict(xs_test)
            ids, cnts = np.unique(preds, return_counts=True)
            pred_taskid = ids[np.argmax(cnts)]
            memx, memy = self.M.get(100, pred_taskid)
        else:
            memx = np.zeros((0, self.M.input_result))
            memy = np.zeros((0, self.M.output_dimension))
            
        orig_w = self.w
        orig_b = self.b
        print(orig_w)

        self.session.run(self.optim, feed_dict={self.x: memx, self.y_: memy})

        acc = self.session.run(
                self.accuracy,
                feed_dict={
                    self.x: xs_test,
                    self.y_: ys_test,
                    self.memory_x_: memx, self.memory_y_: memy,
                })
        
        print(orig_w)
        self.w = orig_w
        self.b = orig_b
        return acc

    @staticmethod
    def network(x, w, b):
        # Basic 2 layers MLP
        y = tf.matmul(x, w) + b
        return y
