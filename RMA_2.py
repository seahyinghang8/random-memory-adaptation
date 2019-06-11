# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 23:50:14 2019

@author: LimJi
"""

import numpy as np
import torch


class RandomMemory:
    def __init__(self, capacity, input_result, output_dimension):
        self.capacity = capacity
        self.states = np.zeros((capacity, input_result))
        self.values = np.zeros((capacity))
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


class OneLayerNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(OneLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y_pred = self.linear1(x)
        return y_pred

class RMA(object):
    def __init__(self, args, dtype):
        self.dtype = dtype
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.memory_each = args.memory_each
        self.model = OneLayerNet(784, 10)
        self.model.type(self.dtype)
        self.M = RandomMemory(args.memory_size, 784, 10)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    
    def get_memory_sample(self, batch_size):
        # Return the embeddings and values sample from memory
        x, y_ = self.M.nn(batch_size)
        return x, y_

    def add_to_memory(self, xs, ys):
        # Adds the given embedding to the memory
        self.M.add(xs, ys)

    def train(self, x_train, y_train, batch_size):
        
        m, _ = x_train.shape
        
        y_pred = self.model(x_train.float())
        tloss = self.loss_fn(y_pred, y_train.long())
        print(tloss)
        dloss = tloss
        while dloss > 1e-4:
            old_loss = tloss
            for i in range(int(m/batch_size)):
                xs = x_train[i*batch_size : (i+1)*batch_size]
                ys = y_train[i*batch_size : (i+1)*batch_size]
                y_pred = self.model(xs.float())
                loss = self.loss_fn(y_pred, ys.long())
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if i % self.memory_each == 0:
                    self.add_to_memory(xs.cpu().numpy(), ys.cpu().numpy())
            y_pred = self.model(x_train.float())
            tloss = self.loss_fn(y_pred, y_train.long())
            dloss = old_loss - tloss
            print(tloss)
        print(tloss)

    def test(self, xs_test, ys_test):
        # assume all xs are from the same task
        ids = np.random.choice(xs_test.shape[0], size=100, replace=False)
        memx = []
        memy = []
        for idx in ids:
            x, y = self.M.nearest_k(xs_test[idx].cpu().numpy(), 10)
            memx.append(x)
            memy.append(y)
        memx = torch.from_numpy(np.concatenate(memx)).type(self.dtype)
        memy = torch.from_numpy(np.concatenate(memy)).type(self.dtype)

        self.train(memx, memy, self.batch_size)
        y_pred = self.model(xs_test.float())
        pred_label = torch.max(y_pred, 1)[1]
        true_label = ys_test
        acc = torch.mean(torch.eq(true_label.long(), pred_label).float())
        
        return acc