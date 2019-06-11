# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 00:13:04 2019

@author: LimJi
"""


import time
import argparse
import numpy as np
import torch
from tensorflow.examples.tutorials.mnist import input_data
from RMA_2 import RMA

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor

def main(args):
    print("\nParameters used:", args, "\n")

    # We create the baseline model
    baseline_model = RMA(args, dtype)
    # We create the memory adapted model
    rma_model = RMA(args, dtype)

    # Permuted MNIST
    # Generate the tasks specifications as a list of random permutations of the input pixels.
    mnist = input_data.read_data_sets("/tmp/")
    task_permutation = []
    for task in range(args.num_tasks_to_run):
        task_permutation.append(np.random.permutation(784))

    # print("\nBaseline MLP training...")
    # start = time.time()
    # last_performance_baseline = training(baseline_model, mnist, task_permutation, False)
    # end = time.time()
    # time_needed_baseline = round(end - start)
    # print("Training time elapsed: ", time_needed_baseline, "s")

    print("\nMemory adapted (RMA) training...")
    start = time.time()
    last_performance_ma = training(rma_model, mnist, task_permutation, True)
    end = time.time()
    time_needed_ma = round(end - start)
    print("Training time elapsed: ", time_needed_ma, "s")

    # Stats
    print("\nDifference in time between using or not memory: ", time_needed_ma - time_needed_baseline, "s")
    print("Test accuracy mean gained due to the memory: ",
          np.round(np.mean(last_performance_ma) - np.mean(last_performance_baseline), 2))

    # Plot the results
    plot_results(args.num_tasks_to_run, last_performance_baseline, last_performance_ma)


def training(model, mnist, task_permutation, use_memory=True):
    # Training the model using or not memory adaptation
    last_performance = []
    for task in range(args.num_tasks_to_run):
        print("\nTraining task: ", task + 1, "/", args.num_tasks_to_run)

        x_train = mnist.train.images
        y_train = mnist.train.labels
        
        model.train(torch.from_numpy(x_train).type(dtype), torch.from_numpy(y_train).type(dtype), args.batch_size)

        # Print test set accuracy to each task encountered so far
        for test_task in range(task + 1):
            test_images = mnist.test.images

            # Permute batch elements
            test_images = test_images[:, task_permutation[test_task]]

            acc = model.test(torch.from_numpy(test_images).type(dtype), torch.from_numpy(mnist.test.labels).type(dtype))
            acc = acc * 100

            if args.num_tasks_to_run == task + 1:
                last_performance.append(acc)

            print("Testing, Task: ", test_task + 1, " \tAccuracy: ", acc)

    return last_performance
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_tasks_to_run', type=int, default=20,
                        help='Number of task to run')
    parser.add_argument('--memory_size', type=int, default=3000,
                        help='Memory size')
    parser.add_argument('--memory_each', type=int, default=1,
                        help='Add to memory after these number of steps')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of batch for updates')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Learning rate')

    args = parser.parse_args()

    main(args)
