import time, os, json
import numpy as np
import matplotlib.pyplot as plt

import layers
import solver
import softmax
import cnn


def main():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # data preprocessing for neural network with fully-connected layers
    data = {
        'X_train': np.array(x_train[:55000], np.float32).reshape((55000, -1)),  # training data
        'y_train': np.array(y_train[:55000], np.int32),  # training labels
        'X_val': np.array(x_train[55000:], np.float32).reshape((5000, -1)),  # validation data
        'y_val': np.array(y_train[55000:], np.int32),  # validation labels
    }
    model = softmax.SoftmaxClassifier(hidden_dim=200)

    # data preprocessing for neural network with convolutional layers
    # data = {
    #    'X_train': np.array(x_train[:55000], np.float32).reshape((55000, 1, 28, 28)),  # training data
    #    'y_train': np.array(y_train[:55000], np.int32),  # training labels
    #    'X_val': np.array(x_train[55000:], np.float32).reshape((5000, 1, 28, 28)),  # validation data
    #    'y_val': np.array(y_train[55000:], np.int32),  # validation labels
    # }
    # model = cnn.ConvNet(hidden_dim=100,filter_size=5)
    # the update rule of 'adam' can be used to replace 'sgd' if it is helpful.
    solvers = solver.Solver(model, data,
                           update_rule='sgd',
                           optim_config={'learning_rate': 1e-3,},
                           lr_decay=0.95,
                           num_epochs=10, batch_size=100,
                           print_every=10)
    solvers.train()

    # Plot the training losses
    plt.plot(solvers.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()
    plt.savefig('loss.png')
    plt.close()

    test_acc = solvers.check_accuracy(X=np.array(x_test, np.float32).reshape((10000, -1)), y=y_test)
    # test_acc = solvers.check_accuracy(X=np.array(x_test, np.float32).reshape((10000, 1, 28, 28)), y=y_test)
    print('Test accuracy', test_acc)


if __name__== "__main__":
    main()

