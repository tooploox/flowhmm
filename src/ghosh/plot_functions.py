# This file contains all the plot functions necessary 

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(network, x, i, savedir):
    """
    Plots the results of modelling the dataset for the given network after specific
    number of iterations

    """
    if os.path.isdir(savedir) == True:
        pass
    else:
        os.makedirs(savedir)

    plt.figure()    
    # noisy_moons = dataset(n_samples=1000, noise=.05)[0].astype(np.float32)
    x = x.permute(0,2,1).numpy()
    z = network.f(torch.from_numpy(x))[0].detach().numpy()
    plt.subplot(221)
    plt.scatter(z[:,0], z[:,1])
    plt.title(r'$z = f(X)$')

    z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000)
    plt.subplot(222)
    plt.scatter(z[:, 0], z[:, 1])
    plt.title(r'$z \sim p(z)$')

    plt.subplot(223)
    plt.scatter(x[:, 0], x[:, 1], c='r')
    plt.title(r'$X \sim p(X)$')

    plt.subplot(224)
    x_gen = network.sample(1000, network.in_channels).detach().numpy()
    plt.scatter(x_gen[:,0], x_gen[:,1], c='r')
    plt.title(r'$X = g(z)$')
    plt.savefig(savedir + "plot__bwrd_frwd__iter{}.pdf".format(i))
    plt.close()

def plot_grad(t, grad_mean, num_layers, savedir, savefile, layer_labels):
    """
    Plots a list of graphs indicating the mean variation of the gradient for different layers of the network
    """
    if os.path.isdir(savedir) == True:
        pass
    else:
        os.makedirs(savedir)

    plt.figure()
    ax = plt.subplot(111)
    for l in range(num_layers):
        ax.plot(list(range(t)), ([g[l] for g in grad_mean[:t]]),label=layer_labels[l])
    # Move the legend to the right of the plot.
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(savedir + savefile)
    plt.close()

def plot_act_value(t, pre_act_val_mean, num_layers, savedir, savefile, layer_labels):
    """
    Plots a list of graphs indicating the mean variation of the gradient for different layers of the network
    """
    if os.path.isdir(savedir) == True:
        pass
    else:
        os.makedirs(savedir)

    plt.figure()
    ax = plt.subplot(111)
    for l in range(num_layers):
        ax.plot(list(range(t)), ([g[l] for g in pre_act_val_mean[:t]]),label=layer_labels[l])
    # Move the legend to the right of the plot.
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(savedir + savefile)
    plt.close()


