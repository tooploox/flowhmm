# This Script is a PyTorch implementation of the GLOW Model described in 
# https://arxiv.org/abs/1807.03039

# Model based on: https://github.com/corenel/pytorch-glow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from sklearn import datasets
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from glow_modules import *
from ops import *
from plot_functions import plot_results
from torch.autograd import Function, Variable, detect_anomaly
from datetime import datetime

class Rescale(torch.nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, num_channels, 1))

    def forward(self, x):
        x = self.weight * x
        #x = torch.nn.utils.weight_norm(x)
        return x

# This class defines the attributes and functions necessary to simulate one step of flow
class FlowStep(nn.Module):
    flow_permutation_list = ['invconv', 'reverse', 'shuffle']
    flow_coupling_list = ['additive', 'affine']

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 actnorm_flag=True,
                 permutation='invconv',
                 coupling='additive',
                 actnorm_scale=1.,
                 lu_decomposition=False,
                 switch_flag=None,
                 p_drop=0,
                 weightnorm_flag=True):
        """
        One step of flow described in paper
                      ▲
                      │
        ┌─────────────┼─────────────┐
        │  ┌──────────┴──────────┐  │
        │  │ flow coupling layer │  │
        │  └──────────▲──────────┘  │
        │             │             │
        │  ┌──────────┴──────────┐  │
        │  │  flow permutation   │  │
        │  │        layer        │  │
        │  └──────────▲──────────┘  │
        │             │             │
        │  ┌──────────┴──────────┐  │
        │  │     activation      │  │
        │  │ normalization layer │  │
        │  └──────────▲──────────┘  │
        └─────────────┼─────────────┘
                      │
                      │
        :param in_channels: number of input channels
        :type in_channels: int
        :param hidden_channels: number of hidden channels
        :type hidden_channels: int
        :param permutation: type of flow permutation
        :type permutation: str
        :param coupling: type of flow coupling
        :type coupling: str
        :param actnorm_scale: scale factor of actnorm layer
        :type actnorm_scale: float
        :param lu_decomposition: whether to use LU decomposition or not
        :type lu_decomposition: bool
        """
        super().__init__()
        # permutation and coupling
        assert permutation in self.flow_permutation_list, 'Unsupported flow permutation: {}'.format(permutation)
        assert coupling in self.flow_coupling_list, 'Unsupported flow coupling: {}'.format(coupling)
        self.permutation = permutation
        self.coupling = coupling
       
        # activation normalization layer
        if actnorm_flag:
            self.actnorm = ActNorm(num_channels=in_channels, scale=actnorm_scale)

        # flow permutation layer
        if permutation == 'invconv':
            self.invconv = Invertible1x1Conv(num_channels=in_channels,
                                                    lu_decomposition=lu_decomposition)
        elif permutation == 'reverse':
            self.permrev = Permutation1d(num_channels=in_channels, shuffle=False, switch_flag=switch_flag)
        else:
            self.shuffle = Permutation1d(num_channels=in_channels, shuffle=True, switch_flag=switch_flag)

        # flow coupling layer
        if coupling == 'additive':
            self.NN = NN(in_channels // 2, hidden_channels, in_channels // 2, weightnorm_flag)
        else:
            self.NN = NN(in_channels // 2, hidden_channels, in_channels, weightnorm_flag)
            self.rescale = torch.nn.utils.weight_norm(Rescale(in_channels // 2))

    def normal_flow(self, x, logdet=None):
        """
        This function describes the complete one step of flow in the Forward (Normalizing)
        direction
        ---
        Args:
        - x: Input Tensor (that is usually the data vector) (Nb x Nc x Ns)
        - logdet: Variable that is going to store the logarithm of the determinant
        Returns:
        - z: Output tensor that is the mapped verison in the latent space
        - logdet : Logarithm of the determinant after traversing through the entire layer
        """
        # z = x.data.clone()
        # Activation normalization layer
        z, logdet = self.actnorm.forward(x, logdet=logdet)

        # Flow permutation layer
        if self.permutation == 'invconv':
            #z, logdet = self.invconv(z, logdet, reverse=False)
            z, logdet = self.invconv.forward(z, logdet)
        elif self.permutation == 'reverse':
            z = self.permrev.forward(z)
        else:
            z = self.shuffle.forward(z)

        # Flow coupling layer (Split the input into two parts and perform activations
        # on one part and identity mapping on the other part)
        z1, z2 = split_channel(z, 'simple')
        if self.coupling == 'additive':
            z2 += self.NN(z1)
        else:
            h = self.NN(z1)
            shift, scale = split_channel(h, 'cross')
            scale = torch.sigmoid(self.rescale(scale) + 2.)
            #scale = torch.sigmoid(scale + 2.)
            #scale = torch.exp(scale)
            z2 += shift
            z2 *= scale
            #logdet = reduce_sum(torch.log(scale), dim=[1, 2]) + logdet
            logdet = reduce_sum(torch.log(scale), dim=[1]) + logdet

        z = cat_channel(z1, z2, split_dim=1) # rejoin the two parts of the channel (along dim 1) and formulate the output tensor
        return z, logdet

    def reverse_flow(self, z, logdet=None):
        """
        This function describes the complete one step of flow in the Reverse (Generative)
        direction
        ---
        Args:
        - z: Input Tensor (that is usually the vector from the latent space) (Nb x Nc x Ns)
        - logdet: Variable that is going to store the logarithm of the determinant
        Returns:
        - x: Output tensor that is the mapped verison in the latent space
        - logdet : Logarithm of the determinant after traversing through the entire layer
        """
        # flow coupling layer (Split the input into two parts and perform activations
        # on one part and identity mapping on the other part)
        x1, x2 = split_channel(z, 'simple')
        if self.coupling == 'additive':
            x2 -= self.NN(x1)
        else:
            h = self.NN(x1)
            shift, scale = split_channel(h, 'cross')
            scale = torch.sigmoid(self.rescale(scale) + 2.)
            #scale = torch.sigmoid(scale + 2.)
            #scale = torch.exp(scale)
            x2 /= scale
            x2 -= shift
            #logdet = -reduce_sum(torch.log(scale), dim=[1, 2]) + logdet
            logdet = -reduce_sum(torch.log(scale), dim=[1]) + logdet
        
        x = cat_channel(x1, x2, split_dim=1) # Rejoins the output tensor along the dimension 1 

        # flow permutation layer 
        if self.permutation == 'invconv':
            x, logdet = self.invconv.reverse(x, logdet)
        elif self.permutation == 'reverse':
            x = self.permrev.reverse(x)
        else:
            x = self.shuffle.reverse(x)

        # activation normalization layer
        x, logdet = self.actnorm.reverse(x, logdet=logdet)

        return x, logdet 

    def forward(self, x, logdet=None):
        """
        Forward one step of flow
        :param x: input tensor
        :type x: torch.Tensor
        :param logdet: log determinant
        :type logdet: torch.Tensor
        
        :return: output and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        assert x.shape[1] % 2 == 0 # Check the channel dimensions are not zero
        return self.normal_flow(x, logdet) # Perform one step of flow in the normalizing direction
        
    def reverse(self, z, logdet=None):
        """
        Reverse one step of flow
        :param z: input tensor
        :type z: torch.Tensor
        :param logdet: log determinant
        :type logdet: torch.Tensor
        :return: output and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        assert z.shape[1] % 2 == 0 # Check the channel dimensions are not zero
        return self.reverse_flow(z, logdet) # Perform one step of flow in the generative direction

# The class defines the complete generative model along with the flow step
class FlowModel_GLOW(nn.Module):

    def __init__(self, in_channels, hidden_channels, K, L, prior, permutation='invconv',\
                 coupling='additive', actnorm_scale=1., lu_decomposition=False, \
                 sq_factor = 1, actnorm_flag = True, p_drop=0, weightnorm_flag=True):

        super().__init__()
        self.K = K # No. of Steps of flow (denoted by K)
        self.L = L # No. of layers in Multi-scale architecture (defined by L)
        self.prior = prior # Prior used for the flow model
        self.in_channels = in_channels # No. of input channels
        self.nets = nn.ModuleList() # Initialise the network with a modulelist of layers
        n_channels = self.in_channels # Defines the mini-batch size, number of channels, no. of samples in each mini-batch
        #self.output_shapes = [] # List to store the data about output shapes
        self.sq_factor = sq_factor

        for l in range(L):
            # Squeeze operation to increase the number of channels at the cost of reducing the samples
            self.nets.append(Squeeze1D(factor=self.sq_factor)) 
            n_channels = n_channels * self.sq_factor
            #self.output_shapes.append([-1, n_channels, n_samples])
            # Flow Step performed K times
            for k in range(K):
            
                if k % 2 == 0:
                    switch_flag = False
                else:
                    switch_flag = True
            
                self.nets.append(FlowStep(
                    in_channels=n_channels,
                    hidden_channels=hidden_channels,
                    actnorm_flag=actnorm_flag,
                    permutation=permutation,
                    coupling=coupling,
                    actnorm_scale=actnorm_scale,
                    lu_decomposition=lu_decomposition,
                    switch_flag=switch_flag,
                    p_drop=p_drop,
                    weightnorm_flag=weightnorm_flag))
                #self.output_shapes.append([-1, n_channels, n_samples])

            # Split operation along the channels
            if l < L - 1:
                self.nets.append(Split1D(num_channels=n_channels, factor=self.sq_factor))
                n_channels = n_channels // self.sq_factor
                #self.output_shapes.append([-1, n_channels, n_samples])

    def f(self, z, logdet=0.):
        """
        Defines the function that forms one complete flow through the total network
        in the normalizing direction
        """
        for layer in self.nets:
            z, logdet = layer.forward(z, logdet)
        return z, logdet

    def g(self, z, eps_std=None):
        """
        Defines the function that forms one complete flow through the toral network 
        in the generative direction
        """
        #for layer in self.nets:
        for layer in reversed(self.nets):
            if isinstance(layer, Split1D):
                z, _ = layer.reverse(z, logdet=0., eps_std=eps_std)
            else:
                z, _ = layer.reverse(z, logdet=0.)
        return z

    def log_prob(self, x, in_channels, x_mask=None):
        """
        This function is supposed to compute the log-likelihood using the logdet and the calculated prior
        """
        # Ensure that the input has proper dimensions (channel-wise)
        #if x.shape[1] != in_channels:
        #    x = x.permute(0,2,1)
        #else:
        #    pass
        # Previously: x is (batchsize, samples, channels/components) 
        # later after permutation, we want to make it (batchsize, channels, samples)
        x = x.permute(0, 2, 1) # Since channel wise processing occurs across dim=1
        z, logp = self.f(x)
        #return self.prior.log_prob(z) + logp
        #if z.shape[1] == in_channels:
        #    z = z.permute(0,2,1)
        # n_samples = count_pixels(z)
        z = z.permute(0, 2, 1) # To get the proper ordering before applying masks
        #px = self.prior.log_prob(z) + logp.view(logp.shape[0], 1)
        px = self.prior.log_prob(z) + logp
        #px = px / float(n_samples)
        if type(x_mask) == type(None): # x_mask decides if there are appended zeros for the samples which are not to be processed
            return px
        else:
            px[1 - x_mask] = 0
            return px
        
    def sample(self, batchSize, in_channels): 
        """
        This function is the push-forward function to sample a vector in the data space from the prior using
        the learned network
        """
        # Ensure that the input has proper dimensions (channel-wise)
        
        z = self.prior.sample((batchSize, 1))
        if z.shape[1] != in_channels:
            z = z.permute(0,2,1)
        else:
            pass
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x

def count_params(model):
    """
    Counts two types of parameters:
    - Total no. of parameters in the model (including trainable parameters)
    - Number of trainable parameters (i.e. parameters whose gradients will be computed)
    """
    total_num_params = sum(p.numel() for p in model.parameters())
    total_num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    return total_num_params, total_num_trainable_params

def NN(in_channels, hidden_channels, out_channels, weightnorm_flag):
    """
    Convolution block
    :param in_channels: number of input channels
    :type in_channels: int
    :param hidden_channels: number of hidden channels
    :type hidden_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    :return: desired convolution block
    :rtype: nn.Module
    """
    return nn.Sequential(
        nn.utils.weight_norm(Conv1d(in_channels, hidden_channels, do_weightnorm=weightnorm_flag), "weight"),
        nn.LeakyReLU(inplace=True),
        nn.utils.weight_norm(Conv1d(hidden_channels, hidden_channels, do_weightnorm=weightnorm_flag),"weight"),
        nn.LeakyReLU(inplace=True),
        #nn.Dropout(p=0.10),
        Conv1dZeros(hidden_channels, out_channels)
    )
    '''
    return nn.Sequential(
        Conv1d(in_channels, hidden_channels, do_weightnorm=weightnorm_flag),
        #nn.BatchNorm1d(hidden_channels),
        nn.LeakyReLU(inplace=True),
        Conv1d(hidden_channels, hidden_channels, do_weightnorm=weightnorm_flag),
        #nn.BatchNorm1d(hidden_channels),
        nn.LeakyReLU(inplace=True),
        #nn.Dropout(p=0.10),
        Conv1dZeros(hidden_channels, out_channels)
    )
    '''
    """
    return nn.Sequential(
        nn.Linear(in_channels, hidden_channels),
        nn.LeakyReLU(inplace=True),
        nn.Linear(hidden_channels, hidden_channels),
        nn.LeakyReLU(inplace=True),
        nn.Linear(hidden_channels, out_channels)
    )
    """
def main():

    # np.random.seed(2)
    # torch.manual_seed(2)
    # Define the input dataset and convert it in the format X : (batchsize x no_of_samples x no_of_channels)
    n_input_samples = 200
    X = datasets.make_moons(n_samples=n_input_samples, noise=.05)[0].astype(np.float32)
    X = torch.from_numpy(X)
    # X = np.random.randn(100, 2)
    # X = torch.FloatTensor(X)
    #if len(X.shape) == 2:
    #    X = X.unsqueeze(dim=1).permute(0, 2, 1) # Insert the dimensionality for the n_samples and permute dims to make Nb x Nc x Ns
    X = X.unsqueeze(dim=1)
    # Dimension of channel : input and hidden
    in_shape = X.size()
    n_IC = 2
    #if in_shape[1] != n_IC:
    #    X = X.permute(0,2,1)
    #else:
    #    pass
    n_HC = 128
    K = 8 # Depth of Flow
    L = 1 # No. of Layers in Multi-Scale architecture
    actnorm_flag = True
    p_drop = 0.0 # Dropout Rate
    weightnorm_flag = False
    prior = distributions.MultivariateNormal(torch.zeros(n_IC), torch.eye(n_IC))
    flow = FlowModel_GLOW(n_IC, n_HC, K, L, prior, permutation='invconv',\
                          coupling='affine', actnorm_scale=1., lu_decomposition=False,\
                          sq_factor=1, actnorm_flag=actnorm_flag, p_drop=p_drop, weightnorm_flag=weightnorm_flag)

    # Display the number of parameters to be trained
    total_num_params, total_num_trainable_params = count_params(flow)
    print("The total number of params: {} and the number of trainable params:{}".format(total_num_params, \
           total_num_trainable_params))
    # Define the optimizer
    optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=1e-4)
    
    N_iter = 4000
    savedir = "./NormFlowModel/GLOW_Model/figures_final_nsamples_LU_WN" + str(n_input_samples) + "/"
    # Training the model
    start_time = datetime.now()

    with detect_anomaly():

        for t in range(N_iter):

            # Optimizer Gradients set to zero initially
            optimizer.zero_grad()

            # Computes the loss in terms of NLL
            #regular_loss = torch.abs(torch.std(flow.f(X)[0]) - 1).mean()
            #loss = -flow.log_prob(X, n_IC).mean() + regular_loss
            loss = -flow.log_prob(X, n_IC).mean()
            # Backward pass to compute gradients using autograd
            loss.backward(retain_graph=True)

            #NOTE: Cheap fix for possible large gradients in LU_decomposition of invertible 1 x 1 convolutions 
            # with a factor of 4 just used heuristically.

            #nn.utils.clip_grad_norm_(flow.parameters(), 4)

            # Updates the learnale params
            optimizer.step()

            # Plotting the loss and iteratio details every 500 iterations
            if t % 500 == 0 and t != 0:
                print("Iteration {}, loss : {}, Elapsed Time:{}".format(t, loss, abs((start_time - datetime.now()).total_seconds())))
                #print("Iteration {}, regularization loss : {}, Elapsed Time:{}".format(t, regular_loss, abs((start_time - datetime.now()).total_seconds())))
                start_time = datetime.now()

            #### Plotting the results every given number of iterations ####

            if (t+1) % 500 == 0:
                plot_results(flow, X.detach(), t+1, savedir)

    return None

if __name__ == "__main__":
    main()
