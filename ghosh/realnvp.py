import torch

class Rescale(torch.nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, 1, num_channels))

    def forward(self, x):
        x = self.weight * x
        return x

class RealNVP(torch.nn.Module):
    """ This is a class for implementing the Real NVP (Non-Volume Preserving) network 
    for Density Estimation. It is based on the principle of invertible, bijective mappings
    or transformations that map a given complex distribution of data to a latent space, 
    through tractable density and sampling calculation as well as tractable inverses.
    The main function is the change of variable rule that is defined by :

    P_x = P_z * det ( df(x) / dx )

    Normalizing (Inference): z = f(x)
    Generation (Sampling): x = g(z)
    -------
    Args: 
          nets: Defines a Network that is to be used as the 'conditioner' function.
                Also serves for implementation of the scale (s) and the transition(t)
                functions of the network

          mask: The transformations are defined often through the use of masks which are
                applied as element wise operations on the input. Mostly they help to split 
                the data into two halves (the identical half and the scaling half) at each 
                coupling layer

          prior: This defines the prior to be used for the distribution on the latent space.
                 Usually, the prior is modeled using a Standard multivariate normal distribution
                 (with zero mean vector  and identity covariance matrix)
    
    Methods: 
          _chunk() : This function is used to divide the function into two possible halves. The 
                  division is decided by the mask applied and takes place along the second dimension.
                
          g() : This function is actually the sampling function that is used for generating a point 
             in the data space X from a point in the latent space Z. 
            
          f() : This function is actually the inference function that is used for inferring a point 
             in the latent space Z from a point in the data space X. It also additionally
             calculates the log of the determinant of the Jacobian matrix that is associated
             with the RealNVP transformation.

          log_prob() : This function is used for estimating the logarithm of the prob density of 
                       the data points in the Data Space X. It follows the change of variable formula
                       as outlined above.
          
          sample() : This function is used to draw a sample from the learnt distribution of the data
                     from the latent space Z
             
    """
    """
    RealNVP module.
    Adapted from https://github.com/senya-ashukha/real-nvp-pytorch
    """
    def __init__(self, nets, mask, prior):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        self.mask = torch.nn.Parameter(mask, requires_grad=False)
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])
        self.rescale = torch.nn.utils.weight_norm(Rescale(int(self.mask.size(1)/2)))
    
    
    def _chunk(self, x, mask):
        """chunk the input x into two chunks along dimension 2
        INPUT: tensor to be chunked, shape: batch_size * n_samples * n_features
        OUTPUT: tow chunks of tensors with equal size
        """
        idx_id = torch.nonzero(mask).reshape(-1)
        idx_scale = (mask == 0).nonzero().reshape(-1)
        #idx_scale = torch.nonzero(~mask).reshape(-1)
        chunk_id = torch.index_select(x, dim=2,
                                      index=idx_id)
        chunk_scale = torch.index_select(x, dim=2,
                                         index=idx_scale)
        return (chunk_id, chunk_scale)
        
    def g(self, z):
        """ This function defines the generation or sampling step. It is the exact opposite of the inference
        step and the masking operations are similar. 

        """
        # not sure about this
        x = z
        # for i in range(len(self.t)):
        #     x_ = x * self.mask[i]
        #     s = self.s[i](x_) * (1 - self.mask[i])
        #     t = self.t[i](x_) * (1 - self.mask[i])
        #     x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        # return x
        pass

    def f(self, x):
        """ This function defines the inference step. The flow of the network is something like:
        
        Z (Latent Space) <-- f^{1} <-- f^{2} ... <-- f^{L} <-- X (Data Space)
        
        Hence, the list is reversed first and then iterated over all the layers
        The input is split into two parts z_id and z_s. The part involving z_id is 
        passed through as identical and the part involving z_id is subjected to two operations of 
        scaling (s(z_id)) and translation (t(z_id)) and mxied with z_s. 
        
        The scaling and the translation output obtained here
        as variables s and t are chunked. The scaling output is actually to processed as tanh and the
        translation output remains as an affine transform output. 
        
        The determinant of the jacobian matrix of this computation step is given as a sum of the
        scaling outputs and thus easily computed without derivatives of s or t functions
        """

        log_det_J, z = x.new_zeros((x.shape[0], x.shape[1])), x
        for i in reversed(range(len(self.s))):
            z_id, z_s = self._chunk(z, self.mask[i])
            
            st = self.s[i](z_id)
            s, t = st.chunk(2,dim=2)
            s = self.rescale(torch.tanh(s))
            
            exp_s = s.exp()
            z_s = (z_s + t) * exp_s
            z =  torch.cat((z_id, z_s), dim=2)
            
            log_det_J += torch.sum(s, dim=2)
        return z, log_det_J

    def log_prob(self, x, mask):
        """The prior log_prob may need be implemented such it adapts cuda computation."""
        z, logp = self.f(x)

        px = self.prior.log_prob(z) + logp
        # set the padded positions as zeros
        px[~mask] = 0
        # px[~mask].zero_()
        #if (px > 0).any():
          #  print("here")
        return px

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x
