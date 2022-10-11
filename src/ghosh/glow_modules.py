# This Script is a PyTorch implementation of the GLOW Model described in
# https://arxiv.org/abs/1807.03039

# Model based on: https://github.com/corenel/pytorch-glow
# Import the necessary libraries
import torch
import copy
from torch import nn
import torch.nn.functional as F
import numpy as np


# Defines a class for the ActNorm Module
class ActNorm(nn.Module):
    def __init__(
        self, num_channels, scale=1.0, logscale_factor=3.0, batch_variance=False
    ):

        super(ActNorm, self).__init__()
        self.num_channels = num_channels
        self.scale = scale
        self.logscale_factor = logscale_factor
        self.batch_variance = batch_variance
        self.bias_inited = False
        self.logs_inited = False
        self.register_parameter(
            "bias", nn.Parameter(torch.zeros(1, self.num_channels, 1))
        )
        self.register_parameter(
            "logs", nn.Parameter(torch.zeros(1, self.num_channels, 1))
        )

    def initialise_bias(self, x):
        """
        Initialise bias
        param x:input
        type x: torch.Tensor
        """
        with torch.no_grad():
            # Compute initial value
            x_mean = -1 * reduce_mean(x, dim=[0, 2], keepdim=True)
            # Copy to parameters
            self.bias.data.copy_(x_mean.data)
            self.bias_inited = True

    def initialize_logs(self, x):
        """
        Initialize logs
        param x: input
        """
        with torch.no_grad():
            if self.batch_variance:
                x_var = reduce_mean(x**2, keepdim=True)
            else:
                x_var = reduce_mean(x**2, dim=[0, 2], keepdim=True)
            logs = (
                torch.log(self.scale / (torch.sqrt(x_var) + 1e-6))
                / self.logscale_factor
            )
            # Copy to parameters
            self.logs.data.copy_(logs.data)
            self.logs_inited = True

    def forward(self, x, logdet=None):
        """
        Forward Activation Normalization layer (Data Space to the Latent Space)
        """
        assert len(x.shape) == 3
        assert (
            x.shape[1] == self.num_channels
        ), "Input shape should be N_B x N_C x N_S , however channels are {} instead of {}".format(
            x.shape[2], self.num_channels
        )

        if not self.logs_inited:
            self.initialize_logs(x)

        if not self.bias_inited:
            self.initialise_bias(x)

        logs = self.logs * self.logscale_factor
        z = (x + self.bias) * torch.exp(logs)
        if logdet is not None:

            # logdet_factor = count_pixels(x)
            logdet_factor = 1
            dlogdet = 1 * logdet_factor * torch.sum(logs)
            logdet += dlogdet

        return z, logdet

    def reverse(self, z, logdet=None):
        """
        Reverse Activation Generative layer (Latent space to Data Space)
        """
        assert len(z.shape) == 3
        assert (
            z.shape[1] == self.num_channels
        ), "Input shape should be N_B x N_C x N_S, however channels are {} instead of {}".format(
            z.shape[2], self.num_channels
        )

        if not self.logs_inited:
            self.initialize_logs(z)

        if not self.bias_inited:
            self.initialise_bias(z)

        logs = self.logs * self.logscale_factor
        x = (z * torch.exp(-logs)) - self.bias

        if logdet is not None:

            logdet_factor = 1
            # logdet_factor = count_pixels(z)
            dlogdet = -1 * logdet_factor * torch.sum(logs)
            logdet += dlogdet

        return x, logdet


# Defines a Linear layer with zeros Initialisation
class LinearZeros(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, logscale_factor=3.0):
        """
        Linear layer with zero initialization

        :param in_features: size of each input sample
        :type in_features: int
        :param out_features: size of each output sample
        :type out_features: int
        :param bias: whether to learn an additive bias.
        :type bias: bool
        :param logscale_factor: factor of logscale
        :type logscale_factor: float
        """
        super().__init__(in_features, out_features, bias)
        self.logscale_factor = logscale_factor
        # zero initialization
        self.weight.data.zero_()
        self.bias.data.zero_()
        # register parameter
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_features)))

    def forward(self, x):
        """
        Forward linear zero layer

        :param x: input
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        output = super().forward(x)
        output *= torch.exp(self.logs * self.logscale_factor)
        return output


class Conv1d(nn.Conv1d):
    @staticmethod
    def get_padding(padding_type, kernel_size, stride):
        """
        Get padding size.

        mentioned in https://github.com/pytorch/pytorch/issues/3867#issuecomment-361775080
        behaves as 'SAME' padding in TensorFlow
        independent on input size when stride is 1

        :param padding_type: type of padding in ['SAME', 'VALID']
        :type padding_type: str
        :param kernel_size: kernel size
        :type kernel_size: tuple(int) or int
        :param stride: stride
        :type stride: int
        :return: padding size
        :rtype: tuple(int)
        """
        assert padding_type in ["SAME", "VALID"], "Unsupported padding type: {}".format(
            padding_type
        )
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
            # kernel_size = [kernel_size, kernel_size]
        if padding_type == "SAME":
            assert stride == 1, "'SAME' padding only supports stride=1"
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1),
        stride=1,
        padding_type="SAME",
        do_weightnorm=True,
        do_actnorm=True,
        dilation=1,
        groups=1,
    ):
        """
        Wrapper of nn.Conv1d with weight normalization and activation normalization

        :param padding_type: type of padding in ['SAME', 'VALID']
        :type padding_type: str
        :param do_weightnorm: whether to do weight normalization after convolution
        :type do_weightnorm: bool
        :param do_actnorm: whether to do activation normalization after convolution
        :type do_actnorm: bool
        """
        padding = self.get_padding(padding_type, kernel_size, stride)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=(not do_actnorm),
        )
        self.do_weight_norm = do_weightnorm
        self.do_actnorm = do_actnorm

        nn.init.xavier_normal_(self.weight)
        # self.weight.data.normal_(mean=0.0, std=0.05)
        if self.do_actnorm:
            self.actnorm = ActNorm(out_channels)
        else:
            self.bias.data.zero_()

    def forward(self, x):
        """
        Forward wrapped Conv1d layer

        :param x: input
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        x = super().forward(x)

        if self.do_weight_norm:
            # normalize N, H and W dims
            x = F.normalize(x, p=2, dim=0)
            x = F.normalize(x, p=2, dim=2)

        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv1dZeros(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding_type="SAME",
        logscale_factor=3,
        dilation=1,
        groups=1,
        bias=True,
    ):
        """
        Wrapper of nn.Conv1d with zero initialization and logs

        :param padding_type: type of padding in ['SAME', 'VALID']
        :type padding_type: str
        :param logscale_factor: factor for logscale
        :type logscale_factor: float
        """
        padding = Conv1d.get_padding(padding_type, kernel_size, stride)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.logscale_factor = logscale_factor
        # initialize variables with zero
        self.bias.data.zero_()
        self.weight.data.zero_()
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1)))

    def forward(self, x):
        """
        Forward wrapped Conv2d layer

        :param x: input
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        x = super().forward(x)
        x *= torch.exp(self.logs * self.logscale_factor)
        return x


class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels, lu_decomposition=False):
        """
        Invertible 1x1 convolution layer

        :param num_channels: number of channels
        :type num_channels: int
        :param lu_decomposition: whether to use LU decomposition
        :type lu_decomposition: bool
        """
        super().__init__()
        self.num_channels = num_channels
        self.lu_decomposition = lu_decomposition
        w_shape = [num_channels, num_channels]
        tolerance = 1e-4
        # Sample a random orthogonal matrix
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype("float32")
        if self.lu_decomposition:
            w_LU_pts, pivots = torch.lu(torch.Tensor(w_init))
            p, w_l, w_u = torch.lu_unpack(w_LU_pts, pivots)
            s = torch.diag(torch.diag(w_u))
            w_u -= s
            print(
                (torch.Tensor(w_init) - torch.matmul(p, torch.matmul(w_l, (w_u + s))))
                .abs()
                .sum()
            )
            assert (
                torch.Tensor(w_init) - torch.matmul(p, torch.matmul(w_l, (w_u + s)))
            ).abs().sum() < tolerance
            self.register_parameter("weight_L", nn.Parameter(torch.Tensor(w_l)))
            self.register_parameter("weight_U", nn.Parameter(torch.Tensor(w_u)))
            self.register_parameter("s", nn.Parameter(torch.Tensor(s)))
            self.register_buffer("weight_P", torch.FloatTensor(p))

        else:
            # w_shape = [num_channels, num_channels]
            # Sample a random orthogonal matrix
            # w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))

    def forward(self, z, logdet=None):
        """

        :param z: input
        :type z: torch.Tensor
        :param logdet: log determinant
        :type logdet:
        :param reverse: whether to reverse bias
        :type reverse: bool
        :return: output and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        if not self.lu_decomposition:
            logdet_factor = 1
            # logdet_factor = count_pixels(z)  # H * W
            dlogdet = torch.log(torch.abs(torch.det(self.weight))) * logdet_factor
            # dlogdet = torch.slogdet(self.weight)[1] * logdet_factor
            weight = self.weight.view(*self.weight.shape, 1)
            x = F.conv1d(z, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return x, logdet
        else:
            # For LU decomposition W = P*L*(U + diag(s)) ,
            # where P = permutation matrix, L = lower traingular matrix with ones on diagonal,
            # U = upper triangular matrix with zeros on diagonal and diag(s) represents a
            # diagonal matrix with the elements of U in original LU decomposition, i.e. s = torch.diag(U)
            # after LU and then diagonal elements of U removed
            logdet_factor = 1
            # logdet_factor = count_pixels(z) # H * W
            dlogdet = (
                torch.sum(torch.log(torch.diag(torch.abs(self.s)))) * logdet_factor
            )
            weight = torch.matmul(
                self.weight_P, torch.matmul(self.weight_L, (self.weight_U + self.s))
            )
            weight = weight.view(*weight.shape, 1)
            x = F.conv1d(z, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return x, logdet

    def reverse(self, x, logdet=None):
        """
        :param x: input
        :type x: torch.Tensor
        :param logdet: log determinant
        :type logdet:
        :param reverse: whether to reverse bias
        :type reverse: bool
        :return: output and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        if not self.lu_decomposition:
            logdet_factor = 1
            # logdet_factor = count_pixels(x)  # H * W
            dlogdet = torch.log(torch.abs(torch.det(self.weight))) * logdet_factor
            # dlogdet = torch.slogdet(self.weight)[1] * logdet_factor
            weight = self.weight.inverse().view(*self.weight.shape, 1)
            z = F.conv1d(x, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet
        else:
            # For LU decomposition W = P*L*(U + diag(s)) ,
            # where P = permutation matrix, L = lower traingular matrix with ones on diagonal,
            # U = upper triangular matrix with zeros on diagonal and diag(s) represents a
            # diagonal matrix with the elements of U in original LU decomposition, i.e. s = torch.diag(U)
            # after LU and then diagonal elements of U removed
            logdet_factor = 1
            # logdet_factor = count_pixels(x) # H * W
            dlogdet = (
                torch.sum(torch.log(torch.diag(torch.abs(self.s)))) * logdet_factor
            )
            # TODO: Inverses here might be causing large gradients during backprop
            weight = torch.matmul(
                (self.weight_U + self.s).inverse().float(),
                torch.matmul(
                    self.weight_L.inverse().float(), self.weight_P.inverse().float()
                ),
            )
            weight = weight.view(*weight.shape, 1)
            z = F.conv1d(x, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class Permutation1d(nn.Module):
    def __init__(self, num_channels, shuffle=False, switch_flag=True):
        """
        Perform permutation on channel dimension

        :param num_channels:
        :type num_channels:
        :param shuffle:
        :type shuffle:
        """
        super().__init__()
        self.num_channels = num_channels
        self.indices = np.arange(self.num_channels - 1, -1, -1, dtype=np.long)
        if shuffle:
            np.random.shuffle(self.indices)
        self.indices_inverse = np.zeros(self.num_channels, dtype=np.long)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
        if shuffle == False and switch_flag == True:
            self.indices = torch.from_numpy(np.flip(self.indices).copy())
            self.indices_inverse = torch.from_numpy(
                np.flip(self.indices_inverse).copy()
            )

    def forward(self, z):

        assert len(z.shape) == 3
        return z[:, self.indices, :]

    def reverse(self, x):

        assert len(x.shape) == 3
        return x[:, self.indices_inverse, :]


class GaussianDiag:
    """
    Generator of gaussian diagonal matrix
    """

    log_2pi = float(np.log(2 * np.pi))

    @staticmethod
    def eps(shape_tensor, eps_std=None):
        """
        Returns a tensor filled with random numbers from a standard normal distribution

        :param shape_tensor: input tensor
        :type shape_tensor: torch.Tensor
        :param eps_std: standard deviation of eps
        :type eps_std: float
        :return: a tensor filled with random numbers from a standard normal distribution
        :rtype: torch.Tensor
        """
        eps_std = eps_std or 1.0
        return torch.normal(
            mean=torch.zeros_like(shape_tensor),
            std=torch.ones_like(shape_tensor) * eps_std,
        )

    @staticmethod
    def flatten_sum(tensor):
        """
        Summarize tensor except first dimension

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :return: summarized tensor
        :rtype: torch.Tensor
        """
        # assert len(tensor.shape) == 4
        # return ops.reduce_sum(tensor, dim=[1, 2, 3])
        assert len(tensor.shape) == 3
        # return reduce_sum(tensor, dim=[1, 2])
        return reduce_sum(tensor, dim=[1])

    @staticmethod
    def logps(mean, logs, x):
        """
        Likehood

        :param mean:
        :type mean: torch.Tensor
        :param logs:
        :type logs: torch.Tensor
        :param x: input tensor
        :type x: torch.Tensor
        :return: likehood
        :rtype: torch.Tensor
        """
        return -0.5 * (
            GaussianDiag.log_2pi
            + 2.0 * logs
            + ((x - mean) ** 2) / torch.exp(2.0 * logs)
        )

    @staticmethod
    def logp(mean, logs, x):
        """
        Summarized likehood

        :param mean:
        :type mean: torch.Tensor
        :param logs:
        :type logs: torch.Tensor
        :param x: input tensor
        :type x: torch.Tensor
        :return:
        :rtype: torch.Tensor
        """
        s = GaussianDiag.logps(mean, logs, x)
        return GaussianDiag.flatten_sum(s)

    @staticmethod
    def sample(mean, logs, eps_std=None):
        """
        Generate smaple

        :type mean: torch.Tensor
        :param logs:
        :type logs: torch.Tensor
        :param eps_std: standard deviation of eps
        :type eps_std: float
        :return: sample
        :rtype: torch.Tensor
        """
        eps = GaussianDiag.eps(mean, eps_std)
        return mean + torch.exp(logs) * eps


# Defining a class for implementing the Split1D operation (subclassed from nn.Module)
class Split1D(nn.Module):
    def __init__(self, num_channels, factor=2):
        """
        Performs the initialisation for the params of the Split operation
        """
        super(Split1D, self).__init__()
        self.factor = factor
        self.num_channels = num_channels
        self.conv1d_zeros = Conv1dZeros(num_channels // 2, num_channels)
        self.conv1d_zeros_f = Conv1dZeros(num_channels // 2, num_channels)
        self.conv1d_zeros_g = Conv1dZeros(num_channels, num_channels)

    def prior(self, z):
        """
        Pre-processing for the input tensor and obtain its mean and standard dev.
        """
        if z.shape[1] == 1:
            h = self.conv1d_zeros_f(z)
        elif z.shape[1] == 3:
            h = self.conv1d_zeros_g(z)
        else:
            h = self.conv1d_zeros(z)
        mean, logs = split_channel(h, "cross")
        return mean, logs

    def forward(self, x, logdet=0.0, eps_std=None):
        """
        Forward Split1D layer
        ---
        Args:

        - x: Input tensor
        - logdet: Log-determinant for the given layer computation
        - eps_std: Eps for standard deviation value

        Returns:
        - Tuple of tensors containing output of split and logdet
        """
        if self.factor == 1:
            return x, logdet
        elif self.factor >= 1:
            z1, z2 = split_channel(x, "simple")
            mean, log_s = self.prior(z1)
            logdet = GaussianDiag.logp(mean, log_s, z2) + logdet
            return z1, logdet

    def reverse(self, x, logdet=0.0, eps_std=None):
        """
        Reverse Split1D layer (basically implements a Join operation)
        ---
        Args:

        - x: Input tensor
        - logdet: Log-determinant for the given layer computation
        - eps_std: Eps for standard deviation value

        Returns:
        - Tuple of tensors containing output of join and logdet
        """
        if self.factor == 1:
            return x, logdet
        elif self.factor >= 1:
            z1 = copy.deepcopy(x)
            mean, log_s = self.prior(z1)
            z2 = GaussianDiag.sample(mean, log_s, eps_std)
            z = cat_channel(z1, z2)
            return z, logdet


# Defining a class for implementing the Squeeze1D operation (subclasssed from nn.Module)
class Squeeze1D(nn.Module):
    def __init__(self, factor=2):
        """
        Performs the initialisation of the factor param for the Squeeze operation
        """
        super(Squeeze1D, self).__init__()
        self.factor = factor

    @staticmethod
    def squeeze(x, factor=2):
        """
        This function performs the Squeeze operation, which is performed in the
        forward direction while going from the data space to the latent space. Basically
        reduces the spatial dimension in favour of increased number of channels
        ---
        Args:
        - x: Input Tensor which is shaped as (Nb x Nc x Ns)
        - factor: Scaling factor for channels which is basically implementing the
        Squeezing operation

        Returns:
        - Output Tensor of the shape (Nb x Nc * factor x Ns // factor)
        """
        assert (
            factor >= 1
        )  # Check if the factor is at least more than one, otherwise invalid
        if (
            factor == 1
        ):  # If the factor is 1, then there is no question of squeezing, identical tensor returned
            return x

        assert len(x.shape) == 3
        _, nc, ns = x.shape

        assert ns % factor == 0  # Check if x is divisible by the factor
        x = x.view(
            -1, nc, ns // factor, factor
        )  # Reshaping the number of spatial dimensions by adjusting the batch size
        x = x.permute(
            0, 1, 3, 2
        ).contiguous()  # Permute the dimensions so that output shape is similar to input shape
        x = x.view(
            -1, nc * factor, ns // factor
        )  # So that shape again becomes (Nb x Nc x Ns)

        return x

    @staticmethod
    def unsqueeze(z, factor=2):
        """
        This function performs the opposite of the Squeeze operation as defined above in
        squeeze()
        ---
        Args:
        - z: Input tensor which is shaped as (Nb x Nc * factor x Ns // factor)
        - factor: factor by which squeezing is to be reversed

        Returns:
        - Output tensor of the shape (Nb x Nc x Ns)
        """
        assert (
            factor >= 1
        )  # Check if the factor is at least more than one, otherwise invalid
        if (
            factor == 1
        ):  # If the factor is 1, then there is no question of squeezing, identical tensor returned
            return z

        assert len(z.shape) == 3
        _, nc, ns = z.shape

        assert nc >= factor and nc % factor == 0
        z = z.view(
            -1, nc // factor, factor, ns
        )  # Reducing the channel dimension by adjusting the batch size
        z = z.permute(
            0, 1, 3, 2
        ).contiguous()  # Permuting the dimensions so that shape is similar to original input
        z = z.view(
            -1, nc // factor, ns * factor
        )  # Reshaping the tensor with proper batch size
        return z

    def forward(self, x, logdet=None):
        """
        Forward Squeeze1d layer
        ---
        Args:
        - x: Input Tensor which is to be squeezed
        - logdet: From previous operations, stores the log-det value to be passed on
        (in case of Multi-scale architecture)

        Returns:
        - output: Output Tensor that is to be returned (Squeezed version)
        """
        output = self.squeeze(
            x, self.factor
        )  # Returns the output of a Squeeze operation
        return output, logdet

    def reverse(self, z, logdet=None):
        """
        Reverse Squeeze1d layer
        ---
        Args:
        - x: Input Tensor which is to be squeezed
        - logdet: From previous operations, stores the log-det value to be passed on
        (in case of Multi-scale architecture)

        Returns:
        - output: Output Tensor that is to be returned (Squeezed version)
        """
        output = self.unsqueeze(
            z, self.factor
        )  # Returns the output of an Unsqueeze operation
        return output, logdet
