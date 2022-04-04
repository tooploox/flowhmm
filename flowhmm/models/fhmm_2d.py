import os.path
from typing import Tuple

import numpy as np
import polyaxon.tracking
import torch
from hmmlearn.hmm import MultinomialHMM
from icecream import ic
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from torch import distributions as td

from flowhmm.utils import build_model_tabular, standard_normal_logprob


# import IPython

#
# def compute_MAD(
#     S_orig, B_orig, S2, P2
# ):  # wszystkie powinny byc wyliczone na grid_large
#
#     B_origT = B_orig.T
#     Q_orig = B_origT.matmul(S_orig.matmul(B_origT.T))
#
#     Q2 = P2.matmul(S2.matmul(P2.T))
#     r = P2.shape[1]
#     MAD = torch.sum(torch.abs(Q_orig - Q2)) / r ** 2
#     return MAD.cpu().detach().numpy()
#
#
# def compute_total_var_dist(B1, B2, grid_large):
#     L = B1.shape[0]
#     m_large = B1.shape[1]
#
#     B1_means = np.zeros(L)
#     B2_means = np.zeros(L)
#
#     for i in np.arange(L):
#         distr1 = B1[i, :]
#         distr1 = distr1 / np.sum(distr1)
#         B1_means[i] = np.sum(distr1 * grid_large)
#
#         distr2 = B2[i, :]
#         distr2 = distr2 / np.sum(distr2)
#         B2_means[i] = np.sum(distr2 * grid_large)
#
#     print("B1_means = ", B1_means)
#     print("B2_means = ", B2_means)
#
#     knn = KNeighborsClassifier(n_neighbors=1)
#     knn.fit(B1_means.reshape(-1, 1), np.arange(L))
#
#     ordering = knn.predict(B2_means.reshape(-1, 1))
#     print("ordering=", ordering)
#
#     total_vars = np.zeros(L)
#
#     for state1, state2 in enumerate(ordering):
#         # print("state1=", state1, ", state2=",state2)
#         distrA = B1[state2, :]
#         distrB = B2[state1, :]
#         distrA = distrA / np.sum(distrA)
#         distrB = distrB / np.sum(distrB)
#         total_vars[state1] = np.sum(np.abs(distrA - distrB)) / 2
#
#     return total_vars


def compute_stat_distr(A):
    evals, evecs = np.linalg.eig(A.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    stat_distr = evec1 / evec1.sum()
    stat_distr = stat_distr.real
    stat_distr = stat_distr.reshape(-1)
    return stat_distr


def nnmf_hmm_discrete(observations, m):
    Q = np.zeros((m, m))  # + 1

    for i in range(1, len(observations)):
        Q[observations[i - 1], observations[i]] += 1

    Q /= len(observations)  # - 1 + m * m # ??

    return Q


def showQQ(Q1, Q1_text, Q2, Q2_text):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(Q1, cmap="gray")
    axs[0].set_title(Q1_text)
    axs[1].imshow(Q2, cmap="gray")
    axs[1].set_title(Q2_text)
    plt.show()


def compute_P_torch(
    grid: torch.Tensor,
    means: torch.Tensor,
    cholesky_L_params: torch.Tensor,
    normalize=True,
    noise_var=0.01,
    add_noise=False,
):

    L = means.shape[0]

    # old P = torch.zeros(len(grid), len(means)).to(grid.device)

    P = torch.zeros(len(grid), L).to(grid.device)

    if add_noise:  #:torch.normal(mean=torch.zeros(5), std=torch.ones(5)*0.1)
        grid = grid + torch.normal(
            mean=torch.zeros((len(grid), 2)), std=torch.ones((len(grid), 2)) * 0.1
        )
    # grid = grid + torch.normal(0, 0.1, size=len(grid)).to(device)

    for i, (mean, chol_param) in enumerate(zip(means, cholesky_L_params)):

        Cholesky_L = torch.zeros((2, 2), device=cholesky_L_params.device)

        Cholesky_L[0, 0] = chol_param[0]
        Cholesky_L[1, 1] = chol_param[1]
        Cholesky_L[0, 1] = chol_param[2]
        Cholesky_L[1, 0] = 0  # chol_param[2]

        cov_matrix = torch.matmul(Cholesky_L, Cholesky_L.T)

        # dist_normal = td.Normal(loc=mean, scale=torch.sqrt(torch.exp(cov_un)))
        dist_normal = td.MultivariateNormal(loc=mean, covariance_matrix=cov_matrix)
        P[:, i] = dist_normal.log_prob(grid)

    P = torch.exp(P)
    if normalize:
        P = torch.nn.functional.normalize(P, p=1, dim=0)
    return P


def compute_Q_torch(
    grid: torch.Tensor,
    Shat: torch.Tensor,
    means: torch.Tensor,
    cholesky_params: torch.Tensor,
    add_noise: False,
    noise_var: 0.01,
):

    # P = torch.exp(compute_P_torch(grid, means, covs))
    P = compute_P_torch(
        grid=grid,
        means=means,
        cholesky_L_params=cholesky_params,
        add_noise=add_noise,
        noise_var=noise_var,
    )
    return P.matmul(Shat.matmul(P.T))


#
# def show_distrib(
#     distr1,
#     distr2,
#     P1_text="Distr. in P1",
#     P2_text="Distr. in P2",
#     grid_large=None,
#     grid=None,
#     show_points=True,
#     show_both_on_rhs=False,
# ):
#     LL1 = distr1.shape[0]
#     LL2 = distr2.shape[0]
#     LM = distr1.shape[1]
#
#     transparency = 0.2
#     ordering1 = np.arange(LL1)
#     ordering2 = np.arange(LL1)
#
#     # ordering2=[0,2,1]
#
#     colors = ["orange", "green", "blue", "brown"]
#
#     if grid_large is not None:
#         grid_large_plot = grid_large
#     else:
#         grid_large_plot = np.arange(LM)
#
#     gr_tmp = -0.1 * (
#         max(np.max(distr1), np.max(distr2)) - min(np.min(distr1), np.min(distr2))
#     )
#
#     plt.subplots(figsize=(15, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Two distr..")
#     if grid is not None:
#         plt.scatter(grid, gr_tmp * np.ones(len(grid)), label="grid", s=1)
#
#     plt.title(P1_text)
#     for nr in np.arange(LL1):
#         plt.plot(
#             grid_large_plot,
#             distr1[ordering1[nr], :],
#             label="P" + str(nr),
#             color=colors[ordering1[nr]],
#         )
#
#         if show_points:
#             plt.scatter(
#                 grid_large_plot, distr1[ordering1[nr], :], color=colors[ordering1[nr]]
#             )
#
#     # plt.legend()
#
#     plt.subplot(1, 2, 2)
#     if grid is not None:
#         plt.scatter(grid, gr_tmp * np.ones(len(grid)), label="grid", s=1)
#
#     plt.title(P2_text)
#     for nr in np.arange(LL2):
#         plt.plot(
#             grid_large_plot,
#             distr2[nr, :],
#             label="P" + str(nr),
#             color=colors[ordering2[nr]],
#         )
#         if show_points:
#             plt.scatter(grid_large_plot, distr2[nr, :], color=colors[ordering2[nr]])
#
#     for nr in np.arange(LL1):
#         if show_both_on_rhs:
#             plt.plot(
#                 grid_large_plot,
#                 distr1[ordering1[nr], :],
#                 label="P" + str(nr),
#                 alpha=transparency,
#                 color=colors[ordering1[nr]],
#             )
#
#     # polyaxon.tracking.log_mpl_image(plt.gcf(), name=P1_text + "__" + P2_text)
#     # plt.legend()
#     # plt.show()

#
# def compute_score_hmmlearn_multin(L, observations, m, get_params=False):
#     model1D_hmmlearn_multin_trained = MultinomialHMM(n_components=L)
#     model1D_hmmlearn_multin_trained.n_features = m
#     model1D_hmmlearn_multin_trained.fit(observations)
#     # print(" in score: obs.shape = ", observations.shape)
#     if get_params:
#         # logprob, mu, A, B
#         return (
#             model1D_hmmlearn_multin_trained.score(observations),
#             model1D_hmmlearn_multin_trained.startprob_,
#             model1D_hmmlearn_multin_trained.transmat_,
#             model1D_hmmlearn_multin_trained.emissionprob_,
#         )
#     else:
#         # logprob only
#         return model1D_hmmlearn_multin_trained.score(observations)


class HMM_NMF_multivariate(torch.nn.Module):
    """
    Hidden Markov Model -- NMF --
    """

    def __init__(
        self,
        Shat_un_init,
        means1d_hat_init,
        cholesky_L_params_init_2d,
        m,
        mm,
        loss_type="old",
    ):
        super(HMM_NMF_multivariate, self).__init__()

        self.L = Shat_un_init.shape[0]  # nr of hidden states

        self.Shat_un = torch.nn.Parameter(
            Shat_un_init.clone().detach().requires_grad_(True)
        )

        self.means1d_hat = torch.nn.Parameter(
            means1d_hat_init.clone().detach().requires_grad_(True)
        )

        self.cholesky_L_params = torch.nn.Parameter(
            cholesky_L_params_init_2d.clone().detach().requires_grad_(True)
        )

        # self.covs1d_hat_un = torch.nn.Parameter(
        #     covs1d_hat_un_init.clone().detach().requires_grad_(True)
        # )

        self.m = m
        self.mm = mm
        self.loss_type = loss_type
        self.device = self.means1d_hat.device
        # use the GPU
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            self.cuda()

    def get_S(self):
        Shat = torch.exp(self.Shat_un)
        return Shat / torch.sum(Shat)

    def get_A(self):
        S = self.get_S()
        A = S.cpu().detach().numpy()
        A = A / np.sum(A, axis=1).reshape(-1, 1)
        return A

    def get_means1d(self):
        return self.means1d_hat

    def get_cholesky_params(self):
        return self.cholesky_L_params

    def get_covs1d_un(self):
        return self.covs1d_hat_un

    def get_covs1d(self):
        return torch.exp(self.covs1d_hat_un)

    def show_params(self):
        ic("Param S:\t", self.get_S())
        ic("Param means:\t ", self.get_means1d())
        ic("Param covs:\t ", self.get_covs1d())

    # def forward(self, x_all):
    #    return 1;
    def get_mu(self):
        evals, evecs = np.linalg.eig(self.get_A().T)
        evec1 = evecs[:, np.isclose(evals, 1)]
        mu = evec1 / evec1.sum()
        mu = mu.real
        mu = mu.reshape(-1)
        return mu

    # def compute_P_torch(self, grid: torch.Tensor, add_noise=False, normalize=True):
    #     P = torch.zeros(len(grid), len(self.means1d_hat)).to(grid.device)
    #     for i, (mean, cov_un) in enumerate(zip(self.means1d_hat, self.covs1d_hat_un)):
    #         dist_normal = td.Normal(loc=mean, scale=torch.sqrt(torch.exp(cov_un)))
    #         P[:, i] = dist_normal.log_prob(grid)
    #
    #     P = torch.exp(P)
    #     if normalize:
    #         P = torch.nn.functional.normalize(P, p=1, dim=0)
    #     return P

    # def score(self, observations):
    #     # ROBIMY TAK: BIERZEMY MULTINOMIAL HMMLEARN I PODSTAWIAMY
    #     model1D_hmmlearn_torch_multin_trained = MultinomialHMM(n_components=self.L)
    #     model1D_hmmlearn_torch_multin_trained.fit(np.arange(self.m).reshape(-1, 1))
    #
    #     model1D_hmmlearn_torch_multin_trained.startprob_ = (
    #         self.get_mu().reshape(-1).astype(float)
    #     )
    #     model1D_hmmlearn_torch_multin_trained.transmat_ = self.get_A().astype(float)
    #
    #     P = compute_P_torch(
    #         torch.arange(self.m).to(self.device),
    #         self.get_means1d(),
    #         self.get_covs1d_un(),
    #     )
    #
    #     model1D_hmmlearn_torch_multin_trained.emissionprob_ = np.array(
    #         P.cpu().detach().numpy().T
    #     ).astype(float)
    #
    #     P = P + 1
    #     P[:, 0] = P[:, 0] / torch.sum(P[:, 0])
    #     P[:, 1] = P[:, 1] / torch.sum(P[:, 1])
    #     P[:, 2] = P[:, 2] / torch.sum(P[:, 2])
    #
    #     print(" in score: obs.shape = ", observations.shape)
    #
    #     return model1D_hmmlearn_torch_multin_trained.score(observations)
    #
    # def transition_model(self, log_alpha):
    #     A = self.get_S() / torch.sum(self.get_S(), dim=1).unsqueeze(1)
    #     log_transition_matrix = torch.log(A).transpose(1, 0)
    #
    #     # Matrix multiplication in the log domain
    #     out = self.log_domain_matmul(
    #         log_transition_matrix, log_alpha.view(-1, 1)
    #     ).transpose(0, 1)
    #     return out

    # def log_domain_matmul(self, log_A, log_B):
    #     """
    #     log_A : m x n
    #     log_B : n x p
    #     output : m x p matrix
    #
    #     Normally, a matrix multiplication
    #     computes out_{i,j} = sum_k A_{i,k} x B_{k,j}
    #
    #     A log domain matrix multiplication
    #     computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
    #     """
    #     m = log_A.shape[0]
    #     n = log_A.shape[1]
    #     p = log_B.shape[1]
    #
    #     # log_A_expanded = torch.stack([log_A] * p, dim=2)
    #     # log_B_expanded = torch.stack([log_B] * m, dim=0)
    #     # fix for PyTorch > 1.5 by egaznep on Github:
    #     log_A_expanded = torch.reshape(log_A, (m, n, 1))
    #     log_B_expanded = torch.reshape(log_B, (1, n, p))
    #
    #     elementwise_sum = log_A_expanded + log_B_expanded
    #     out = torch.logsumexp(elementwise_sum, dim=1)
    #
    #     return out
    #
    # def continuous_score(self, observations):
    #     log_probs = []
    #     x_all = observations.squeeze()
    #     log_alpha = torch.zeros(x_all.shape[0], self.L).to(self.device)
    #     log_state_priors = torch.log(torch.tensor(self.get_mu()).to(self.device))
    #     x_tensor = torch.tensor(x_all).float().to(self.device)
    #     for i, (mean, cov_un) in enumerate(zip(self.means1d_hat, self.covs1d_hat_un)):
    #         dist_normal = td.Normal(loc=mean, scale=torch.sqrt(torch.exp(cov_un)))
    #         log_px = dist_normal.log_prob(x_tensor)
    #         log_probs.append(log_px)
    #     emt = torch.stack([log_prob for log_prob in log_probs]).T.squeeze()
    #     log_alpha[0] = emt[0] + log_state_priors
    #     for t, x_t in enumerate(x_all[1:], 1):
    #         # transition_model bierze alphy z poprzedniego kroku i tam w srodku uzywa A
    #         log_alpha[t] = emt[t] + self.transition_model(log_alpha[t - 1])
    #
    #     # Select the sum for the final timestep (each x may have different length).
    #     log_sums = log_alpha.logsumexp(dim=1)
    #
    #     return log_sums[-1].detach().cpu().numpy()

    def fit(
        self,
        grid,
        observation_labels,
        nr_epochs=5000,
        lr=0.01,
        display_info_every_step=50,
        add_noise=False,
        noise_var=0.01,
    ):

        Q_empir = nnmf_hmm_discrete(observation_labels, self.mm)
        Q_empir_torch = torch.from_numpy(Q_empir).to(self.device)

        # print("Q_empir = ", Q_empir)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.00001)
        for it in np.arange(nr_epochs):

            optimizer.zero_grad()

            Shat = torch.exp(self.Shat_un)
            Shat = Shat / torch.sum(Shat)

            Q_torch = compute_Q_torch(
                grid,
                Shat,
                self.means1d_hat,
                self.cholesky_L_params,
                add_noise=add_noise,
                noise_var=0.01,
            )
            if self.loss_type == "old":
                loss = torch.norm(Q_torch - Q_empir_torch)
            elif self.loss_type == "kld":
                loss = torch.sum(
                    Q_torch
                    * (
                        torch.log(0.001 + Q_torch)
                        - torch.log(0.001 + Q_empir_torch.float())
                    )
                )
            else:
                loss = None

            loss.backward(retain_graph=True)
            optimizer.step()
            if it < 50 or np.mod(it, display_info_every_step) == 0:
                print(
                    "Epoch = ",
                    it,
                    "/",
                    nr_epochs,
                    ",\t loss: ",
                    np.round(loss.cpu().detach().numpy(), 6),
                )
                polyaxon.tracking.log_metric(
                    "train/loss", loss.cpu().detach().numpy(), step=it
                )

        return True


class ListModule(torch.nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError("index {} is out of range".format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
