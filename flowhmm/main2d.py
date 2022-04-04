import argparse
import os
import pickle
from typing import Optional, Dict

import numpy as np
import pandas as pd
import polyaxon
import polyaxon.tracking
import scipy.stats
import torch
from hmmlearn.hmm import GaussianHMM
from icecream import ic
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from termcolor import colored

from models.fhmm import HMM_NMF, HMM_NMF_FLOW
from models.fhmm import compute_stat_distr
from models.fhmm import show_distrib, compute_total_var_dist, compute_MAD

from models.fhmm_2d import HMM_NMF_multivariate


from utils import set_seed, load_example


from matplotlib.patches import Ellipse

# sample usage
# python -i flowhmm/main2d.py -e examples/SYNTHETIC_2d_data_2G_1U.yaml --nr_epochs_torch 200  --show_plots=yes
# python flowhmm/main2d.py -e examples/SYNTHETIC_2d_data_2G_1U.yaml --nr_epochs_torch 5000 --seed 139


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def ParseArguments():
    NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
    SOLVERS = [
        "dopri5",
        "bdf",
        "rk4",
        "midpoint",
        "adams",
        "explicit_adams",
        "fixed_adams",
    ]
    LAYERS = [
        "ignore",
        "concat",
        "concat_v2",
        "squash",
        "concatsquash",
        "concatcoord",
        "hyper",
        "blend",
    ]

    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument(
        "-e",
        "--example_yaml",
        type=str,
        default="examples/SYNTHETIC_2d_data_2G_1U.yaml",
        help="Path to example YAML config file",
    )
    parser.add_argument("--show_plots", default="yes", type=str, help="Show plots?")
    parser.add_argument(
        "--nr_epochs", default=10, type=int, required=False, help="nr of epochs"
    )
    parser.add_argument("--loss_type", type=str, default="kld", choices=["old", "kld"])

    parser.add_argument("--pretrain_flow", type=eval, default=False)
    parser.add_argument(
        "--nr_epochs_torch",
        default=10,
        type=int,
        required=False,
        help="nr of epochs for torch",
    )
    parser.add_argument(
        "--init_with_kmeans",
        type=boolean_string,
        default=True,
        required=False,
        help="Init with kmeans' centers",
    )
    parser.add_argument(
        "--set_seed",
        type=boolean_string,
        default=True,
        required=False,
        help="False = do not set",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        required=False,
        help="default seed"
        #    "--seed", default=116, type=int, required=False, help="default seed"
    )
    parser.add_argument("--lrate", default="0.01", required=False, help="learning rate")
    parser.add_argument(
        "--output_file", default=None, required=False, help="file to save results (pkl)"
    )
    parser.add_argument(
        "--layer_type",
        type=str,
        default="concatsquash",
        choices=LAYERS,
    )
    parser.add_argument("--dims", type=str, default="16-16")
    parser.add_argument(
        "--num_blocks", type=int, default=2, help="Number of stacked CNFs."
    )
    parser.add_argument("--time_length", type=float, default=0.5)
    parser.add_argument("--train_T", type=eval, default=True)
    parser.add_argument("--add_noise", type=eval, default=False, choices=[True, False])
    parser.add_argument("--noise_var", type=float, default=0.01)
    parser.add_argument(
        "--divergence_fn",
        type=str,
        default="brute_force",
        choices=["brute_force", "approximate"],
    )
    parser.add_argument(
        "--nonlinearity", type=str, default="tanh", choices=NONLINEARITIES
    )

    parser.add_argument("--solver", type=str, default="dopri5", choices=SOLVERS)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--step_size", type=float, default=None, help="Optional fixed step size."
    )

    parser.add_argument(
        "--test_solver", type=str, default=None, choices=SOLVERS + [None]
    )
    parser.add_argument("--test_atol", type=float, default=None)
    parser.add_argument("--test_rtol", type=float, default=None)

    parser.add_argument("--residual", type=eval, default=False, choices=[True, False])
    parser.add_argument("--rademacher", type=eval, default=False, choices=[True, False])
    parser.add_argument(
        "--spectral_norm", type=eval, default=False, choices=[True, False]
    )
    parser.add_argument("--batch_norm", type=eval, default=True, choices=[True, False])
    parser.add_argument("--bn_lag", type=float, default=0)
    parser.add_argument("--polyaxon", type=boolean_string, default=False)
    parser.add_argument("--extra_n", type=int, required=False)
    parser.add_argument("--extra_L", type=int, required=False)
    args = parser.parse_args()
    return args


def simulate_observations(n, mu, transmat, distributions):
    def sample(name: str, params: Optional[Dict]):
        mapping = {
            "beta": np.random.beta,
            "uniform": np.random.uniform,
            "normal": np.random.normal,
        }
        func = mapping[name]
        return func(**params)

    observations = np.zeros(n)
    n_states = len(distributions)
    current_state = np.random.choice(np.arange(n_states), size=1, p=mu.reshape(-1))[0]
    for k in np.arange(n):
        observations[k] = sample(**distributions[current_state])
        current_state = np.random.choice(
            np.arange(n_states), 1, p=transmat[current_state, :].reshape(-1)
        )[0]
    return observations


def simulate_observations_multivariate(n, mu, transmat, distributions):
    # def sample(name: str, params: Optional[Dict]):
    #     mapping = {
    #         "beta": np.random.beta,
    #         "uniform": np.random.uniform,
    #         "normal": np.random.normal,
    #     }
    #     func = mapping[name]
    #     return func(**params)

    # dimension:
    dim = 2  # to powinno sie odczytac z params

    observations = np.zeros((n, dim))
    # n_states = len(distributions)
    n_states = transmat.shape[0]
    current_state = np.random.choice(np.arange(n_states), size=1, p=mu.reshape(-1))[0]
    for k in np.arange(n):
        ## to na pewno da sie lepiej:
        if distributions[current_state]["name"] == "uniform":
            params_low = distributions[current_state]["params"]["low"]
            params_high = distributions[current_state]["params"]["high"]
            observations[k, :] = [
                np.random.uniform(params_low[0], params_high[0]),
                np.random.uniform(params_low[1], params_high[1]),
            ]
            # sample(**distributions[current_state])

        if distributions[current_state]["name"] == "normal":
            params_mean = distributions[current_state]["params"]["mean"]
            params_cov = distributions[current_state]["params"]["cov"]
            observations[k, :] = np.random.multivariate_normal(params_mean, params_cov)

        current_state = np.random.choice(
            np.arange(n_states), 1, p=transmat[current_state, :].reshape(-1)
        )[0]
    return observations


def get_dist(name, params):
    if name == "beta":
        return scipy.stats.beta(a=params["a"], b=params["b"])
    elif name == "uniform":
        return scipy.stats.uniform(
            loc=params["low"], scale=params["high"] - params["low"]
        )
    elif name == "normal":
        return scipy.stats.norm(loc=params["loc"], scale=params["scale"])
    else:
        raise NotImplementedError


def compute_pdf_matrix(distributions, grid):
    L = len(distributions)
    grid_size = len(grid)

    B_orig_large = np.zeros((L, grid_size))
    for i, distribution in enumerate(distributions):
        B_orig_large[i, :] = get_dist(**distribution).pdf(grid)
    return B_orig_large


def draw_ellipse(position, covariance, ax, alpha):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, alpha=alpha))


def main():
    args = ParseArguments()
    ic(args)
    polyaxon.tracking.init(is_offline=not args.polyaxon)
    polyaxon.tracking.log_inputs(args=args.__dict__)

    init_with_kmeans = bool(args.init_with_kmeans)
    print("init_with_kmeans = ", init_with_kmeans)

    if args.set_seed:
        set_seed(args.seed)
    example_config = load_example(args.example_yaml)

    EXAMPLE, _ = os.path.basename(args.example_yaml).rsplit(".", 1)
    ic(EXAMPLE, example_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    show_plots = args.show_plots == "yes"
    nr_epochs = args.nr_epochs
    nr_epochs_torch = args.nr_epochs_torch
    add_noise = args.add_noise
    noise_var = args.noise_var
    loss_type = args.loss_type
    n = example_config.nr_observations
    print(
        "AAAAAAA nr_epochs=",
        nr_epochs,
        ", nr_epochs_torch=",
        nr_epochs_torch,
        ", n=",
        n,
    )

    lrate = float(args.lrate)

    output_file = args.output_file

    # Debugging
    ic(device, show_plots, nr_epochs, nr_epochs_torch, lrate)

    obs_flow = None
    obs_test_flow = None
    obs_test_all = None
    obs_train_nr_of_seq = 1

    # if(show_plots):
    #     show_distrib(B_orig, B_orig, P1_text="B_orig", P2_text="B_orig")

    generated_bool = False
    print("A")
    DATA_TYPE = example_config.data_type.upper()
    print("B")
    polyaxon.tracking.log_inputs(yaml=example_config)
    if example_config.dataset:
        polyaxon.tracking.log_inputs(**example_config.dataset)
    # REAL DATASETS
    if example_config.data_type == "real":
        dataset = example_config.dataset
        # A_orig = None
        # df = pd.read_csv(dataset["filepath"], **dataset["load_args"])
        # ic(dataset, df.columns, df.describe())
        #
        # df_train = df[dataset["column"]].values
        # polyaxon.tracking.log_dataframe(df, "df_raw")
        # # df_train = (df_train - df_train.mean()) / df_train.std()
        # polyaxon.tracking.log_dataframe(df, "df_train")
        # ic(pd.DataFrame(df_train).describe())
        # n = example_config.nr_observations
        # n_train = n
        # n_test = min(n_train, len(df_train) - n_train)
        # obs_train = np.array(df_train[:n_train], dtype=np.float)
        # obs_test = np.array(df_train[n_train : n_train + n_test], dtype=np.float)
        #
        # m = example_config.grid_size
        # L = example_config.nr_hidden_states
        #
        # x_min = np.min(obs_train) - 0.05 * np.abs(np.min(obs_train))
        # x_max = np.max(obs_train) + 0.05 * np.abs(np.max(obs_train))
        #
        # grid_strategy = example_config.grid_strategy
        # # grid_strategy = "uniform"
        # # grid_strategy = "kmeans"
        # # grid_strategy = "mixed"
        #
        # if grid_strategy == "uniform":
        #     grid = np.linspace(x_min, x_max, m)
        #
        # if grid_strategy == "kmeans":
        #     kmeans = KMeans(n_clusters=m)
        #     kmeans.fit(obs_train.reshape(-1, 1))
        #     grid = kmeans.cluster_centers_.reshape(-1)
        #
        # if grid_strategy == "mixed":
        #     kmeans = KMeans(n_clusters=int(np.floor(m / 2)))
        #     kmeans.fit(obs_train.reshape(-1, 1))
        #     grid_kmeans = kmeans.cluster_centers_.reshape(-1)
        #     grid_unif = np.linspace(x_min, x_max, m - int(np.floor(m / 2)))
        #
        #     grid = np.sort(np.concatenate((grid_kmeans, grid_unif)))
        #
        # ic(grid)
        #
        # grid_labels = list(range(m))
        # grid_large = np.linspace(np.min(grid), np.max(grid), m * 10)
        # # plt.show()
        #
        # m_large = len(grid_large)
        #
        # grid_labels = list(range(m))
        #
        # knn = KNeighborsClassifier(n_neighbors=1)
        #
        # knn.fit(grid.reshape(-1, 1), grid_labels)
        #
        # obs_train_grid_labels = (
        #     knn.predict(obs_train.reshape(-1, 1)).reshape(-1, 1).astype(int)
        # )
        # obs_test_grid_labels = (
        #     knn.predict(obs_test.reshape(-1, 1)).reshape(-1, 1).astype(int)
        # )
        #
        # obs_train_grid = grid[obs_train_grid_labels]
        # obs_test_grid = grid[obs_test_grid_labels]

        generated_bool = False
    # SYNTHETIC DATASETS
    elif example_config.data_type == "synthetic":

        L = example_config.nr_hidden_states or len(
            example_config.hidden_states_distributions
        )

        dim = 2  # na razie recznie

        m = example_config.grid_size  # na jednej wspolrzednej
        n = example_config.nr_observations
        if args.extra_n:
            n = args.extra_n
        if args.extra_L:
            L = args.extra_L

        A_orig = np.array(example_config.transition_matrix)
        mu_orig = compute_stat_distr(A_orig)
        S_orig = torch.tensor(np.dot(np.diag(mu_orig), A_orig))

        # SIMULATE OBSERVATIONS:
        obs_train = simulate_observations_multivariate(
            n,
            mu=mu_orig,
            transmat=A_orig,
            distributions=example_config.hidden_states_distributions,
        )
        obs_test = simulate_observations_multivariate(
            n,
            mu=mu_orig,
            transmat=A_orig,
            distributions=example_config.hidden_states_distributions,
        )

        # quit()
        x_min = np.min(obs_train[:, 0]) - 0.05 * np.abs(np.min(obs_train[:, 0]))
        x_max = np.max(obs_train[:, 0]) + 0.05 * np.abs(np.min(obs_train[:, 0]))

        y_min = np.min(obs_train[:, 1]) - 0.05 * np.abs(np.min(obs_train[:, 1]))
        y_max = np.max(obs_train[:, 1]) + 0.05 * np.abs(np.min(obs_train[:, 1]))

        #
        grid_strategy = example_config.grid_strategy
        # # grid_strategy = "uniform"
        # # grid_strategy = "kmeans"
        # # grid_strategy = "mixed"
        #
        if grid_strategy == "uniform":
            grid_x = np.linspace(x_min, x_max, m)
            grid_y = np.linspace(y_min, y_max, m)
            # cartesian product
            grid_all = np.transpose(
                [np.tile(grid_x, len(grid_y)), np.repeat(grid_y, len(grid_x))]
            )
            # m = grid size na 1 wspolrzednej, ostateczny jest mm
            mm = grid_all.shape[0]

        elif grid_strategy == "kmeans":
            mm = example_config.grid_size_all
            kmeans = KMeans(n_clusters=mm)
            kmeans.fit(obs_train)
            grid_all = kmeans.cluster_centers_

        elif grid_strategy == "kmeans2":
            mm = example_config.grid_size_all
            # tutaj robimy tak, ze osobno na x, osobno na y

            kmeans_x = KMeans(n_clusters=m)
            kmeans_x.fit(obs_train[:, 0].reshape(-1, 1))
            grid_x = kmeans_x.cluster_centers_.reshape(-1)

            kmeans_y = KMeans(n_clusters=m)
            kmeans_y.fit(obs_train[:, 1].reshape(-1, 1))
            grid_y = kmeans_y.cluster_centers_.reshape(-1)

            grid_all = np.transpose(
                [np.tile(grid_x, len(grid_y)), np.repeat(grid_y, len(grid_x))]
            )
            # grid_all=np.zeros((mm,2))
            # grid_all[:,0]=grid_x
            # grid_all[:,1]=grid_y
            mm = grid_all.shape[0]
            print("test")
        # elif grid_strategy == "mixed":
        #     kmeans = KMeans(n_clusters=int(np.floor(m / 2)))
        #     kmeans.fit(obs_train.reshape(-1, 1))
        #     grid_kmeans = kmeans.cluster_centers_.reshape(-1)
        #     grid_unif = np.linspace(x_min, x_max, m - int(np.floor(m / 2)))
        #
        #     grid = np.sort(np.concatenate((grid_kmeans, grid_unif)))
        # elif grid_strategy == "arange_0_max":
        #     grid = np.arange(0, np.ceil(x_max) + 2)
        #     m = len(grid)
        # else:
        #     raise ValueError(f"Unknown grid_strategy={grid_strategy}")
        #
        # ic(grid)
        # polyaxon.tracking.log_inputs(n=n)
        # polyaxon.tracking.log_inputs(L=L)
        # polyaxon.tracking.log_inputs(m=m)
        #

        # Pamietajmy: m=grid_all.shape[0]
        grid_labels = list(range(mm))
        # old: grid_labels = list(range(m))

        # grid_large = np.linspace(np.min(grid), np.max(grid), m * 10)
        #
        # m_large = len(grid_large)
        #
        # grid_labels = list(range(m))
        #

        knn = KNeighborsClassifier(n_neighbors=1)
        #
        knn.fit(grid_all, grid_labels)
        #
        obs_train_grid_labels = knn.predict(obs_train).reshape(-1, 1).astype(int)
        obs_test_grid_labels = knn.predict(obs_test).reshape(-1, 1).astype(int)
        #
        obs_train_grid = grid_all[obs_train_grid_labels].reshape(n, dim)
        obs_test_grid = grid_all[obs_test_grid_labels].reshape(n, dim)
        #
        # B_orig_large = compute_pdf_matrix(
        #     example_config.hidden_states_distributions, grid_large
        # )

        generated_bool = False
    else:
        raise ValueError(f"Unknown example.data_type {example_config.data_type}")

    polyaxon.tracking.log_outputs(m=m, L=L, grid_strategy=grid_strategy)

    ################
    # INFER START
    ################

    if show_plots:
        plt.figure()
        show_nr_points = np.minimum(1000, obs_train.shape[0])
        plt.title(
            "Continuous observations: "
            + str(show_nr_points)
            + " of 2d points from obst_train"
        )
        plt.scatter(
            grid_all[:, 0],
            grid_all[:, 1],
            s=4,
            color="gray",
            label="grid",
            alpha=0.5,
            marker="+",
        )
        plt.scatter(
            obs_train[:show_nr_points, 0],
            obs_train[:show_nr_points, 1],
            s=6,
            color="red",
            label="cont. obs",
        )
        plt.scatter(
            obs_train_grid[:show_nr_points, 0],
            obs_train_grid[:show_nr_points, 1],
            s=16,
            color="green",
            label="discr. obs",
            alpha=0.6,
        )

        plt.legend()

        plt.figure()
        show_nr_points = np.minimum(1000, obs_train_grid.shape[0])
        plt.title(
            "Discretized observations: "
            + str(show_nr_points)
            + " of 2d points from obst_train"
        )
        plt.scatter(
            grid_all[:, 0],
            grid_all[:, 1],
            s=14,
            color="gray",
            label="grid",
            alpha=0.5,
            marker="+",
        )
        plt.scatter(
            obs_train_grid[:show_nr_points, 0],
            obs_train_grid[:show_nr_points, 1],
            s=6,
            color="red",
            label="discr. obs",
        )
        plt.legend()

    # RECZNIE rozmiary, do zmiany

    L2 = L

    # Gauss HMM
    # hmmlearn GaussianHMM
    model_hmmlearn_gaussian_trained = GaussianHMM(
        n_components=L2, covariance_type="full"
    )
    model_hmmlearn_gaussian_trained.fit(obs_train)

    logprob_hmmlearn_gaussian_trained = model_hmmlearn_gaussian_trained.score(obs_test)

    model_hmmlearn_gaussian_grid_trained = GaussianHMM(
        n_components=L2, covariance_type="full"
    )
    model_hmmlearn_gaussian_grid_trained.fit(obs_train_grid)

    logprob_hmmlearn_gaussian_grid_trained = model_hmmlearn_gaussian_grid_trained.score(
        obs_test
    )

    # Gauss TORCH
    print(
        "DDDDDDDDD init_with_kmeans =",
        init_with_kmeans,
        ", args.init_with_kmeans = ",
        args.init_with_kmeans,
    )
    if init_with_kmeans == True:
        kmeans = KMeans(n_clusters=L, n_init=10)
        kmeans.fit(obs_train_grid)
        means1d_hat_init_2d = torch.nn.Parameter(
            torch.tensor(kmeans.cluster_centers_)
        ).to(device)
        print("TTTTTTTTTTT")

    else:
        tmp = torch.rand(
            (L, dim)
        )  # dla kazdego ukrytego stanu dim-wymiarowy punkt = srednia
        tmp[:, 0] = tmp[:, 0] * (x_max - x_min) + x_min
        tmp[:, 1] = tmp[:, 1] * (y_max - y_min) + y_min
        means1d_hat_init_2d = torch.nn.Parameter(tmp).to(device)
        print("FFFFFFFFFFF")

    print("asdf")

    # np.
    # tensor([[3.3121, 2.7315],
    # [3.0476, 2.1365],
    # [3.0364, 2.5332]], requires_grad=True)

    print("means1d_hat_init=", means1d_hat_init_2d)

    # to tak jak bylo w 1d
    Shat_un_init = torch.nn.Parameter(torch.randn(L2, L2)).to(device)
    # un = unnormalized
    # covs1d_hat_un_init = torch.nn.Parameter(torch.randn(L2) / 2).to(device)

    # UWAGA: to trzeba sprytniej... maja byc macierze, ale symetryczne..
    # dlatego parametrem dla kazdego stanu ukrytego jest (c00,c01,c11)
    # z czego potem robimy macierz L = [[c00,0],[c01,c11]] = dolnotrojkatna
    # a potem covariance_matrix = L*L^T (rozklad Choleskyego)

    cholesky_L_params_init_2d = torch.nn.Parameter((2 * torch.rand(L2, 3) - 1) / 10).to(
        device
    )
    ic(cholesky_L_params_init_2d)
    # na przyklad:
    #  tensor([[ 0.4633, -0.1349,  0.1217],
    #     [-0.1054, -0.0575,  0.0717],
    #     [-0.1036, -0.2325, -0.0115]], requires_grad=True)

    model_hmm_nmf_torch_multivariate = HMM_NMF_multivariate(
        Shat_un_init=Shat_un_init,
        means1d_hat_init=means1d_hat_init_2d,
        # covs1d_hat_un_init=covs1d_hat_un_init_2d,
        # lepsza nazwa:
        cholesky_L_params_init_2d=cholesky_L_params_init_2d,
        m=m,
        mm=mm,
        # loss_type="old" #
        loss_type=loss_type,
    )

    # print("Test")
    # P_torch_init_large = model_hmm_nmf_torch.compute_P_torch(
    #     torch.tensor(grid_large).to(device), normalize=False
    # )

    print(colored("Fitting model_hmm_nmf_torch ... ", "red"))
    model_hmm_nmf_torch_multivariate.fit(
        torch.Tensor(grid_all).to(device),
        obs_train_grid_labels.reshape(-1),
        lr=lrate,
        nr_epochs=nr_epochs_torch,
        add_noise=add_noise,
        noise_var=noise_var,
    )
    print(colored("DONE (fitting model_hmm_nmf_torch) ", "red"))

    print("means1d_hat_init=", means1d_hat_init_2d)

    trained_means = model_hmm_nmf_torch_multivariate.get_means1d()
    trained_cholesky_L_params = model_hmm_nmf_torch_multivariate.get_cholesky_params()

    print("trained_means =", trained_means)

    if show_plots:
        if add_noise:
            print("Add noise var: ", noise_var)
            grid_all_noise = torch.tensor(grid_all) + torch.normal(
                mean=torch.zeros((len(grid_all), 2)),
                std=torch.ones((len(grid_all), 2)) * noise_var,
            )
            grid_all_noise = grid_all_noise.detach().numpy()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(
                grid_all[:, 0],
                grid_all[:, 1],
                s=4,
                color="gray",
                label="grid",
                alpha=0.5,
                marker="+",
            )

            ax.scatter(
                grid_all_noise[:, 0],
                grid_all_noise[:, 1],
                s=6,
                color="brown",
                label="grid_noise",
                alpha=0.9,
                marker="+",
            )
            ax.set_title("grid with noise")

        else:
            grid_all_noise = grid_all

        # INIT GAUSSIANS
        fig = plt.figure()
        ax = fig.add_subplot(111)

        show_nr_points = np.minimum(1000, obs_train.shape[0])
        ax.set_title("Continuous obs. + INIT gaussians")
        ax.scatter(
            grid_all[:, 0],
            grid_all[:, 1],
            s=4,
            color="gray",
            label="grid",
            alpha=0.5,
            marker="+",
        )
        ax.scatter(
            obs_train[:show_nr_points, 0],
            obs_train[:show_nr_points, 1],
            s=6,
            color="red",
            label="cont. obs",
        )
        # ax.scatter(obs_train_grid[:show_nr_points, 0], obs_train_grid[:show_nr_points, 1], s=6, color='green',
        #            label="discr. obs", alpha=0.3)

        for i, (mean, chol_param) in enumerate(
            zip(means1d_hat_init_2d, cholesky_L_params_init_2d)
        ):
            Cholesky_L = torch.zeros((2, 2))
            Cholesky_L[0, 0] = chol_param[0]
            Cholesky_L[1, 1] = chol_param[1]
            Cholesky_L[0, 1] = chol_param[2]
            Cholesky_L[1, 0] = chol_param[2]

            cov_matrix = torch.matmul(Cholesky_L, Cholesky_L.T)

            draw_ellipse(
                mean.detach().cpu().numpy(), cov_matrix.detach().cpu().numpy(), ax, alpha=0.4
            )

        # FITTED GAUSSIANS HMMLEARN to orig. cont obs.
        fig = plt.figure()
        ax = fig.add_subplot(111)

        show_nr_points = np.minimum(1000, obs_train.shape[0])
        ax.set_title("Continuous obs. + fitted gaussians HMMLEARN to orig. cont. obs")
        ax.scatter(
            grid_all[:, 0],
            grid_all[:, 1],
            s=4,
            color="gray",
            label="grid",
            alpha=0.5,
            marker="+",
        )
        ax.scatter(
            obs_train[:show_nr_points, 0],
            obs_train[:show_nr_points, 1],
            s=6,
            color="red",
            label="cont. obs",
        )

        for i, (mean, cov_matrix) in enumerate(
            zip(
                model_hmmlearn_gaussian_trained.means_,
                model_hmmlearn_gaussian_trained.covars_,
            )
        ):
            draw_ellipse(mean, cov_matrix, ax, alpha=0.4)

            print("i = ", i)
            print("mean = ", mean)
            print("cov matrix = ", cov_matrix)

        # FITTED GAUSSIANS HMMLEARN to DISCRETIZED obs.
        fig = plt.figure()
        ax = fig.add_subplot(111)

        show_nr_points = np.minimum(1000, obs_train.shape[0])
        ax.set_title("Continuous obs. + fitted gaussians HMMLEARN to DISCRETIZED  obs.")
        ax.scatter(
            grid_all[:, 0],
            grid_all[:, 1],
            s=4,
            color="gray",
            label="grid",
            alpha=0.5,
            marker="+",
        )
        ax.scatter(
            obs_train[:show_nr_points, 0],
            obs_train[:show_nr_points, 1],
            s=6,
            color="red",
            label="cont. obs",
        )

        for i, (mean, cov_matrix) in enumerate(
            zip(
                model_hmmlearn_gaussian_grid_trained.means_,
                model_hmmlearn_gaussian_grid_trained.covars_,
            )
        ):
            draw_ellipse(mean, cov_matrix, ax, alpha=0.4)

            print("i = ", i)
            print("mean = ", mean)
            print("cov matrix = ", cov_matrix)

        # FITTED GAUSSIANS TORCH
        fig = plt.figure()
        ax = fig.add_subplot(111)

        show_nr_points = np.minimum(1000, obs_train.shape[0])
        ax.set_title(
            "Continuous obs. + fitted gaussians TORCH, loss = " + str(loss_type)
        )
        ax.scatter(
            grid_all[:, 0],
            grid_all[:, 1],
            s=14,
            color="gray",
            label="grid",
            alpha=0.5,
            marker="+",
        )
        ax.scatter(
            obs_train[:show_nr_points, 0],
            obs_train[:show_nr_points, 1],
            s=6,
            color="red",
            label="cont. obs",
        )
        # ax.scatter(obs_train_grid[:show_nr_points, 0], obs_train_grid[:show_nr_points, 1], s=6, color='green',
        #             label="discr. obs", alpha=0.3)
        ax.legend()

        # ax.scatter(obs_train_grid[:show_nr_points, 0], obs_train_grid[:show_nr_points, 1], s=6, color='green',
        #            label="discr. obs", alpha=0.3)

        for i, (mean, chol_param) in enumerate(
            zip(trained_means, trained_cholesky_L_params)
        ):
            Cholesky_L = torch.zeros((2, 2))
            Cholesky_L[0, 0] = chol_param[0]
            Cholesky_L[1, 1] = chol_param[1]
            Cholesky_L[0, 1] = chol_param[2]
            Cholesky_L[1, 0] = chol_param[2]

            cov_matrix = torch.matmul(Cholesky_L, Cholesky_L.T)

            draw_ellipse(
                mean.detach().cpu().numpy(), cov_matrix.detach().cpu().numpy(), ax, alpha=0.4
            )

            print("i = ", i)
            print("mean = ", mean)
            print("cov matrix = ", cov_matrix)

    plt.show()
    print("done")
    quit()

    #
    # H, bins = np.histogram(obs_train, bins=m)
    # plt.figure()
    # plt.bar(bins[:-1], H)
    # plt.title("Histogram of observations")
    #
    # print(colored("EXAMPLE  = " + EXAMPLE, "red"))
    #
    # L2 = L
    #
    # # hmmlearn GaussianHMM
    # model1D_hmmlearn_gaussian_trained = GaussianHMM(
    #     n_components=L2, covariance_type="full"
    # )
    # model1D_hmmlearn_gaussian_trained.fit(obs_train.reshape(-1, 1))
    # logprob_hmmlearn_gaussian_trained = model1D_hmmlearn_gaussian_trained.score(
    #     obs_test.reshape(-1, 1)
    # )
    #
    # means_trained_GaussianHMM = model1D_hmmlearn_gaussian_trained.means_.reshape(-1)
    # covs_trained_GaussianHMM = model1D_hmmlearn_gaussian_trained.covars_.reshape(-1)
    #
    # B_large_GaussianHMM = np.zeros((L2, m_large))
    #
    # for i in np.arange(L2):
    #     B_large_GaussianHMM[i, :] = np.array(
    #         [
    #             scipy.stats.norm.pdf(
    #                 x,
    #                 means_trained_GaussianHMM[i].reshape(-1),
    #                 np.sqrt(covs_trained_GaussianHMM[i].reshape(-1)),
    #             )
    #             for x in grid_large
    #         ]
    #     ).reshape(-1)
    #
    # A_gauss = model1D_hmmlearn_gaussian_trained.transmat_
    # mu_gauss = compute_stat_distr(A_gauss)
    # S_gauss = torch.tensor(np.dot(np.diag(mu_gauss), A_gauss)).to(device)
    #
    # means1d_hat_init = torch.nn.Parameter(
    #     torch.rand(L2) * (np.max(grid) - np.min(grid))
    # ).to(device)
    # print("means1d_hat_init=", means1d_hat_init)
    #
    # Shat_un_init = torch.nn.Parameter(torch.randn(L2, L2)).to(
    #     device
    # )  # un = unnormalized
    # covs1d_hat_un_init = torch.nn.Parameter(torch.randn(L2) / 2).to(device)
    #
    # model_hmm_nmf_torch = HMM_NMF(
    #     Shat_un_init=Shat_un_init,
    #     means1d_hat_init=means1d_hat_init,
    #     covs1d_hat_un_init=covs1d_hat_un_init,
    #     m=m,
    # )
    #
    # P_torch_init_large = model_hmm_nmf_torch.compute_P_torch(
    #     torch.tensor(grid_large).to(device), normalize=False
    # )
    #
    # print(colored("Fitting model_hmm_nmf_torch ... ", "red"))
    # model_hmm_nmf_torch.fit(
    #     torch.Tensor(grid).to(device),
    #     obs_train_grid_labels.reshape(-1),
    #     lr=lrate,
    #     nr_epochs=nr_epochs_torch,
    # )
    # print(colored("DONE (fitting model_hmm_nmf_torch) ", "red"))
    #
    # P_torch_trained_large = model_hmm_nmf_torch.compute_P_torch(
    #     torch.tensor(grid_large).to(device), normalize=False
    # )
    #
    # Shat_un_init = torch.nn.Parameter(torch.ones(L2, L2)).to(device)
    #
    # # mu_temp = model1D_hmmlearn_gaussian_trained.startprob_
    # # A_temp = model1D_hmmlearn_gaussian_trained.transmat_
    # # Shat_un_init = torch.tensor(np.dot(np.diag(mu_temp), A_temp)).float().to(device)
    # # Shat_un_init = torch.log(0.0001 + Shat_un_init)
    #
    # # model_hmm_nmf_torch_flow = HMM_NMF_FLOW(Shat_un_init=Shat_un_init, m=m,
    # #                                         params=args,
    # #                                         init_params=(model1D_hmmlearn_gaussian_trained.means_,
    # #                                                      model1D_hmmlearn_gaussian_trained.covars_))
    #
    # if args.pretrain_flow:
    #     model_hmm_nmf_torch_flow = HMM_NMF_FLOW(
    #         Shat_un_init=model_hmm_nmf_torch.Shat_un,
    #         m=m,
    #         params=args,
    #         init_params=(
    #             model_hmm_nmf_torch.get_means1d(),
    #             model_hmm_nmf_torch.get_covs1d(),
    #         ),
    #     )
    # else:
    #     model_hmm_nmf_torch_flow = HMM_NMF_FLOW(
    #         Shat_un_init=Shat_un_init, m=m, params=args
    #     )
    #
    # print(colored("Fitting model_hmm_nmf_torch_flow ... ", "red"))
    # # model_hmm_nmf_torch_flow.train()
    # model_hmm_nmf_torch_flow.fit(
    #     torch.Tensor(grid).to(device),
    #     obs_train_grid_labels.reshape(-1),
    #     lr=lrate,
    #     nr_epochs=nr_epochs,
    #     display_info_every_step=1,
    # )
    # model_hmm_nmf_torch_flow.eval()
    # print(colored("DONE (fitting model_hmm_nmf_torch_flow) ", "red"))
    #
    # P_torch_flow_trained_large = model_hmm_nmf_torch_flow.compute_P_torch(
    #     torch.tensor(grid_large).to(device), normalize=False
    # )
    #
    # P_torch_flow_trained_large_norm = model_hmm_nmf_torch_flow.compute_P_torch(
    #     torch.tensor(grid_large).to(device), normalize=True
    # )
    #
    # if show_plots:
    #     plt.figure()
    #     show_steps = min(1000, len(obs_train))
    #     plt.title("Continuous observations")
    #     plt.scatter(
    #         np.arange(show_steps), obs_train[:show_steps], s=3, label="cont. obs"
    #     )
    #
    #     plt.figure()
    #     plt.title("Discretized observations")
    #     for y in grid:
    #         plt.plot([0, show_steps - 1], [y, y], color="gray", alpha=0.2)
    #
    #     plt.scatter(
    #         np.arange(show_steps),
    #         obs_train_grid[:show_steps],
    #         s=3,
    #         label="discrete obs",
    #     )
    #     plt.legend()
    #
    # if DATA_TYPE == "REAL":
    #     show_distrib(
    #         P_torch_trained_large.T.cpu().detach().numpy(),
    #         B_large_GaussianHMM,
    #         P1_text="P_torch_trained_large",
    #         P2_text="B_large_GaussianHMM",
    #         show_points=False,
    #         grid_large=grid_large,
    #         grid=grid,
    #         show_both_on_rhs=True,
    #     )
    #
    #     show_distrib(
    #         P_torch_flow_trained_large.T.cpu().detach().numpy(),
    #         P_torch_flow_trained_large.T.cpu().detach().numpy(),
    #         P1_text="P_torch_flow_trained_large",
    #         P2_text="P_torch_flow_trained_large",
    #         show_points=False,
    #         grid_large=grid_large,
    #         grid=grid,
    #         show_both_on_rhs=True,
    #     )
    #
    # if DATA_TYPE != "REAL":
    #     MAD_gauss = compute_MAD(
    #         S_orig.to(device),
    #         torch.tensor(B_orig_large).to(device),
    #         S_gauss,
    #         torch.tensor(B_large_GaussianHMM.T).to(device),
    #     )
    #     MAD_torch = compute_MAD(
    #         S_orig.to(device),
    #         torch.tensor(B_orig_large).to(device),
    #         model_hmm_nmf_torch.get_S(),
    #         P_torch_trained_large,
    #     )
    #     MAD_torch_flow = compute_MAD(
    #         S_orig.to(device),
    #         torch.tensor(B_orig_large).to(device),
    #         model_hmm_nmf_torch_flow.get_S(),
    #         P_torch_flow_trained_large,
    #     )
    #
    #     print(colored("MADs:", color="red"))
    #     print(
    #         "MAD_gauss = ",
    #         MAD_gauss,
    #         ",\tMAD_torch = ",
    #         MAD_torch,
    #         ",\tMAD_torch_flow = ",
    #         MAD_torch_flow,
    #     )
    #     print("\n")
    #
    #     # print("grid = ", grid)
    #     try:
    #         if show_plots:
    #             # plt.figure()
    #             # show_steps = min(400, len(obs_train))
    #             # plt.title("Continuous observations")
    #             # plt.scatter(np.arange(show_steps), obs_train[:show_steps], s=3, label="cont. obs")
    #             #
    #             # plt.figure()
    #             # plt.title("Discretized observations")
    #             # for y in grid:
    #             #     plt.plot([0, show_steps - 1], [y, y], color="gray", alpha=0.2)
    #             #
    #             # plt.scatter(np.arange(show_steps), obs_train_grid[:show_steps], s=3, label='discrete obs')
    #             # plt.legend()
    #             show_distrib(
    #                 B_orig_large,
    #                 B_large_GaussianHMM,
    #                 P1_text="B_orig_large",
    #                 P2_text="B_large_GaussianHMM",
    #                 show_points=False,
    #                 grid_large=grid_large,
    #                 grid=grid,
    #                 show_both_on_rhs=True,
    #             )
    #
    #             show_distrib(
    #                 B_orig_large,
    #                 P_torch_trained_large.T.cpu().detach().numpy(),
    #                 P1_text="B_orig_large",
    #                 P2_text="P_torch_trained_large",
    #                 show_points=False,
    #                 grid_large=grid_large,
    #                 grid=grid,
    #                 show_both_on_rhs=True,
    #             )
    #             show_distrib(
    #                 P_torch_trained_large.T.cpu().detach().numpy(),
    #                 B_large_GaussianHMM,
    #                 P1_text="P_torch_trained_large",
    #                 P2_text="B_large_GaussianHMM",
    #                 show_points=False,
    #                 show_both_on_rhs=True,
    #                 grid_large=grid_large,
    #                 grid=grid,
    #             )
    #
    #             show_distrib(
    #                 B_orig_large,
    #                 P_torch_flow_trained_large.T.cpu().detach().numpy(),
    #                 P1_text="B_orig_large",
    #                 P2_text="P_torch_flow_trained_large",
    #                 show_points=False,
    #                 grid_large=grid_large,
    #                 grid=grid,
    #                 show_both_on_rhs=True,
    #             )
    #     except ValueError as e:
    #         print("Received ValueError: ", e)
    #
    #     if L2 == len(example_config.hidden_states_distributions):
    #         total_vars_GaussianHMM_trained = compute_total_var_dist(
    #             B_orig_large, B_large_GaussianHMM, grid_large
    #         )
    #         print(
    #             colored(
    #                 "TOTAL VARIATION between B_orig_large AND B_large_GaussianHMM:",
    #                 "red",
    #             )
    #         )
    #         for i, tv in enumerate(total_vars_GaussianHMM_trained):
    #             print("ROW i=", i, ", \t TOTAL VAR = ", tv)
    #         print("MEAN TOTAL VAR = ", np.mean(total_vars_GaussianHMM_trained))
    #         polyaxon.tracking.log_outputs(
    #             total_vars_GaussianHMM_trained=np.mean(
    #                 total_vars_GaussianHMM_trained
    #             ).item()
    #         )
    #
    #         total_vars_torch_trained = compute_total_var_dist(
    #             B_orig_large, P_torch_trained_large.T.cpu().detach().numpy(), grid_large
    #         )
    #         print(
    #             colored(
    #                 "TOTAL VARIATION between  B_orig_large AND P_torch_trained_large:",
    #                 "red",
    #             )
    #         )
    #         for i, tv in enumerate(total_vars_torch_trained):
    #             print("ROW i=", i, ", \t TOTAL VAR = ", tv)
    #         print("MEAN TOTAL VAR = ", np.mean(total_vars_torch_trained))
    #
    #         total_vars_torch_flow_trained = compute_total_var_dist(
    #             B_orig_large,
    #             P_torch_flow_trained_large.T.cpu().detach().numpy(),
    #             grid_large,
    #         )
    #         print(
    #             colored(
    #                 "TOTAL VARIATION between  B_orig_large AND P_torch_flor_trained_large:",
    #                 "red",
    #             )
    #         )
    #         for i, tv in enumerate(total_vars_torch_flow_trained):
    #             print("ROW i=", i, ", \t TOTAL VAR = ", tv)
    #         print("MEAN TOTAL VAR = ", np.mean(total_vars_torch_flow_trained))
    #         polyaxon.tracking.log_outputs(
    #             total_vars_torch_flow_trained=np.mean(
    #                 total_vars_torch_flow_trained
    #             ).item()
    #         )
    #     else:
    #         total_vars_GaussianHMM_trained = -1
    #         total_vars_torch_trained = -1
    #         total_vars_torch_flow_trained = -1
    #
    #     print("\nn=", n)
    #
    # # P_torch_trained_large = model_hmm_nmf_torch
    # print("Computing model_hmm_nmf_torch.continuous_score(obs_test) .... ")
    # logprob_torch_trained_continuous1 = model_hmm_nmf_torch.continuous_score(obs_test)
    # # logprob_torch_trained_continuous2 = model_hmm_nmf_torch.continuous_score(obs_test_grid_labels)
    #
    # print("Computing model_hmm_nmf_torch_flow.continuous_score(obs_test) .... ")
    # logprob_flow_trained_continuous = model_hmm_nmf_torch_flow.continuous_score(
    #     obs_test
    # )
    #
    # # powyzsze troche trwa, co ciekawe, ponizsze (na danych "zgridowanych" do integerow) jest duzo szybsze
    # # logprob_flow_trained_continuous2 = model_hmm_nmf_torch_flow.continuous_score(obs_test_grid_labels)
    #
    # print("logprob_hmmlearn_gaussian_trained =\t\t", logprob_hmmlearn_gaussian_trained)
    # print("logprob_torch_trained_continuous1= \t\t", logprob_torch_trained_continuous1)
    # # print("logprob_torch_trained_continuous2= \t\t", logprob_torch_trained_continuous2)
    # print("logprob_flow_trained_continuous1= \t\t", logprob_flow_trained_continuous)
    #
    # log_prob_results = dict(
    #     hmmlearn=logprob_hmmlearn_gaussian_trained,
    #     cont=logprob_torch_trained_continuous1.item(),
    #     cflow=logprob_flow_trained_continuous.item(),
    # )
    # polyaxon.tracking.log_outputs(
    #     better_flow=1
    #     if (log_prob_results["cflow"] > log_prob_results["hmmlearn"])
    #     else 0
    # )
    # polyaxon.tracking.log_outputs(
    #     better_torch=1
    #     if (log_prob_results["cont"] > log_prob_results["hmmlearn"])
    #     else 0
    # )
    # polyaxon.tracking.log_metrics(
    #     **{"logprob_" + key: val for key, val in log_prob_results.items()}
    # )
    # polyaxon.tracking.log_outputs(
    #     **{"o_" + key: val for key, val in log_prob_results.items()}
    # )
    # model_hmm_nmf_torch_flow.save_weights("model_hmm_nmf_torch_flow.pt")
    # polyaxon.tracking.log_model(
    #     "model_hmm_nmf_torch_flow.pt", name="hmm_nmf_flow", framework="torch"
    # )
    # print("Done.")
    # print(log_prob_results)
    # print({"FlowHMM": log_prob_results["cflow"]/n,  "H-Gauss (hmmlearn)": log_prob_results["hmmlearn"]/n})
    #
    # if output_file is not None:
    #     data_to_save = {
    #         "EXAMPLE": EXAMPLE,
    #         "L": L,
    #         "A_orig": A_orig,
    #         "m": m,
    #         "grid_large": grid_large,
    #         "grid": grid,
    #         "B_large_GaussianHMM": B_large_GaussianHMM,
    #         "P_torch_trained_large": P_torch_trained_large.detach().cpu(),
    #         "P_torch_flow_trained_large": P_torch_flow_trained_large.detach().cpu(),
    #         "logprob_hmmlearn_gaussian_trained": logprob_hmmlearn_gaussian_trained,
    #         "logprob_torch_trained_continuous1": logprob_torch_trained_continuous1,
    #         #'logprob_torch_trained_continuous2': logprob_torch_trained_continuous2,
    #         "logprob_flow_trained_continuous": logprob_flow_trained_continuous,
    #         #'logprob_flow_trained_continuous2':logprob_flow_trained_continuous2,
    #         "obs_train": obs_train,
    #         "obs_test": obs_test,
    #         "obs_train_grid": obs_train_grid,
    #         "obs_test_grid": obs_test_grid,
    #         "obs_train_grid_labels": obs_train_grid_labels,
    #         "obs_test_grid_labels": obs_test_grid_labels,
    #         "nr_epochs": nr_epochs,
    #         "nr_epochs_torch": nr_epochs_torch,
    #         "n": n,
    #         "DATA_TYPE": DATA_TYPE,
    #     }
    #     if DATA_TYPE != "REAL":
    #         data_to_save.update(
    #             {
    #                 "B_orig_large": B_orig_large,
    #                 "MAD_gauss": MAD_gauss,
    #                 "MAD_torch": MAD_torch,
    #                 "MAD_torch_flow": MAD_torch_flow,
    #                 "total_vars_GaussianHMM_trained": total_vars_GaussianHMM_trained,
    #                 "total_vars_torch_trained": total_vars_torch_trained,
    #                 "total_vars_torch_flow_trained": total_vars_torch_flow_trained,
    #             }
    #         )
    #     data_outfile = open(output_file, "wb+")
    #     pickle.dump(data_to_save, data_outfile)
    #     print("Data written in: ", output_file)
    #
    # if show_plots:
    #     plt.show()
    #


if __name__ == "__main__":
    main()
