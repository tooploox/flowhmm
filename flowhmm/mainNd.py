import argparse
import os
import pandas as pd
from itertools import permutations
from typing import Optional, Dict

import numpy as np
import polyaxon
import polyaxon.tracking
import scipy.stats
import torch
from hmmlearn.hmm import GaussianHMM, GMMHMM
import wandb
from hmmlearn.hmm import GaussianHMM
from icecream import ic
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from termcolor import colored
from sklearn.datasets import make_moons
from sklearn import metrics

from models.fhmm_2d import HMM_NMF_multivariate, HMM_NMF_FLOW_multivariate
from utils import set_seed, load_example, compute_stat_distr, compute_joint_trans_matrix


# sample usage

# python  flowhmm/main2d.py -e examples/SYNTHETIC_2d_data_2G_1U.yaml --nr_epochs 1000  --show_plots --extra_n 30000 --loss_type kld --lrate 0.01

# python  flowhmm/main2d.py -e examples/SYNTHETIC_2d_data_2G_1U.yaml --nr_epochs_torch 1000  --show_plots --extra_n 30000 --loss_type kld --lrate 0.01

# python -i flowhmm/main2d.py -e examples/SYNTHETIC_2d_data_2G_1U.yaml --nr_epochs_torch 200  --show_plots=yes
# python flowhmm/main2d.py -e examples/SYNTHETIC_2d_data_2G_1U.yaml --nr_epochs_torch 5000 --seed 139
# python flowhmm/main2d.py -e examples/SYNTHETIC_2d_data_2G_1U.yaml --seed 4 --loss_type old  --nr_epochs_torch 3000 --init_with_kmeans True


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
        # default="examples/SYNTHETIC_2d_data_2G_1U.yaml",
        default="examples/SYNTHETIC_2d_data_1G_1U_1GeomBrownianMotion.yaml",
        help="Path to example YAML config file",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Whether to show plots after the training.",
    )
    parser.add_argument(
        "--nr_epochs", default=10, type=int, required=False, help="nr of epochs"
    )
    parser.add_argument("--loss_type", type=str, default="kld", choices=["old", "kld"])

    parser.add_argument(
        "--pretrain_flow",
        action="store_true",
        help="Whether to use pretrained weights for training the flow model.",
    )
    parser.add_argument(
        "--nr_epochs_torch",
        default=10,
        type=int,
        required=False,
        help="nr of epochs for torch",
    )
    parser.add_argument(
        "--init_with_kmeans",
        action="store_true",
        help="Whether to init with kmeans' centers",
    )
    parser.add_argument(
        "--seed", default=1, type=int, required=False, help="default seed"
    )
    parser.add_argument(
        "--training_type", type=str, default="Q_training", choices=["EM", "Q_training"]
    )
    parser.add_argument(
        "--run_name", type=str, default="wandb_run_name"
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
    parser.add_argument("--train_T", action="store_false")
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--noise_var", type=float, default=0.1)
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

    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--rademacher", action="store_true")
    parser.add_argument("--spectral_norm", action="store_true")
    parser.add_argument("--batch_norm", action="store_true")
    parser.add_argument("--bn_lag", type=float, default=0)
    parser.add_argument("--polyaxon", action="store_true")
    parser.add_argument("--extra_n", type=int, required=False)
    parser.add_argument("--extra_L", type=int, required=False)
    parser.add_argument("--max_shape", type=int, default=1000, required=False, help="max number of samples used when training EM")
    parser.add_argument(
        "--use_wandb_logging",
        action="store_true",
        help="Whether to log results to wandb",
    )
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

    moons_boolean = False

    for i in np.arange(len(distributions)):
        if distributions[i]["name"] == "moon1" or distributions[i]["name"] == "moon2":
            moons_boolean = True
            j = i

    if moons_boolean:
        XY, moon_class = make_moons(n_samples=2 * n, noise=0.1)
        XY_moon1 = XY[moon_class == 0]
        XY_moon2 = XY[moon_class == 1]
        moon1_next = 0
        moon2_next = 0

    observations = np.zeros((n, dim))
    hidden_states = np.zeros(n)

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

        if distributions[current_state]["name"] == "moon1":
            observations[k, :] = XY_moon1[moon1_next]
            moon1_next = moon1_next + 1

        if distributions[current_state]["name"] == "moon2":
            observations[k, :] = XY_moon2[moon2_next]
            moon2_next = moon2_next + 1

        if distributions[current_state]["name"] == "gbm":
            params_r = distributions[current_state]["params"]["r"]
            params_sigma = distributions[current_state]["params"]["sigma"]
            params_mu = params_r - params_sigma**2 / 2
            params_S0 = distributions[current_state]["params"]["S0"]
            p_cov_matrix = np.array([[1, 1 / 2], [1 / 2, 1]])
            p_means = np.array([0, 0])
            B05, B1 = np.random.multivariate_normal(p_means, p_cov_matrix)
            S05 = params_S0 * np.exp(params_mu * 0.5 + params_sigma * B05)
            S1 = params_S0 * np.exp(params_mu * 1 + params_sigma * B1)

            observations[k, :] = np.array([[S05, S1]])

        hidden_states[k] = current_state
        current_state = np.random.choice(
            np.arange(n_states), 1, p=transmat[current_state, :].reshape(-1)
        )[0]
    return observations, hidden_states


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


def calculate_accuracy(confusion_matrix: np.ndarray):
    total = np.sum(confusion_matrix)
    n = confusion_matrix.shape[0]
    return max([
        np.sum(np.diag(confusion_matrix[range(n), perm]))
        for perm in permutations(range(n))
    ])/total


def main():
    args = ParseArguments()

    wandb_logging = args.use_wandb_logging

    ic(args)

    polyaxon.tracking.init(is_offline=not args.polyaxon)
    polyaxon.tracking.log_inputs(args=args.__dict__)
    set_seed(args.seed)
    example_config = load_example(args.example_yaml)

    wandb.init(
        mode="online" if wandb_logging else "offline",
        entity="tooploox-ai",
        project="flow-hmm",
        config=args,
        group=args.example_yaml,
        name=args.run_name
    )
    wandb.config["example_config"] = example_config._asdict()

    EXAMPLE, _ = os.path.basename(args.example_yaml).rsplit(".", 1)
    ic(EXAMPLE, example_config)
    polyaxon.tracking.log_inputs(yaml=example_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    show_plots = args.show_plots
    nr_epochs = args.nr_epochs
    nr_epochs_torch = args.nr_epochs_torch
    add_noise = args.add_noise
    noise_var = args.noise_var
    loss_type = args.loss_type
    training_type = args.training_type
    n = example_config.nr_observations
    lrate = float(args.lrate)

    # Debugging
    ic(nr_epochs, nr_epochs_torch, n, device, show_plots, lrate)

    if example_config.dataset:
        polyaxon.tracking.log_inputs(**example_config.dataset)

    obs_train, obs_test, grid_strategy = None, None, None

    # REAL DATASETS
    if example_config.data_type == "real":
        dataset = example_config.dataset
        df = pd.read_csv(dataset["filepath"], **dataset["load_args"])
        ic(dataset, df.columns, df.describe())

        df_train = df[dataset["column"]].values
        polyaxon.tracking.log_dataframe(df, "df_raw")
        # df_train = (df_train - df_train.mean()) / df_train.std()
        polyaxon.tracking.log_dataframe(df, "df_train")
        ic(pd.DataFrame(df_train).describe())
        n = example_config.nr_observations
        n_train = n

        # OLD: n_test = min(n_train, len(df_train) - n_train)
        n_test = len(df_train) - n_train

        print("n_train = ", n_train, ", n_test = ", n_test)
        obs_train = np.array(df_train[:n_train], dtype=np.float)
        obs_test = np.array(df_train[n_train : n_train + n_test], dtype=np.float)
        m = example_config.grid_size
        L = example_config.nr_hidden_states

        obs_mean = np.mean(obs_train, axis=0)
        obs_std = np.std(obs_train, axis=0)
        obs_train = (obs_train - obs_mean)/obs_std
        obs_test = (obs_test - obs_mean)/obs_std

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

        wandb.log({"n": n, "L": L})

        A_orig = np.array(example_config.transition_matrix)
        mu_orig = compute_stat_distr(A_orig)
        S_orig = torch.tensor(np.dot(np.diag(mu_orig), A_orig))

        ic(A_orig, mu_orig, S_orig)

        # SIMULATE OBSERVATIONS:
        obs_train, hidden_states_train = simulate_observations_multivariate(
            n,
            mu=mu_orig,
            transmat=A_orig,
            distributions=example_config.hidden_states_distributions,
        )
        obs_test, hidden_states_test = simulate_observations_multivariate(
            n,
            mu=mu_orig,
            transmat=A_orig,
            distributions=example_config.hidden_states_distributions,
        )

        n_train = n
        n_test = n

        # old:
        # x_min = np.min(obs_train[:, 0]) - 0.05 * np.abs(np.min(obs_train[:, 0]))
        # x_max = np.max(obs_train[:, 0]) + 0.05 * np.abs(np.min(obs_train[:, 0]))
        #
        # y_min = np.min(obs_train[:, 1]) - 0.05 * np.abs(np.min(obs_train[:, 1]))
        # y_max = np.max(obs_train[:, 1]) + 0.05 * np.abs(np.min(obs_train[:, 1]))

        # new:

        ## ADDED: normalization
        obs_mean = np.mean(obs_train, axis=0)
        obs_train = obs_train - obs_mean
        obs_test = obs_test - obs_mean

        x_min = np.min(obs_train[:, 0])
        x_max = np.max(obs_train[:, 0])

        y_min = np.min(obs_train[:, 1])
        y_max = np.max(obs_train[:, 1])

        x_min2 = x_min - np.var(obs_train[:, 0])
        x_max2 = x_max + np.var(obs_train[:, 0])
        y_min2 = y_min - np.var(obs_train[:, 1])
        y_max2 = y_max + np.var(obs_train[:, 1])

        x_min3 = x_min + 0.2 * (x_max - x_min)
        x_max3 = x_max - 0.2 * (x_max - x_min)

        y_min3 = y_min + 0.2 * (y_max - x_min)
        y_max3 = y_max - 0.2 * (y_max - x_min)

        ic(x_min, x_max, y_min, y_max)
        ic(x_min2, x_max2, y_min2, y_max2)
        ic(x_min3, x_max3, y_min3, y_max3)

        grid_strategy = example_config.grid_strategy
        # # grid_strategy = "uniform"
        # # grid_strategy = "kmeans"
        # # grid_strategy = "mixed"
        #
        if grid_strategy == "uniform":
            grid_x = np.linspace(x_min2, x_max2, m)
            grid_y = np.linspace(y_min2, y_max2, m)
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
            #  mm = example_config.grid_size_all
            # tutaj robimy tak, ze osobno na x, osobno na y

            m_k = m
            # m_k = int(m*0.8)
            # m_extra_left = int((m-m_k)/2)
            # m_extra_right = m-m_k-m_extra_left

            kmeans_x = KMeans(n_clusters=m_k)
            kmeans_x.fit(obs_train[:, 0].reshape(-1, 1))
            grid_x = np.sort(kmeans_x.cluster_centers_.reshape(-1))

            # mean_x_dist = np.mean(np.diff(grid_x))
            # extra_x_grid_right = np.arange(np.max(grid_x), np.max(grid_x)+m_extra_right*mean_x_dist, mean_x_dist)
            # grid_x = np.hstack([grid_x,extra_x_grid_right])
            #
            # extra_x_grid_left =np.arange(np.min(grid_x)-m_extra_left*mean_x_dist,np.min(grid_x),mean_x_dist)
            # grid_x = np.hstack([extra_x_grid_left,grid_x])
            #

            kmeans_y = KMeans(n_clusters=m_k)
            kmeans_y.fit(obs_train[:, 1].reshape(-1, 1))
            grid_y = np.sort(kmeans_y.cluster_centers_.reshape(-1))

            # mean_y_dist = np.mean(np.diff(grid_y))
            # extra_y_grid = np.arange(np.max(grid_y), np.max(grid_y) + m_extra_right * mean_y_dist, mean_y_dist)
            # grid_y = np.hstack([grid_y, extra_y_grid])
            #
            # extra_y_grid_left = np.arange(np.min(grid_y) - m_extra_left * mean_y_dist, np.min(grid_y), mean_y_dist)
            # grid_y = np.hstack([extra_y_grid_left, grid_y])

            grid_all = np.transpose(
                [np.tile(grid_x, len(grid_y)), np.repeat(grid_y, len(grid_x))]
            )
            # grid_all=np.zeros((mm,2))
            # grid_all[:,0]=grid_x
            # grid_all[:,1]=grid_y
            mm = grid_all.shape[0]
            print("test")
        elif grid_strategy == "mixed":

            m_half_uniform = int(m / 2)
            m_half_kmeans = m - m_half_uniform

            grid_uniform_x = np.linspace(x_min2, x_max2, m_half_uniform)
            grid_uniform_y = np.linspace(y_min2, y_max2, m_half_uniform)

            kmeans_x = KMeans(n_clusters=m_half_kmeans)
            kmeans_x.fit(obs_train[:, 0].reshape(-1, 1))
            grid_kmeans_x = np.sort(kmeans_x.cluster_centers_.reshape(-1))

            kmeans_y = KMeans(n_clusters=m_half_kmeans)
            kmeans_y.fit(obs_train[:, 1].reshape(-1, 1))
            grid_kmeans_y = np.sort(kmeans_y.cluster_centers_.reshape(-1))

            grid_x = np.sort(np.hstack([grid_uniform_x, grid_kmeans_x]))
            grid_y = np.sort(np.hstack([grid_uniform_y, grid_kmeans_y]))

            grid_all = np.transpose(
                [np.tile(grid_x, len(grid_y)), np.repeat(grid_y, len(grid_x))]
            )
            # grid_all=np.zeros((mm,2))
            # grid_all[:,0]=grid_x
            # grid_all[:,1]=grid_y
            mm = grid_all.shape[0]

            # half from kmeans, half from uniform

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

        wandb.log(
            {
                "n": n,
                "T": n,
                "L": L,
                "m": m
            })

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

    ## GMMHMM START
    n_mix_list = [1, 5, 10, 20]

    model_hmmlearn_gmmhmm_trained_models = [
        GMMHMM(n_components=L, covariance_type="full", n_mix=k) for k in n_mix_list
    ]

    logprob_hmmlearn_gmmhmm_trained_models = np.zeros(len(n_mix_list))

    model_hmmlearn_gaussian_trained_test = GaussianHMM(
        n_components=L, covariance_type="full"
    )

    model_hmmlearn_gaussian_trained_test.fit(obs_train)
    logprob_hmmlearn_gaussian_trained_models = model_hmmlearn_gaussian_trained_test.score(obs_test)
    for i in np.arange(len(n_mix_list)):
        model_hmmlearn_gmmhmm_trained_models[i].fit(obs_train)
        logprob_hmmlearn_gmmhmm_trained_models[
            i
        ] = model_hmmlearn_gmmhmm_trained_models[i].score(obs_test)

    Shat_un_init = torch.nn.Parameter(torch.ones(L, L)).to(device)

    # if args.pretrain_flow:
    #     model_hmm_nmf_torch_flow_multivariate = HMM_NMF_FLOW_multivariate(
    #         Shat_un_init=model_hmm_nmf_torch_multivariate.Shat_un,
    #         m=m,
    #         mm=mm,
    #         params=args,
    #         dim=grid_all.shape[1],
    #         init_params=(
    #             model_hmm_nmf_torch_multivariate.get_means1d(),
    #             model_hmm_nmf_torch_multivariate.get_covs1d(),
    #         ),
    #     )
    # else:
    model_hmm_nmf_torch_flow_multivariate = HMM_NMF_FLOW_multivariate(
        Shat_un_init=Shat_un_init, m=1, mm=1, dim=obs_train.shape[1], params=args
    )

    model_hmm_nmf_torch_flow_multivariate.fit_EM(
        torch.tensor(obs_train, device=device),
        lr=lrate,
        nr_epochs=nr_epochs,
        display_info_every_step=1,
    )
    model_hmm_nmf_torch_flow_multivariate.eval()

    logprob_flow_test_continuous = (
        model_hmm_nmf_torch_flow_multivariate.continuous_score(obs_test)
    )

    print(
        "logprob_hmmlearn_gaussian_trained_models =\t\t",
        logprob_hmmlearn_gaussian_trained_models / obs_test.shape[0],
    )
    wandb.log({"G": logprob_hmmlearn_gaussian_trained_models / obs_test.shape[0]})

    for i in np.arange(len(n_mix_list)):
        print(
            "logprob_hmmlearn_gmmhmm_trained_models " + str(n_mix_list[i]) + "=\t\t",
            logprob_hmmlearn_gmmhmm_trained_models[i] / obs_test.shape[0],
        )
        wandb.log({f"G{str(n_mix_list[i])}": logprob_hmmlearn_gmmhmm_trained_models[i] / obs_test.shape[0]})

    print(
        "logprob_flow_test_continuous =\t\t",
        logprob_flow_test_continuous / obs_test.shape[0],
    )
    wandb.log({"F": logprob_flow_test_continuous / obs_test.shape[0]})


if __name__ == "__main__":
    main()
