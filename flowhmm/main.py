import argparse
import os
import pickle
from dataclasses import asdict
from typing import Optional, Dict

import numpy as np
import pandas as pd
import polyaxon
import polyaxon.tracking
import scipy.stats
import torch
import wandb
from hmmlearn.hmm import GaussianHMM, GMMHMM
from icecream import ic
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from termcolor import colored

from models.fhmm import HMM_NMF, HMM_NMF_FLOW
from models.fhmm import show_distrib, compute_total_var_dist, compute_MAD
from utils import (
    set_seed,
    load_example,
    compute_stat_distr,
    compute_density_in_grid,
    compute_joint_trans_matrix,
)

from sklearn import metrics


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
        default="examples/SYNTHETIC_1B_1U_1G.yaml",
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
        "--training_type", type=str, default="Q_training", choices=["EM", "Q_training"]
    )
    parser.add_argument("--pretrain_flow", type=eval, default=False)

    parser.add_argument(
        "--nr_epochs_torch",
        default=10,
        type=int,
        required=False,
        help="nr of epochs for torch",
    )
    parser.add_argument(
        "--seed",
        default=117,
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

    parser.add_argument(
        "--n_mix", type=int, default=2, help="only for GMMHMM: number of mixtures"
    )

    parser.add_argument("--add_noise", type=eval, default=False, choices=[True, False])
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

    parser.add_argument("--residual", type=eval, default=False, choices=[True, False])
    parser.add_argument("--rademacher", type=eval, default=False, choices=[True, False])
    parser.add_argument(
        "--spectral_norm", type=eval, default=False, choices=[True, False]
    )
    parser.add_argument("--batch_norm", type=eval, default=True, choices=[True, False])
    parser.add_argument("--bn_lag", type=float, default=0)
    parser.add_argument("--polyaxon", type=bool, default=False)
    parser.add_argument("--extra_n", type=int, required=False)
    parser.add_argument("--extra_L", type=int, required=False)

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
    hidden_states = np.zeros(n)

    n_states = len(distributions)
    current_state = np.random.choice(np.arange(n_states), size=1, p=mu.reshape(-1))[0]
    for k in np.arange(n):
        observations[k] = sample(**distributions[current_state])
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
        name=args.example_yaml
    )
    wandb.config["example_config"] = example_config._asdict()

    EXAMPLE, _ = os.path.basename(args.example_yaml).rsplit(".", 1)
    ic(EXAMPLE, example_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    show_plots = args.show_plots
    nr_epochs = args.nr_epochs
    nr_epochs_torch = args.nr_epochs_torch
    n_mix = args.n_mix

    lrate = float(args.lrate)
    training_type = args.training_type
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

    DATA_TYPE = example_config.data_type.upper()
    polyaxon.tracking.log_inputs(yaml=example_config)
    if example_config.dataset:
        polyaxon.tracking.log_inputs(**example_config.dataset)
    # REAL DATASETS
    if example_config.data_type == "real":
        dataset = example_config.dataset
        A_orig = None
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

        x_min = np.min(obs_train) - 0.05 * np.abs(np.min(obs_train))
        x_max = np.max(obs_train) + 0.05 * np.abs(np.max(obs_train))

        grid_strategy = example_config.grid_strategy
        # grid_strategy = "uniform"
        # grid_strategy = "kmeans"
        # grid_strategy = "mixed"

        if grid_strategy == "uniform":
            grid = np.linspace(x_min, x_max, m)

        if grid_strategy == "kmeans":
            kmeans = KMeans(n_clusters=m)
            kmeans.fit(obs_train.reshape(-1, 1))
            grid = kmeans.cluster_centers_.reshape(-1)

        if grid_strategy == "mixed":
            kmeans = KMeans(n_clusters=int(np.floor(m / 2)))
            kmeans.fit(obs_train.reshape(-1, 1))
            grid_kmeans = kmeans.cluster_centers_.reshape(-1)
            grid_unif = np.linspace(x_min, x_max, m - int(np.floor(m / 2)))

            grid = np.sort(np.concatenate((grid_kmeans, grid_unif)))

        ic(grid)

        grid_labels = list(range(m))
        grid_large = np.linspace(np.min(grid), np.max(grid), m * 10)
        # plt.show()

        m_large = len(grid_large)

        grid_labels = list(range(m))

        knn = KNeighborsClassifier(n_neighbors=1)

        knn.fit(grid.reshape(-1, 1), grid_labels)

        obs_train_grid_labels = (
            knn.predict(obs_train.reshape(-1, 1)).reshape(-1, 1).astype(int)
        )
        obs_test_grid_labels = (
            knn.predict(obs_test.reshape(-1, 1)).reshape(-1, 1).astype(int)
        )

        obs_train_grid = grid[obs_train_grid_labels]
        obs_test_grid = grid[obs_test_grid_labels]

        generated_bool = False
    # SYNTHETIC DATASETS
    elif example_config.data_type == "synthetic":

        L = (
            example_config.nr_hidden_states
            or len(example_config.hidden_states_distributions)
        )

        m = example_config.grid_size
        n = example_config.nr_observations

        if args.extra_n:
            n = args.extra_n
        if args.extra_L:
            L = args.extra_L

        A_orig = np.array(example_config.transition_matrix)
        mu_orig = compute_stat_distr(A_orig)
        S_orig = torch.tensor(np.dot(np.diag(mu_orig), A_orig))

        # SIMULATE OBSERVATIONS:
        obs_train, hidden_states_train = simulate_observations(
            n,
            mu=mu_orig,
            transmat=A_orig,
            distributions=example_config.hidden_states_distributions,
        )
        obs_test, hidden_states_test = simulate_observations(
            n,
            mu=mu_orig,
            transmat=A_orig,
            distributions=example_config.hidden_states_distributions,
        )

        # in all our examples we take n_train=n_test=n
        n_train = n
        n_test = n

        x_min = np.min(obs_train) - 0.05 * np.abs(np.min(obs_train))
        x_max = np.max(obs_train) + 0.05 * np.abs(np.max(obs_train))

        grid_strategy = example_config.grid_strategy
        # grid_strategy = "uniform"
        # grid_strategy = "kmeans"
        # grid_strategy = "mixed"

        if grid_strategy == "uniform":
            grid = np.linspace(x_min, x_max, m)
        elif grid_strategy == "kmeans":
            kmeans = KMeans(n_clusters=m)
            kmeans.fit(obs_train.reshape(-1, 1))
            grid = kmeans.cluster_centers_.reshape(-1)
        elif grid_strategy == "mixed":
            kmeans = KMeans(n_clusters=int(np.floor(m / 2)))
            kmeans.fit(obs_train.reshape(-1, 1))
            grid_kmeans = kmeans.cluster_centers_.reshape(-1)
            grid_unif = np.linspace(x_min, x_max, m - int(np.floor(m / 2)))

            grid = np.sort(np.concatenate((grid_kmeans, grid_unif)))
        elif grid_strategy == "arange_0_max":
            grid = np.arange(0, np.ceil(x_max) + 2)
            m = len(grid)
        else:
            raise ValueError(f"Unknown grid_strategy={grid_strategy}")

        ic(grid, n, L, m)
        polyaxon.tracking.log_inputs(n=n)
        polyaxon.tracking.log_inputs(L=L)
        polyaxon.tracking.log_inputs(m=m)

        wandb.log(
            {
                "n": n,
                "T": n,
                "L": L,
                "m": m
            })

        grid_labels = list(range(m))
        grid_large = np.linspace(np.min(grid), np.max(grid), m * 10)

        m_large = len(grid_large)

        grid_labels = list(range(m))

        knn = KNeighborsClassifier(n_neighbors=1)

        knn.fit(grid.reshape(-1, 1), grid_labels)

        obs_train_grid_labels = (
            knn.predict(obs_train.reshape(-1, 1)).reshape(-1, 1).astype(int)
        )
        obs_test_grid_labels = (
            knn.predict(obs_test.reshape(-1, 1)).reshape(-1, 1).astype(int)
        )

        obs_train_grid = grid[obs_train_grid_labels]
        obs_test_grid = grid[obs_test_grid_labels]

        B_orig_large = compute_pdf_matrix(
            example_config.hidden_states_distributions, grid_large
        )

        generated_bool = False
    else:
        raise ValueError(f"Unknown example.data_type {example_config.data_type}")

    polyaxon.tracking.log_outputs(m=m, L=L, grid_strategy=grid_strategy)

    ################
    # INFER START
    ################

    H, bins = np.histogram(obs_train, bins=m)
    plt.figure()
    plt.bar(bins[:-1], H)
    plt.title("Histogram of observations")

    print(colored("EXAMPLE  = " + EXAMPLE, "yellow"))

    L2 = L

    # hmmlearn GaussianHMM
    # L2 hidden states, each state is a mixture of n_mix components

    # model1D_hmmlearn_gmmhmm_trained -- this will be with n_mix of mixtures
    model1D_hmmlearn_gmmhmm_trained = GMMHMM(
        n_components=L2, covariance_type="full", n_mix=n_mix
    )

    model1D_hmmlearn_gmmhmm_trained.fit(obs_train.reshape(-1, 1))

    logprob_hmmlearn_gmmhmm_trained = model1D_hmmlearn_gmmhmm_trained.score(
        obs_test.reshape(-1, 1)
    )

    # separately we will make models with 2,5,10,15 mixtures, only to compute logprobs
    # n_mix_list = [1, 2, 5, 10, 15, 20]
    n_mix_list = [1, 5, 10, 20]

    model1D_hmmlearn_gmmhmm_trained_models = [
        GMMHMM(n_components=L2, covariance_type="full", n_mix=k) for k in n_mix_list
    ]
    logprob_hmmlearn_gmmhmm_trained_models = np.zeros(len(n_mix_list))
    total_vars_means_hmmlearn_gmmhmm_trained_models = np.zeros(len(n_mix_list))

    for i in np.arange(len(n_mix_list)):
        model1D_hmmlearn_gmmhmm_trained_models[i].fit(obs_train.reshape(-1, 1))
        logprob_hmmlearn_gmmhmm_trained_models[
            i
        ] = model1D_hmmlearn_gmmhmm_trained_models[i].score(obs_test.reshape(-1, 1))

    # hmmlearn GaussianHMM
    model1D_hmmlearn_gaussian_trained = GaussianHMM(
        n_components=L2, covariance_type="full"
    )
    model1D_hmmlearn_gaussian_trained.fit(obs_train.reshape(-1, 1))
    logprob_hmmlearn_gaussian_trained = model1D_hmmlearn_gaussian_trained.score(
        obs_test.reshape(-1, 1)
    )

    means_trained_GaussianHMM = model1D_hmmlearn_gaussian_trained.means_.reshape(-1)
    covs_trained_GaussianHMM = model1D_hmmlearn_gaussian_trained.covars_.reshape(-1)

    # GMMHMM
    B_large_GMMHMM = compute_density_in_grid(
        model1D_hmmlearn_gmmhmm_trained, L2, m_large, grid_large
    )

    B_large_GMMHMM_list = [
        compute_density_in_grid(
            model1D_hmmlearn_gmmhmm_trained_models[i], L2, m_large, grid_large
        )
        for i in np.arange(len(n_mix_list))
    ]

    # GMM
    B_large_GaussianHMM = np.zeros((L2, m_large))

    for i in np.arange(L2):
        B_large_GaussianHMM[i, :] = np.array(
            [
                scipy.stats.norm.pdf(
                    x,
                    means_trained_GaussianHMM[i].reshape(-1),
                    np.sqrt(covs_trained_GaussianHMM[i].reshape(-1)),
                )
                for x in grid_large
            ]
        ).reshape(-1)

    A_gauss = model1D_hmmlearn_gaussian_trained.transmat_
    mu_gauss = compute_stat_distr(A_gauss)
    S_gauss = torch.tensor(np.dot(np.diag(mu_gauss), A_gauss), device=device)

    means1d_hat_init = torch.nn.Parameter(
        torch.rand(L2) * (np.max(grid) - np.min(grid))
    ).to(device)
    print("means1d_hat_init=", means1d_hat_init)

    Shat_un_init = torch.nn.Parameter(torch.randn(L2, L2)).to(
        device
    )  # un = unnormalized
    covs1d_hat_un_init = torch.nn.Parameter(torch.randn(L2) / 2).to(device)

    model_hmm_nmf_torch = HMM_NMF(
        Shat_un_init=Shat_un_init,
        means1d_hat_init=means1d_hat_init,
        covs1d_hat_un_init=covs1d_hat_un_init,
        m=m,
    )

    P_torch_init_large = model_hmm_nmf_torch.compute_P_torch(
        torch.tensor(grid_large, device=device), normalize=False
    )

    print(colored("Fitting model_hmm_nmf_torch ... ", "yellow"))
    model_hmm_nmf_torch.fit(
        torch.tensor(grid, device=device),
        obs_train_grid_labels.reshape(-1),
        lr=lrate,
        nr_epochs=nr_epochs_torch,
    )
    print(colored("DONE (fitting model_hmm_nmf_torch) ", "yellow"))

    P_torch_trained_large = model_hmm_nmf_torch.compute_P_torch(
        torch.tensor(grid_large, device=device), normalize=False
    )

    Shat_un_init = torch.nn.Parameter(torch.ones(L2, L2)).to(device)

    # mu_temp = model1D_hmmlearn_gaussian_trained.startprob_
    # A_temp = model1D_hmmlearn_gaussian_trained.transmat_
    # Shat_un_init = torch.tensor(np.dot(np.diag(mu_temp), A_temp)).float().to(device)
    # Shat_un_init = torch.log(0.0001 + Shat_un_init)

    # model_hmm_nmf_torch_flow = HMM_NMF_FLOW(Shat_un_init=Shat_un_init, m=m,
    #                                         params=args,
    #                                         init_params=(model1D_hmmlearn_gaussian_trained.means_,
    #                                                      model1D_hmmlearn_gaussian_trained.covars_))

    if args.pretrain_flow:
        model_hmm_nmf_torch_flow = HMM_NMF_FLOW(
            Shat_un_init=model_hmm_nmf_torch.Shat_un,
            m=m,
            params=args,
            init_params=(
                model_hmm_nmf_torch.get_means1d(),
                model_hmm_nmf_torch.get_covs1d(),
            ),
        )
    else:
        model_hmm_nmf_torch_flow = HMM_NMF_FLOW(
            Shat_un_init=Shat_un_init, m=m, params=args
        )

    print(colored("Fitting model_hmm_nmf_torch_flow ... ", "yellow"))
    # model_hmm_nmf_torch_flow.train()
    if training_type == "Q_training":
        model_hmm_nmf_torch_flow.fit(
            torch.tensor(grid, device=device),
            obs_train_grid_labels.reshape(-1),
            lr=lrate,
            nr_epochs=nr_epochs,
            display_info_every_step=1,
        )
    if training_type == "EM":
        model_hmm_nmf_torch_flow.fit_EM(
            torch.tensor(obs_train, device=device).float(),
            lr=lrate,
            nr_epochs=nr_epochs,
            display_info_every_step=1,
        )

    model_hmm_nmf_torch_flow.eval()
    print(colored("DONE (fitting model_hmm_nmf_torch_flow) ", "yellow"))

    P_torch_flow_trained_large = model_hmm_nmf_torch_flow.compute_P_torch(
        torch.tensor(grid_large, device=device), normalize=False
    )

    P_torch_flow_trained_large_norm = model_hmm_nmf_torch_flow.compute_P_torch(
        torch.tensor(grid_large, device=device), normalize=True
    )

    if show_plots:
        plt.figure()
        show_steps = min(1000, len(obs_train))
        plt.title("Continuous observations")
        plt.scatter(
            np.arange(show_steps), obs_train[:show_steps], s=3, label="cont. obs"
        )

        plt.figure()
        plt.title("Discretized observations")
        for y in grid:
            plt.plot([0, show_steps - 1], [y, y], color="gray", alpha=0.2)

        plt.scatter(
            np.arange(show_steps),
            obs_train_grid[:show_steps],
            s=3,
            label="discrete obs",
        )
        plt.legend()

    if DATA_TYPE == "REAL":
        show_distrib(
            P_torch_trained_large.T.cpu().detach().numpy(),
            B_large_GaussianHMM,
            P1_text="P_torch_trained_large",
            P2_text="B_large_GaussianHMM",
            show_points=False,
            grid_large=grid_large,
            grid=grid,
            show_both_on_rhs=True,
        )

        show_distrib(
            P_torch_flow_trained_large.T.cpu().detach().numpy(),
            P_torch_flow_trained_large.T.cpu().detach().numpy(),
            P1_text="P_torch_flow_trained_large",
            P2_text="P_torch_flow_trained_large",
            show_points=False,
            grid_large=grid_large,
            grid=grid,
            show_both_on_rhs=True,
        )

        show_distrib(
            P_torch_flow_trained_large.T.cpu().detach().numpy(),
            P_torch_flow_trained_large.T.cpu().detach().numpy(),
            P1_text="P_torch_flow_trained_large",
            P2_text="P_torch_flow_trained_large",
            show_points=False,
            grid_large=grid_large,
            grid=grid,
            show_both_on_rhs=True,
        )

    B_large_GMMHMM_list = [
        compute_density_in_grid(
            model1D_hmmlearn_gmmhmm_trained_models[i], L2, m_large, grid_large
        )
        for i in np.arange(len(n_mix_list))
    ]

    S_GMMHMM_list = [
        compute_joint_trans_matrix(
            torch.tensor(model1D_hmmlearn_gmmhmm_trained_models[i].transmat_, device=device),
            device=device
        )
        for i in np.arange(len(B_large_GMMHMM_list))
    ]

    if DATA_TYPE != "REAL":

        # ACCURACY / CONF. MATRICES:

        model1D_hmmlearn_gaussian_hidden_states_test_predicted = (
            model1D_hmmlearn_gaussian_trained.predict(obs_test.reshape(-1, 1))
        )

        model1D_hmmlearn_gaussian_confusion_matrix = metrics.confusion_matrix(
            hidden_states_test, model1D_hmmlearn_gaussian_hidden_states_test_predicted
        )
        model1D_hmmlearn_gaussian_accuracy = metrics.accuracy_score(
            hidden_states_test, model1D_hmmlearn_gaussian_hidden_states_test_predicted
        )

        GMMHMM_accuracy_scores = [
            metrics.accuracy_score(
                hidden_states_test,
                model1D_hmmlearn_gmmhmm_trained_models[i].predict(
                    obs_test.reshape(-1, 1)
                ),
            )
            for i in np.arange(len(n_mix_list))
        ]

        GMMHMM_conf_matrices = [
            metrics.confusion_matrix(
                hidden_states_test,
                model1D_hmmlearn_gmmhmm_trained_models[i].predict(
                    obs_test.reshape(-1, 1)
                ),
            )
            for i in np.arange(len(n_mix_list))
        ]

        print(colored("ACCURACY (predict hidden states, compare to known ones)"), "red")

        print(
            colored(
                "ACCURACY: hmmlearn_gaussian =\t\t"
                + str(model1D_hmmlearn_gaussian_accuracy),
                "red",
            )
        )

        for nr, i in enumerate(n_mix_list):
            print(
                colored(
                    "ACCURACY: GMM-HMM (hmmlearn), n_mix = "
                    + str(i)
                    + " :"
                    + str(GMMHMM_accuracy_scores[nr]),
                    "red",
                )
            )

        print(colored("CONFUSION MATRICES", "red"))
        print(
            colored("CONF. MATRIX: hmmlear_gaussian", "red"),
            model1D_hmmlearn_gaussian_confusion_matrix,
        )
        for nr, i in enumerate(n_mix_list):
            print(
                colored(
                    "CONF_MATRIX: GMM-HMM (hmmlearn), n_mix = "
                    + str(i)
                    + " :\n"
                    + str(GMMHMM_conf_matrices[nr]),
                    "red",
                )
            )

        MAD_gauss = compute_MAD(
            S_orig.to(device),
            torch.tensor(B_orig_large, device=device),
            S_gauss,
            torch.tensor(B_large_GaussianHMM.T, device=device),
        )

        MADs_GMMHMM_list = [
            compute_MAD(
                S_orig.to(device),
                torch.tensor(B_orig_large, device=device),
                S_GMMHMM_list[i],
                torch.tensor(B_large_GMMHMM_list[i].T, device=device),
            )
            for i in np.arange(len(B_large_GMMHMM_list))
        ]

        #
        # S_GMMHMM_list = [compute_joint_trans_matrix(torch.tensor(model1D_hmmlearn_gmmhmm_trained_models[i].transmat_))
        #                  for i in np.arange(len(B_large_GMMHMM_list))]

        MAD_torch = compute_MAD(
            S_orig.to(device),
            torch.tensor(B_orig_large, device=device),
            model_hmm_nmf_torch.get_S(),
            P_torch_trained_large,
        )
        MAD_torch_flow = compute_MAD(
            S_orig.to(device),
            torch.tensor(B_orig_large, device=device),
            model_hmm_nmf_torch_flow.get_S(),
            P_torch_flow_trained_large,
        )

        print("\n")
        print(colored("NUMBERS TO BE REPORTED (except MADs?):", color="red"))
        print(colored("MADs:", color="yellow"))
        print(
            colored(
                "MAD_gauss = "
                + str(MAD_gauss)
                + ",\tMAD_torch = "
                + str(MAD_torch)
                + str(",\tMAD_torch_flow = ")
                + str(MAD_torch_flow),
                "red",
            )
        )

        for nr, i in enumerate(n_mix_list):
            print(
                colored(
                    "MAD GMM-HMM, n_mix = " + str(i) + " :" + str(MADs_GMMHMM_list[nr]),
                    "red",
                )
            )

        print("\n")

        # print("grid = ", grid)
        try:
            if show_plots:
                # plt.figure()
                # show_steps = min(400, len(obs_train))
                # plt.title("Continuous observations")
                # plt.scatter(np.arange(show_steps), obs_train[:show_steps], s=3, label="cont. obs")
                #
                # plt.figure()
                # plt.title("Discretized observations")
                # for y in grid:
                #     plt.plot([0, show_steps - 1], [y, y], color="gray", alpha=0.2)
                #
                # plt.scatter(np.arange(show_steps), obs_train_grid[:show_steps], s=3, label='discrete obs')
                # plt.legend()
                show_distrib(
                    B_orig_large,
                    B_large_GaussianHMM,
                    P1_text="B_orig_large",
                    P2_text="B_large_GaussianHMM",
                    show_points=False,
                    grid_large=grid_large,
                    grid=grid,
                    show_both_on_rhs=True,
                )

                show_distrib(
                    B_orig_large,
                    B_large_GMMHMM,
                    P1_text="B_orig_large",
                    P2_text="B_large_GMMHMM",
                    show_points=False,
                    grid_large=grid_large,
                    grid=grid,
                    show_both_on_rhs=True,
                )

                show_distrib(
                    B_orig_large,
                    P_torch_trained_large.T.cpu().detach().numpy(),
                    P1_text="B_orig_large",
                    P2_text="P_torch_trained_large",
                    show_points=False,
                    grid_large=grid_large,
                    grid=grid,
                    show_both_on_rhs=True,
                )
                show_distrib(
                    P_torch_trained_large.T.cpu().detach().numpy(),
                    B_large_GaussianHMM,
                    P1_text="P_torch_trained_large",
                    P2_text="B_large_GaussianHMM",
                    show_points=False,
                    show_both_on_rhs=True,
                    grid_large=grid_large,
                    grid=grid,
                )

                show_distrib(
                    B_orig_large,
                    P_torch_flow_trained_large.T.cpu().detach().numpy(),
                    P1_text="B_orig_large",
                    P2_text="P_torch_flow_trained_large",
                    show_points=False,
                    grid_large=grid_large,
                    grid=grid,
                    show_both_on_rhs=True,
                )
        except ValueError as e:
            print("Received ValueError: ", e)

        if L2 == len(example_config.hidden_states_distributions):
            total_vars_GaussianHMM_trained = compute_total_var_dist(
                B_orig_large, B_large_GaussianHMM, grid_large
            )

            total_vars_means_GMMHMM_trained = [
                np.mean(
                    compute_total_var_dist(
                        B_orig_large, B_large_GMMHMM_list[i], grid_large
                    )
                )
                for i in np.arange(len(B_large_GMMHMM_list))
            ]

            print(
                colored(
                    "TOTAL VARIATION between B_orig_large AND B_large_GaussianHMM:",
                    "yellow",
                )
            )
            for i, tv in enumerate(total_vars_GaussianHMM_trained):
                # print(colored("ROW i=" + str(i) + ", \t TOTAL VAR = " + str(tv),"grey"))
                print("ROW i=" + str(i) + ", \t TOTAL VAR = " + str(tv))
            print(
                colored(
                    "MEAN TOTAL VAR (GMM) = "
                    + str(np.mean(total_vars_GaussianHMM_trained)),
                    "red",
                )
            )

            wandb.log({"dtv-G": np.mean(total_vars_GaussianHMM_trained)})

            polyaxon.tracking.log_outputs(
                total_vars_GaussianHMM_trained=np.mean(
                    total_vars_GaussianHMM_trained
                ).item()
            )

            total_vars_torch_trained = compute_total_var_dist(
                B_orig_large, P_torch_trained_large.T.cpu().detach().numpy(), grid_large
            )
            print(
                colored(
                    "TOTAL VARIATION between  B_orig_large AND P_torch_trained_large:",
                    "yellow",
                )
            )
            for i, tv in enumerate(total_vars_torch_trained):
                # print(colored("ROW i=" + str(i) + ", \t TOTAL VAR = " + str(tv), "grey"))
                print("ROW i=" + str(i) + ", \t TOTAL VAR = " + str(tv))
            print("MEAN TOTAL VAR = " + str(np.mean(total_vars_torch_trained)))

            total_vars_torch_flow_trained = compute_total_var_dist(
                B_orig_large,
                P_torch_flow_trained_large.T.cpu().detach().numpy(),
                grid_large,
            )
            print(
                colored(
                    "TOTAL VARIATION between  B_orig_large AND P_torch_flow_trained_large:",
                    "yellow",
                )
            )
            for i, tv in enumerate(total_vars_torch_flow_trained):
                print("ROW i=", i, ", \t TOTAL VAR = ", tv)
            print(
                colored(
                    "MEAN TOTAL VAR (FLOW) = "
                    + str(np.mean(total_vars_torch_flow_trained)),
                    "red",
                )
            )

            wandb.log({"dtv-F": np.mean(total_vars_torch_flow_trained)})

            polyaxon.tracking.log_outputs(
                total_vars_torch_flow_trained=np.mean(
                    total_vars_torch_flow_trained
                ).item()
            )
            print("\n")
            for nr, i in enumerate(n_mix_list):
                print(
                    colored(
                        "MEAN TOTAL VAR GMM-HMM (hmmlearn), n_mix = "
                        + str(i)
                        + " :"
                        + str(total_vars_means_GMMHMM_trained[nr]),
                        "red",
                    )
                )

                wandb.log({f"dtv-G{i}": total_vars_means_GMMHMM_trained[nr]})


        else:
            total_vars_GaussianHMM_trained = -1
            total_vars_torch_trained = -1
            total_vars_torch_flow_trained = -1

        print("\nn=", n)

    # P_torch_trained_large = model_hmm_nmf_torch
    print("Computing model_hmm_nmf_torch.continuous_score(obs_test) .... ")
    logprob_torch_trained_continuous1 = model_hmm_nmf_torch.continuous_score(obs_test)
    # logprob_torch_trained_continuous2 = model_hmm_nmf_torch.continuous_score(obs_test_grid_labels)

    print("Computing model_hmm_nmf_torch_flow.continuous_score(obs_test) .... ")
    logprob_flow_trained_continuous = model_hmm_nmf_torch_flow.continuous_score(
        obs_test
    )

    print("logprob_hmmlearn_gaussian_trained =\t\t", logprob_hmmlearn_gaussian_trained)

    print("logprob_hmmlearn_gmmhmm_trained =\t\t", logprob_hmmlearn_gmmhmm_trained)

    print("logprob_torch_trained_continuous1= \t\t", logprob_torch_trained_continuous1)
    # print("logprob_torch_trained_continuous2= \t\t", logprob_torch_trained_continuous2)
    print("logprob_flow_trained_continuous1= \t\t", logprob_flow_trained_continuous)

    log_prob_results = dict(
        hmmlearn=logprob_hmmlearn_gaussian_trained,
        gmmhmm=logprob_hmmlearn_gmmhmm_trained,
        cont=logprob_torch_trained_continuous1.item(),
        cflow=logprob_flow_trained_continuous.item(),
    )
    polyaxon.tracking.log_outputs(
        better_flow=1
        if (log_prob_results["cflow"] > log_prob_results["hmmlearn"])
        else 0
    )
    polyaxon.tracking.log_outputs(
        better_torch=1
        if (log_prob_results["cont"] > log_prob_results["hmmlearn"])
        else 0
    )
    polyaxon.tracking.log_metrics(
        **{"logprob_" + key: val for key, val in log_prob_results.items()}
    )
    polyaxon.tracking.log_outputs(
        **{"o_" + key: val for key, val in log_prob_results.items()}
    )
    model_hmm_nmf_torch_flow.save_weights("model_hmm_nmf_torch_flow.pt")
    polyaxon.tracking.log_model(
        "model_hmm_nmf_torch_flow.pt", name="hmm_nmf_flow", framework="torch"
    )
    print("Done.")
    print(log_prob_results)

    # print("Noralized logprobs: \n",
    #     {
    #         "FlowHMM": log_prob_results["cflow"] / n,
    #         "H-Gauss (hmmlearn)": log_prob_results["hmmlearn"] / n,
    #         "GMM-HMM (hmmlearn, n_mix="+str(n_mix)+")": log_prob_results["gmmhmm"] / n,
    #     }
    # )
    print("n = ", n, ", n_train = ", n_train, ", n_test = ", n_test)
    print(colored("NLLs", "red"))

    print(
        colored(
            "logprob_hmmlearn_gaussian_trained =\t\t"
            + str(logprob_hmmlearn_gaussian_trained / n_test),
            "red",
        )
    )

    wandb.log({"logprob_hmmlearn_gaussian_trained": logprob_hmmlearn_gaussian_trained / n_test})
    wandb.log({"G": logprob_hmmlearn_gaussian_trained / n_test})

    print(
        colored(
            "logprob_hmmlearn_gmmhmm_trained =\t\t"
            + str(logprob_hmmlearn_gmmhmm_trained / n_test),
            "red",
        )
    )

    print(
        colored(
            "logprob_torch_trained_continuous1= \t\t"
            + str(logprob_torch_trained_continuous1 / n_test),
            "red",
        )
    )
    # print("logprob_torch_trained_continuous2= \t\t", logprob_torch_trained_continuous2/ n)
    print(
        colored(
            "logprob_flow_trained_continuous1= \t\t"
            + str(logprob_flow_trained_continuous / n_test),
            "red",
        )
    )

    wandb.log({"F": logprob_flow_trained_continuous / n_test})

    for nr, i in enumerate(n_mix_list):
        print(
            colored(
                "GMM-HMM (hmmlearn), n_mix = "
                + str(i)
                + " :"
                + str(logprob_hmmlearn_gmmhmm_trained_models[nr] / n_test),
                "red",
            )
        )
        if wandb_logging:
            wandb.log({f"G{i}": logprob_hmmlearn_gmmhmm_trained_models[nr] / n_test})

    if output_file is not None:
        data_to_save = {
            "EXAMPLE": EXAMPLE,
            "L": L,
            "A_orig": A_orig,
            "m": m,
            "grid_large": grid_large,
            "grid": grid,
            "B_large_GaussianHMM": B_large_GaussianHMM,
            "P_torch_trained_large": P_torch_trained_large.detach().cpu(),
            "P_torch_flow_trained_large": P_torch_flow_trained_large.detach().cpu(),
            "logprob_hmmlearn_gaussian_trained": logprob_hmmlearn_gaussian_trained,
            "logprob_torch_trained_continuous1": logprob_torch_trained_continuous1,
            #'logprob_torch_trained_continuous2': logprob_torch_trained_continuous2,
            "logprob_flow_trained_continuous": logprob_flow_trained_continuous,
            #'logprob_flow_trained_continuous2':logprob_flow_trained_continuous2,
            "obs_train": obs_train,
            "obs_test": obs_test,
            "obs_train_grid": obs_train_grid,
            "obs_test_grid": obs_test_grid,
            "obs_train_grid_labels": obs_train_grid_labels,
            "obs_test_grid_labels": obs_test_grid_labels,
            "nr_epochs": nr_epochs,
            "nr_epochs_torch": nr_epochs_torch,
            "n": n,
            "DATA_TYPE": DATA_TYPE,
        }
        if DATA_TYPE != "REAL":
            data_to_save.update(
                {
                    "B_orig_large": B_orig_large,
                    "MAD_gauss": MAD_gauss,
                    "MAD_torch": MAD_torch,
                    "MAD_torch_flow": MAD_torch_flow,
                    "total_vars_GaussianHMM_trained": total_vars_GaussianHMM_trained,
                    "total_vars_torch_trained": total_vars_torch_trained,
                    "total_vars_torch_flow_trained": total_vars_torch_flow_trained,
                }
            )
        data_outfile = open(output_file, "wb+")
        pickle.dump(data_to_save, data_outfile)
        print("Data written in: ", output_file)

    if show_plots:
        plt.show()



if __name__ == "__main__":
    main()
