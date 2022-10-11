# FlowHMM: Flow-based continuous hidden Markov models
<img alt="FlowHMM schema" style="float: right; width: 450px; padding-left: 25px" src="img/FlowHMM_schema.png">

***Abstract***

Continuous hidden Markov models (HMMs) assume that observations 
are generated from a mixture of Gaussian densities, limiting their ability to model more complex distributions. 
In this work, we address this shortcoming and propose a novel continuous HMM
that allows to learn general continuous observation densities without constraining them to follow a Gaussian 
distribution or their mixtures.
To that end, we leverage deep flow-based architectures that model complex, non-Gaussian functions. Moreover, 
to simplify optimization and avoid costly expectation-maximization algorithm, we use the co-occurrence matrix
of discretized observations and consider the joint distribution of pairs of co-observed values.
Even though our model is trained on discretized observations, 
it represents a continuous variant of HMM during inference, thanks 
to applying a separate flow model for each hidden state. The experiments 
on synthetic and real datasets show that our method outperforms 
Gaussian baselines.

# Installation
## Conda environment
* Create the new conda environment
> `conda env create -f conda.yml`
* or update the existing one
> `conda env update -f conda.yml --prune`

## Packages installation
* Install all the required packages with single command
> `poetry install`

# Examples

* synthetic dataset with the following distributions (see **Example 1** in paper):
  * 2 gaussians
  * 1 uniform
```bash
python main.py -e examples/SYNTHETIC_2G_1U.yaml \
 --nr_epochs=500 \
 --training_type=Q_training # or EM \
 --add_noise=True --noise_var=0.1 \
 --show_plots \
 --extra_n=$N 
```
where `N` variable is the length of training observations.
We chose `N=1000, 10000, 100000`; see [SYNTHETIC_2G_1U.yaml](examples/SYNTHETIC_2G_1U.yaml) for more details.

* synthetic dataset with the following distributions (see **Example 2** in paper):
  * 1 beta
  * 1 uniform
  * 1 gaussian

```bash
python main.py -e examples/SYNTHETIC_1B_1U.yaml \
 --nr_epochs=1000 \
 --add_noise=True --noise_var=0.00005 \
 --training_type=Q_training # or EM \
 --show_plots \
 --extra_n=$N --extra_L=$L 
```
where `N` variable is the length of training observations and `L`
is the number of hidden states (flow models to learn).
We chose `N`=1000, 10000, 100000 and `L`=2, 3;
see [SYNTHETIC_1B_1U_1G.yaml](examples/SYNTHETIC_1B_1U_1G.yaml) for more details.

* 2D synthetic dataset with two "Moons" and one Uniform distribution (see **Example 5** in paper):
```bash
python main2d.py \
 -e examples/SYNTHETIC_2d_data_1U_2Moons.yaml # variant (a) \
 # -e examples/SYNTHETIC_2d_data_1U_2Moons_A2.yaml # variant (b) \
 --nr_epochs=500 \
 --training_type=Q_training # or EM \
 --extra_n=1000 --lrate=0.01 \
 --add_noise --noise_var=0.001 \
 --show_plots
```

* 2D synthetic dataset with one bivariate Gaussian, one Uniform and one related to geometric Brownian motion (see **Example 6** in paper):
```bash
python main2d.py \
  -e examples/SYNTHETIC_2d_data_1G_1U_1GeomBrownianMotion.yaml # variant (a) \
  # -e examples/SYNTHETIC_2d_data_1G_1U_1GeomBrownianMotion_A2.yaml # variant (b) \
  --nr_epochs=500
  --training_type=Q_training # or EM \
  --extra_n=1000 \ 
  --lrate=0.01  \
  --add_noise --noise_var=0.001 \
  --show_plots
```

[//]: # (TODO: add run examples with 6D datasets)

