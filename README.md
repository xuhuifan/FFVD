# FFVD: Free-Form Variational Dynamics for Gaussian Process State-Space Models

FFVD provides an implementation of the method in the paper [Free-Form Variational Inference for Gaussian Process State-Space Models](https://arxiv.org/abs/2302.09921)  in Tensorflow. 

## Usage

The main workhorse of our library is the `FFVD_Main.py` function.


## Code structure
```
Factnonlin_ini: the model initializations (trained by factorized non-linear model)
data: the six real-world datasets
vfegpssm: the source code of FFVD
```


### Requirements
* tensorflow
* ticks
* xarray


## Example script

We provide an example script for training FFVD on real-world datasets in `FFVD_Main.py`.


To train a FFVD, you can run the script for example as:
```
python FFVD_Main.py
```
The full list of command line options for the script is:

```
--iterations: number of iterations in FFVD, type=int, default=2000
--num_inducing: number of inducing points, type=int, default=100
--posterior_sample_spacing: number of sample spacing in SG-HMC, type=int, default=50
--file_id: the index of initialization file, type=int, default=3
--file_index: the index of dataset, type=int, default=2
--case_val: which model configuration we are using, type=int, default=4
--x_dims: number of latent states, type=list, default=[4]
--samples: number of posterior samples, type=int, default=10
--ratio: training versus testing data, type = float, default=0.5
--kernel_type: kernel typle used in GP, choices=['SquaredExponential', 'LinearK'], default='SquaredExponential'
--data_index: type=int, default=4
```


If the repository is helpful to your research, please cite the following:

```
@article{
icml2022ffvd,
title={Free-Form Variational Inference for Gaussian Process State-Space Models},
author={Xuhui Fan and Edwin V. Bonilla and Terence J. O’Kane and Scott A. Sisson},
journal={International Conference on Machine Learning},
year={2023}
}
```


