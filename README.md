# FFVD: Free-Form Variational Dynamics for Gaussian Process State-Space Models

FFVD provides Free-Form Variational Dynamics Inference in Tensorflow. 

## Usage

The main workhorse of our library is the `FFVD` function.

For example, you may run as an example:
```
FFVD()
```

Then the FFVD can be trained as follows:
```
yhat = net(x_train)
nll = F.cross_entropy(yhat, y_train)
kl = sum(m.kl_divergence() for m in net.modules()
         if hasattr(m, "kl_divergence"))
loss = nll + kl / dataset_size
loss.backward()
optim.step()
```


## Code structure


### Requirements
* tensorflow
* ticks
* xarray
* torchdiffeq


## Example script

We provide an example script for training FFVD on real-world datasets in `scripts/FFVD.py`.


To train a FFVD, you can run the script for example as:
```
python scripts/FFVD.py --inference-config=configs/ffg_u_cifar10.json \
    --num-epochs=200 --ml-epochs=100 --annealing-epochs=50 --lr=1e-3 \
    --milestones=100 --resnet=18 --cifar=10 --verbose --progress-bar
```
The full list of command line options for the script is:
```
  --num-epochs: Total number of training epochs
  --train-samples: Number of MC samples to draw from the variational posterior during training for the data log likelihood  
  --test-samples: Number of samples to average the predictive posterior during testing.
  --annealing-epochs: Number of training epochs over which the weight of the KL term is annealed linearly.
  --ml-epochs: Number of training epochs where the weight of the KL term is 0.
  --inference-config: Path to the inference config file
  --output-dir: Directory in which to store state dicts of the network and optimizer, and the final calibration plot. 
  --verbose: Switch for printing validation accuracy and calibration at every epoch.
  --progress-bar: Switch for tqdm progress bar for epochs and batches.
  --lr: Initial learning rate.
  --seed: Random seed.
  --cifar: 10 or 100 for the corresponding CIFAR dataset.
  --optimizer: sgd or adam for the corresponding optimizer.
  --momentum: momentum if using sgd.
  --milestones: Comma-separated list of epochs after which to decay the learning rate by a factor of gamma. 
  --gamma: Multiplicative decay factor for the learning rate.
  --resnet: Which ResNet architecture from torchvision to use (must be one of {18, 34, 50, 101, 152}).
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


