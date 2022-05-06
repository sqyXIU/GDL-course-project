## Optimize Non-Parametric Instance Discrimination Clustering via Implicit Label Merging

This the course project for CS 8395 Special Topics in Deep learning

## Highlight

- We improve memory bank non parametric unsupervised learning by clustering.
- Combination of two update schemas on Convnet
- Faster convergence on training.



## Usage

Our code extends the pytorch implementation of memory bank non parametric unsupervised learning in (https://github.com/zhirongw/lemniscate.pytorch). 

- Training on CIFAR10:

  `python cifar.py --nce-k 0 --nce-t 0.1 --lr 0.03`

- Training on MNIST:

  `python cifar2.py --nce-k 0 --nce-t 0.1 --lr 0.03`
