# Einsum Networks -- Fast and Scalable Learning of Tractable Probabilistic Circuits

PyTorch implementation of Einsum Netwrks, proposed in 

R. Peharz, S. Lang, A. Vergari, K. Stelzner, A. Molina, M. Trapp, G. Van den Broeck, K. Kersting, Z. Ghahramani,
**Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits**,
*ICML 2020*.

We are still about to clean the code and add some experiments, but the implementation is already fully there and ready to play.

# Setup 

This will clone the repo, install a python virtual env (requires pythn 3.6), the required packages, and will download some datasets.

    git clone https://github.com/cambridge-mlg/EinsumNetworks
    cd EinsumNetworks
    ./setup.sh

# Demos

We have add some quick run demos, to illustrate the usage of the code.

    source ./venv/bin/activate
    cd src
    python demo_mnist.py
    python demo_debd.py

# Train Mixture of EiNets on SVHN

    source ./venv/bin/activate
    cd src
    python train_svhn_mixture.py
    python eval_svhn_mixture.py
