import os
import sys

import numpy as np
import torch

sys.path.append("../src")
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

demo_text = """
This scripts loads mnist, trains an EiNet for one epoch, saves it to disk, then reloads it. 
"""
print(demo_text)

# Make EinsumNetwork
######################################

def make_region_graph(structure="poon-domingos",
                      height=None, width=None,
                      pd_pieces=None, depth=None, n_repetitions=None):

    
    graph = None
    if structure == 'poon-domingos':
        assert pd_pieces is not None
        pd_delta = [[height / d, width / d] for d in pd_pieces]
        graph = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)
    elif structure == 'binary-trees':
        n_vars = height * width
        graph = Graph.random_binary_trees(num_var=n_vars,
                                          depth=depth,
                                          num_repetitions=n_repetitions)
    else:
        raise AssertionError("Unknown Structure")

    return graph

def make_einet(region_graph,
               n_vars,
               n_classes=1,
               n_sums=10,
               n_input_dists=10,
               exp_fam="binomial", exp_fam_args={'N': 255},
               use_em=False,
               em_freq=None, em_stepsize=None):

    if exp_fam == "binomial":
        exponential_family = EinsumNetwork.BinomialArray
    elif exp_fam == "categorical":
        exponential_family = EinsumNetwork.CategoricalArray
    else:
        raise ValueError(f"Unrecognized exponential family: {exp_fam}")

    args = EinsumNetwork.Args(
        num_var=n_vars,
        num_dims=1,
        num_classes=n_classes,
        num_sums=n_sums,
        num_input_distributions=n_input_dists,
        exponential_family=exponential_family,
        exponential_family_args=exp_fam_args,
        use_em=use_em,
        online_em_frequency=em_freq,
        online_em_stepsize=em_stepsize)

    einet = EinsumNetwork.EinsumNetwork(region_graph, args)
    return einet

def save_einet(einet, graph, out_path):
    # save model
    os.makedirs(out_path, exist_ok=True)
    graph_file = os.path.join(out_path, "einet.rg")
    Graph.write_gpickle(graph, graph_file)
    print("Saved PC graph to {}".format(graph_file))
    model_file = os.path.join(out_path, "einet.pth")
    torch.save(einet, model_file)
    print("Saved model to {}".format(model_file))

def save_einet_state(einet, graph, out_path):
    # save model
    os.makedirs(out_path, exist_ok=True)
    # graph_file = os.path.join(out_path, "einet.rg")
    # Graph.write_gpickle(graph, graph_file)
    # print("Saved PC graph to {}".format(graph_file))
    model_file = os.path.join(out_path, "einet.pth")
    torch.save(einet.state_dict(), model_file)
    print("Saved model to {}".format(model_file))

def load_einet(model_path, einet_file='einet.pth', graph_file='einet.rg'):

    # reload model
    einet, graph = None, None
    model_file = os.path.join(model_path, einet_file)
    einet = torch.load(model_file)
    print("Loaded model from {}".format(model_file))
    
    if graph_file:
        graph_file = os.path.join(model_path, graph_file)
        graph = Graph.read_gpickle(graph_file)
        
    return einet, graph

def load_einet_state(model_path,
                     einet_file='einet.pth',
                     graph_file='einet.rg',
                     n_vars=None, n_classes=None, n_sums=None, n_input_dists=None,
                     exp_fam=None, exp_fam_args=None,
                     use_em=None, em_freq=None, em_stepsize=None,
                     graph=None):

    # reload model
    einet = None

    if graph is None:
        if graph_file:
            graph_file = os.path.join(model_path, graph_file)
            graph = Graph.read_gpickle(graph_file)
        else:
            raise ValueError(f"Cannot create graph")
    
    model_file = os.path.join(model_path, einet_file)
    
    einet = make_einet(graph,
               n_vars=n_vars,
               n_classes=n_classes,
               n_sums=n_sums,
               n_input_dists=n_input_dists,
               exp_fam=exp_fam, exp_fam_args=exp_fam_args,
               use_em=use_em,
               em_freq=em_freq, em_stepsize=em_stepsize)
    einet.load_state_dict(torch.load(model_file))
    
    print("Loaded model from {}".format(model_file))
        
    return einet, graph

def check_einets_eq(e1, e2):
    assert len(e1.einet_layers) == len(e2.einet_layers)
    for l, l_p  in zip(e1.einet_layers, e2.einet_layers):
        if hasattr(l, "params"):
            assert torch.all(torch.eq(l.params, l_p.params))


classes = [7]
num_epochs = 5
batch_size = 100

############################################################################


# get data
train_x, train_labels, test_x, test_labels = datasets.load_mnist()

# validation split
valid_x = train_x[-10000:, :]
train_x = train_x[:-10000, :]
valid_labels = train_labels[-10000:]
train_labels = train_labels[:-10000]

# pick the selected classes
if classes is not None:
    train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
    valid_x = valid_x[np.any(np.stack([valid_labels == c for c in classes], 1), 1), :]
    test_x = test_x[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]

train_x = torch.from_numpy(train_x).to(torch.device(device))
valid_x = torch.from_numpy(valid_x).to(torch.device(device))
test_x = torch.from_numpy(test_x).to(torch.device(device))


use_em = False
rg = make_region_graph(structure="poon-domingos",
                      height=28, width=28,
                      pd_pieces=[4], depth=None, n_repetitions=None)
einet = make_einet(region_graph=rg,
               n_vars=28*28,
               n_classes=1,
               n_sums=10,
               n_input_dists=10,
               exp_fam="binomial", exp_fam_args={'N': 255},
               use_em=use_em,
               # em_freq=1, em_stepsize=0.05
               em_freq=None, em_stepsize=None
               )
einet.initialize()
einet.to(device)
print(einet)


# Train for one epoch
######################################

train_N = train_x.shape[0]
valid_N = valid_x.shape[0]
test_N = test_x.shape[0]


##### evaluate
einet.eval()
train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
print("pre-train   train LL {}   valid LL {}   test LL {}".format(
    train_ll / train_N,
    valid_ll / valid_N,
    test_ll / test_N))


######################################
# Save untrained
#
# out_path = './saved-pretrained-state'
# save_einet_state(einet, rg, out_path=out_path)
# einet_p, graph_p = load_einet_state(out_path, graph=rg,
#                                     n_vars=28*28,
#                n_classes=1,
#                n_sums=10,
#                n_input_dists=10,
#                exp_fam="binomial", exp_fam_args={'N': 255},
#                use_em=False,
#                # em_freq=1, em_stepsize=0.05
#                em_freq=None, em_stepsize=None)

out_path = './saved-pretrained'
save_einet(einet, rg, out_path=out_path)
einet_p, graph_p = load_einet(out_path)


##### check same model
check_einets_eq(einet, einet_p)
print("Einets equal")

##### evaluate
einet_p.eval()
train_ll = EinsumNetwork.eval_loglikelihood_batched(einet_p, train_x, batch_size=batch_size)
valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet_p, valid_x, batch_size=batch_size)
test_ll = EinsumNetwork.eval_loglikelihood_batched(einet_p, test_x, batch_size=batch_size)
print("pre-train (after loading)   train LL {}   valid LL {}   test LL {}".format(
    train_ll / train_N,
    valid_ll / valid_N,
    test_ll / test_N))


#####
einet.train()

optimizer = None
if not use_em:
    optimizer = torch.optim.Adam(einet.parameters(), lr=1e-2)

n_epochs = 5
for epoch in range(n_epochs):
    idx_batches = torch.randperm(train_N, device=device).split(batch_size)

    total_ll = 0.0
    for idx in idx_batches:
        batch_x = train_x[idx, :]
        outputs = einet.forward(batch_x)
        ll_sample = EinsumNetwork.log_likelihoods(outputs)
        log_likelihood = ll_sample.sum()
        

        if use_em:
            log_likelihood.backward()
            einet.em_process_batch()
        else:
            optimizer.zero_grad()
            objective = -log_likelihood
            objective.backward()
            optimizer.step()
            
        total_ll += log_likelihood.detach().item()
    print(f"done epoch {epoch+1}/{n_epochs}")

    if use_em:
        einet.em_update()


# evaluate log-likelihoods
einet.eval()
# evaluate log-likelihoods on re-loaded model
train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
print()
print("post-training (before saving) --- train LL {}   valid LL {}   test LL {}".format(
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))


save_einet(einet, rg, out_path=out_path)
einet_p, graph_p = load_einet(out_path)

##### check same model
check_einets_eq(einet, einet_p)
print("Einets equal")

##### evaluate
einet_p.eval()
train_ll = EinsumNetwork.eval_loglikelihood_batched(einet_p, train_x, batch_size=batch_size)
valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet_p, valid_x, batch_size=batch_size)
test_ll = EinsumNetwork.eval_loglikelihood_batched(einet_p, test_x, batch_size=batch_size)
print("pre-train (after re-loading)   train LL {}   valid LL {}   test LL {}".format(
    train_ll / train_N,
    valid_ll / valid_N,
    test_ll / test_N))

