import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
from EinsumNetwork.EinetMixture import EinetMixture
import pickle
import os
import datasets
import utils
from PIL import Image



device = 'cpu'

einet_path = '../models/einet/svhn/num_clusters_100'

sample_path = '../samples/svhn'


utils.mkdir_p(sample_path)



height = 32
width = 32
num_clusters = 100

poon_domingos_pieces = [4]
num_sums = 40
input_multiplier = 1

structure = 'poon_domingos_vertical'

exponential_family = EinsumNetwork.NormalArray
exponential_family_args = {'min_var': 1e-6, 'max_var': 0.01}

if input_multiplier == 1:
    block_mix_input = None
    num_input_distributions = num_sums
else:
    block_mix_input = num_sums
    num_input_distributions = num_sums * input_multiplier

#######################################################################################


def compute_cluster_means(data, cluster_idx):
    unique_idx = np.unique(cluster_idx)
    means = np.zeros((len(unique_idx), height, width, 3), dtype=np.float32)
    for k in unique_idx:
        means[k, ...] = np.mean(data[cluster_idx == k, ...].astype(np.float32), 0)
    return means


def compute_cluster_idx(data, cluster_means):
    cluster_idx = np.zeros(len(data), dtype=np.uint32)
    for k in range(len(data)):
        img = data[k].astype(np.float32)
        cluster_idx[k] = np.argmin(np.sum((cluster_means.reshape(-1, height * width * 3) - img.reshape(1, height * width * 3)) ** 2, 1))
    return cluster_idx


#############################
# Data

print("loading data")
train_x_all, train_labels, test_x_all, test_labels, extra_x, extra_labels = datasets.load_svhn()

valid_x_all = train_x_all[50000:, ...]
train_x_all = np.concatenate((train_x_all[0:50000, ...], extra_x), 0)
train_x_all = train_x_all.reshape(train_x_all.shape[0], height, width, 3)
valid_x_all = valid_x_all.reshape(valid_x_all.shape[0], height, width, 3)
test_x_all = test_x_all.reshape(test_x_all.shape[0], height, width, 3)

train_x_all = torch.tensor(train_x_all, device=device, dtype=torch.float32).reshape(-1, width*height, 3) / 255.
valid_x_all = torch.tensor(valid_x_all, device=device, dtype=torch.float32).reshape(-1, width*height, 3) / 255.
test_x_all = torch.tensor(test_x_all, device=device, dtype=torch.float32).reshape(-1, width*height, 3) / 255.
print("done")

_, cluster_idx = pickle.load(open('../auxiliary/svhn/kmeans_{}.pkl'.format(num_clusters), 'rb'))

# make a mixture of EiNets
p = np.histogram(cluster_idx, num_clusters)[0].astype(np.float32)
p = p / p.sum()

einets = []
for k in range(num_clusters):
    print("Load model for cluster {}".format(k))
    model_file = os.path.join(einet_path, 'cluster_{}'.format(k), 'einet.mdl')
    einets.append(torch.load(model_file).to(device))

mixture = EinetMixture(p, einets)

L = 7
samples = mixture.sample(L**2, std_correction=0.0)
utils.save_image_stack(samples.reshape(-1, height, width, 3),
                       L, L,
                       os.path.join(sample_path, 'einet_samples.png'),
                       margin=2,
                       margin_gray_val=0.,
                       frame=2,
                       frame_gray_val=0.0)
print("Saved samples to {}".format(os.path.join(sample_path, 'einet_samples.png')))


num_reconstructions = 10

rp = np.random.permutation(test_x_all.shape[0])
test_x = test_x_all[rp[0:num_reconstructions], ...]

# Make covered images -- Top
test_x_covered_top = np.reshape(test_x.clone().cpu().numpy(), (num_reconstructions, height, width, 3))
test_x_covered_top[:, 0:round(height/2), ...] = 0.0

# Draw conditional samples for reconstruction -- Top
image_scope = np.array(range(height * width)).reshape(height, width)
marginalize_idx = list(image_scope[0:round(height/2), :].reshape(-1))
rec_samples_top = mixture.conditional_sample(test_x, marginalize_idx, std_correction=0.0)

# Make covered images -- Left
test_x_covered_left = np.reshape(test_x.clone().cpu().numpy(), (num_reconstructions, height, width, 3))
test_x_covered_left[:, :, 0:round(width/2), ...] = 0.0

# Draw conditional samples for reconstruction -- Left
image_scope = np.array(range(height * width)).reshape(height, width)
marginalize_idx = list(image_scope[:, 0:round(width/2)].reshape(-1))
rec_samples_left = mixture.conditional_sample(test_x, marginalize_idx, std_correction=0.0)

reconstruction_stack = np.concatenate((np.reshape(test_x.cpu().numpy(), (num_reconstructions, height, width, 3)),
                                       np.reshape(test_x_covered_top, (num_reconstructions, height, width, 3)),
                                       np.reshape(rec_samples_top, (num_reconstructions, height, width, 3)),
                                       np.reshape(test_x_covered_left, (num_reconstructions, height, width, 3)),
                                       np.reshape(rec_samples_left, (num_reconstructions, height, width, 3))), 0)

reconstruction_stack -= reconstruction_stack.min()
reconstruction_stack /= reconstruction_stack.max()
utils.save_image_stack(reconstruction_stack, 5,
                       num_reconstructions, os.path.join(sample_path, 'einet_reconstructions.png'),
                       margin=2,
                       margin_gray_val=0.,
                       frame=2,
                       frame_gray_val=0.0)
print("Saved reconstructions to {}".format(os.path.join(sample_path, 'einet_reconstructions.png')))

print("Compute test log-likelihood...")
ll = mixture.log_likelihood(test_x_all)
print("log-likelihood = {}".format(ll / test_x_all.shape[0]))
