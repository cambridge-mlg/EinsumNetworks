import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
import pickle
import os
import time
import utils
import datasets
from PIL import Image
from sklearn.cluster import KMeans

device = 'cuda' if torch.cuda.is_available() else 'cpu'

##################################################################
num_clusters = 100
result_base_path = '../models/einet/svhn/'

num_sums = 40

exponential_family = EinsumNetwork.NormalArray
exponential_family_args = {'min_var': 1e-6, 'max_var': 0.01}

num_epochs = 3
batch_size = 10
online_em_frequency = 50
online_em_stepsize = 0.5

height = 32
width = 32
##################################################################

print("loading data")
train_x_all, train_labels, test_x_all, test_labels, extra_x, extra_labels = datasets.load_svhn()

valid_x_all = train_x_all[50000:, ...]
train_x_all = np.concatenate((train_x_all[0:50000, ...], extra_x), 0)

train_x_all = train_x_all.reshape(train_x_all.shape[0], height, width, 3)
valid_x_all = valid_x_all.reshape(valid_x_all.shape[0], height, width, 3)
test_x_all = test_x_all.reshape(test_x_all.shape[0], height, width, 3)
print("done")


def get_clusters(train_x, num_clusters=100):
    cluster_path = "../auxiliary/svhn"
    filename = os.path.join(cluster_path, "kmeans_{}.pkl".format(num_clusters))

    if not os.path.isfile(filename):
        print("running kmeans...")
        kmeans = KMeans(n_clusters=num_clusters,
                        verbose=3,
                        max_iter=100,
                        n_init=3).fit(train_x.reshape(train_x.shape[0], -1))
        means = kmeans.cluster_centers_
        idx = kmeans.labels_
        utils.mkdir_p(cluster_path)
        pickle.dump((means, idx), open(filename, "wb"))
    else:
        means, idx = pickle.load(open(filename, "rb"))

    return means, idx


def make_shuffled_batch(N, batch_size):
    idx = np.random.permutation(N)
    num_full_batches = N // batch_size
    k = num_full_batches * batch_size
    b_idx = np.array_split(idx[0:k], num_full_batches)
    if k < N:
        b_idx.append(idx[k:])
    return b_idx


def eval_ll(einet, mean, valid_x, batch_size):
    with torch.no_grad():
        shuffled_batch = make_shuffled_batch(len(valid_x), batch_size)
        ll = 0.0
        for batch_idx in shuffled_batch:
            batch = torch.tensor(valid_x[batch_idx, :]).to(device).float()
            batch = batch.reshape(batch.shape[0], height * width, 3)
            batch = batch - mean
            batch = batch / 255.
            ll_sample = einet.forward(batch)
            ll = ll_sample.sum() + ll
        return ll / len(valid_x)


def compute_cluster_means(data, cluster_idx):
    unique_idx = np.unique(cluster_idx)
    means = np.zeros((len(unique_idx), 32, 32, 3), dtype=np.float32)
    for k in unique_idx:
        means[k, ...] = np.mean(data[cluster_idx == k, ...].astype(np.float32), 0)
    return means


def compute_cluster_idx(data, cluster_means):
    cluster_idx = np.zeros(len(data), dtype=np.uint32)
    for k in range(len(data)):
        img = data[k].astype(np.float32)
        cluster_idx[k] = np.argmin(np.sum((cluster_means.reshape(-1, height * width * 3) - img.reshape(1, height * width * 3)) ** 2, 1))
    return cluster_idx


def train(einet, mean, train_x, valid_x, test_x, result_path):
    model_file = os.path.join(result_path, 'einet.mdl')
    graph_file = os.path.join(result_path, 'einet.pc')
    record_file = os.path.join(result_path, 'record.pkl')
    sample_dir = os.path.join(result_path, 'samples')
    utils.mkdir_p(sample_dir)

    record = {'train_ll': [],
              'valid_ll': [],
              'test_ll': [],
              'best_validation_ll': None}

    for epoch_count in range(num_epochs):

        shuffled_batch = make_shuffled_batch(len(train_x), batch_size)
        for batch_counter, batch_idx in enumerate(shuffled_batch):
            batch = torch.tensor(train_x[batch_idx, :]).to(device).float()
            batch = batch.reshape(batch.shape[0], height * width, 3)
            # we subtract the mean for this cluster -- centered data seems to help EM learning
            # we will re-add the mean to the Gaussian means below
            batch = batch - mean
            batch = batch / 255.

            ll_sample = einet.forward(batch)
            log_likelihood = ll_sample.sum()
            log_likelihood.backward()
            einet.em_process_batch()
        einet.em_update()

        ##### evaluate
        train_ll = eval_ll(einet, mean, train_x, batch_size=batch_size)
        valid_ll = eval_ll(einet, mean, valid_x, batch_size=batch_size)
        test_ll = eval_ll(einet, mean, test_x, batch_size=batch_size)

        ##### store results
        record['train_ll'].append(train_ll)
        record['valid_ll'].append(valid_ll)
        record['test_ll'].append(test_ll)

        pickle.dump(record, open(record_file, 'wb'))

        print("[{}]   train LL {}   valid LL {}   test LL {}".format(epoch_count, train_ll, valid_ll, test_ll))

        if record['best_validation_ll'] is None or valid_ll > record['best_validation_ll']:
            record['best_validation_ll'] = valid_ll
            torch.save(einet, model_file)
            Graph.write_gpickle(graph, graph_file)

        if epoch_count % 10 == 0:
            # draw some samples
            samples = einet.sample(num_samples=25, std_correction=0.0).cpu().numpy()
            samples = samples + mean.detach().cpu().numpy() / 255.
            samples -= samples.min()
            samples /= samples.max()
            samples = samples.reshape(samples.shape[0], height, width, 3)
            img = np.zeros((height*5 + 40, width*5 + 40, 3))
            for h in range(5):
                for w in range(5):
                    img[h*(height+10):h*(height+10)+height, w*(width+10):w*(width+10)+width, :] = samples[h*5 + w, :]
            img = Image.fromarray(np.round(img * 255.).astype(np.uint8))
            img.save(os.path.join(sample_dir, "samples{}.jpg".format(epoch_count)))

    # We subtract the mean for the current cluster from the data (centering it at 0).
    # Here we re-add the mean to the Gaussian means. A hacky solution at the moment...
    einet = torch.load(model_file)
    with torch.no_grad():
        params = einet.einet_layers[0].ef_array.params
        mu2 = params[..., 0:3] ** 2
        params[..., 3:] -= mu2
        params[..., 3:] = torch.clamp(params[..., 3:], exponential_family_args['min_var'], exponential_family_args['max_var'])
        params[..., 0:3] += mean.reshape((width*height, 1, 1, 3)) / 255.
        params[..., 3:] += params[..., 0:3] ** 2
    torch.save(einet, model_file)


means, cluster_idx = get_clusters(train_x_all, num_clusters)

print("compute train cluster means")
cluster_means = compute_cluster_means(train_x_all, cluster_idx)
print("compute validation cluster idx")
valid_cluster_idx = compute_cluster_idx(valid_x_all, cluster_means)
print("compute test cluster idx")
test_cluster_idx = compute_cluster_idx(test_x_all, cluster_means)

start_time = time.time()

for cluster_n in range(num_clusters):
    train_x = train_x_all[cluster_idx == cluster_n, ...]
    valid_x = valid_x_all[valid_cluster_idx == cluster_n, ...]
    test_x = test_x_all[test_cluster_idx == cluster_n, ...]

    mean = cluster_means[cluster_n, ...]
    mean = mean.reshape(1, height * width, 3)
    mean = torch.tensor(mean, device=device)

    result_path = result_base_path
    result_path = os.path.join(result_path, "num_clusters_{}".format(num_clusters))
    result_path = os.path.join(result_path, "cluster_{}".format(cluster_n))

    graph = Graph.poon_domingos_structure(shape=(height, width), axes=[1], delta=[8])

    args = EinsumNetwork.Args(
        num_var=height*width,
        num_dims=3,
        num_classes=1,
        num_sums=num_sums,
        num_input_distributions=num_sums,
        exponential_family=exponential_family,
        exponential_family_args=exponential_family_args,
        online_em_frequency=online_em_frequency,
        online_em_stepsize=online_em_stepsize)

    print()
    print(result_path)

    utils.mkdir_p(result_path)
    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    print(einet)

    train(einet, mean, train_x, valid_x, test_x, result_path)

print()
print("elapsed time {}".format(time.time() - start_time))
