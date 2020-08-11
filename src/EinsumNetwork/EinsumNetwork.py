from EinsumNetwork import Graph
from EinsumNetwork.FactorizedLeafLayer import *
from EinsumNetwork.SumLayer import *

class Args(object):
    """
    Arguments for EinsumNetwork class.

    num_var: number of random variables (RVs). An RV might be multidimensional though -- see num_dims.
    num_dims: number of dimensions per RV. E.g. you can model an 32x32 RGB image as an 32x32 array of three dimensional
              RVs.
    num_input_distributions: number of distributions per input region (K in the paper).
    num_sums: number of sum nodes per internal region (K in the paper).
    num_classes: number of outputs of the PC.
    exponential_family: which exponential family to use; (sub-class ExponentialFamilyTensor).
    exponential_family_args: arguments for the exponential family, e.g. trial-number N for Binomial.
    use_em: determines if the internal em algorithm shall be used; otherwise you might use e.g. SGD.
    online_em_frequency: how often shall online be triggered in terms, of batches? 1 means after each batch, None means
                         batch EM. In the latter case, EM updates must be triggered manually after each epoch.
    online_em_stepsize: stepsize for inline EM. Only relevant if online_em_frequency not is None.
    """
    def __init__(self,
                 num_var=20,
                 num_dims=1,
                 num_input_distributions=10,
                 num_sums=10,
                 num_classes=1,
                 exponential_family=NormalArray,
                 exponential_family_args=None,
                 use_em=True,
                 online_em_frequency=1,
                 online_em_stepsize=0.05):
        self.num_var = num_var
        self.num_dims = num_dims
        self.num_input_distributions = num_input_distributions
        self.num_sums = num_sums
        self.num_classes = num_classes
        self.exponential_family = exponential_family
        if exponential_family_args is None:
            exponential_family_args = {}
        self.exponential_family_args = exponential_family_args
        self.use_em = use_em
        self.online_em_frequency = online_em_frequency
        self.online_em_stepsize = online_em_stepsize


class EinsumNetwork(torch.nn.Module):
    """
    Implements Einsum Networks (EiNets).

    The basic philosophy of EiNets is to summarize many PC nodes in monolithic GPU-friendly parallel operations.
    An EiNet can be seen as a special layered feed-forward neural network, consisting of a sequence of layers. Each
    layer can in principle get input from all layers before.

    As a general design principle, each layer in EinsumNetworks produces a tensor of log-densities in the forward pass,
    of generic shape
            (batch_size, vector_length, num_nodes)
    where
        batch_size is the number of samples in a mini-batch.
        vector_length is the length of the vectorized operations; this is called K in the paper -- in the paper we
                      assumed this constant over the whole EiNet, but this can be partially relaxed.
        num_nodes is the number of nodes which are realized in parallel using this layer.
    Thus, in classical PCs, we would interpret the each layer as a collection of vector_length * num_nodes PC nodes.

    The class EinsumNetork mainly governs the layer-wise layout, initialization, forward() calls, EM learning, etc.
    """

    def __init__(self, graph, args=None):
        """Make an EinsumNetwork."""
        super(EinsumNetwork, self).__init__()

        check_flag, check_msg = Graph.check_graph(graph)
        if not check_flag:
            raise AssertionError(check_msg)
        self.graph = graph

        self.args = args if args is not None else Args()

        if len(Graph.get_roots(self.graph)) != 1:
            raise AssertionError("Currently only EinNets with single root node supported.")

        root = Graph.get_roots(self.graph)[0]
        if tuple(range(self.args.num_var)) != root.scope:
            raise AssertionError("The graph should be over tuple(range(num_var)).")

        for node in Graph.get_leaves(self.graph):
            node.num_dist = self.args.num_input_distributions

        for node in Graph.get_sums(self.graph):
            if node is root:
                node.num_dist = self.args.num_classes
            else:
                node.num_dist = self.args.num_sums

        # Algorithm 1 in the paper -- organize the PC in layers
        self.graph_layers = Graph.topological_layers(self.graph)

        # input layer
        einet_layers = [FactorizedLeafLayer(self.graph_layers[0],
                                            self.args.num_var,
                                            self.args.num_dims,
                                            self.args.exponential_family,
                                            self.args.exponential_family_args,
                                            use_em=self.args.use_em)]

        # internal layers
        for c, layer in enumerate(self.graph_layers[1:]):
            if c % 2 == 0:   # product layer
                einet_layers.append(EinsumLayer(self.graph, layer, einet_layers, use_em=self.args.use_em))
            else:     # sum layer
                # the Mixing layer is only for regions which have multiple partitions as children.
                multi_sums = [n for n in layer if len(graph.succ[n]) > 1]
                if multi_sums:
                    einet_layers.append(EinsumMixingLayer(graph, multi_sums, einet_layers[-1], use_em=self.args.use_em))

        self.einet_layers = torch.nn.ModuleList(einet_layers)
        self.em_set_hyperparams(self.args.online_em_frequency, self.args.online_em_stepsize)

    def initialize(self, init_dict=None):
        """
        Initialize layers.

        :param init_dict: None; or
                          dictionary int->initializer; mapping layer index to initializers; or
                          dictionary layer->initializer;
                          the init_dict does not need to have an initializer for all layers
        :return: None
        """
        if init_dict is None:
            init_dict = dict()
        if all([type(k) == int for k in init_dict.keys()]):
            init_dict = {self.einet_layers[k]: init_dict[k] for k in init_dict.keys()}
        for layer in self.einet_layers:
            layer.initialize(init_dict.get(layer, 'default'))

    def set_marginalization_idx(self, idx):
        """Set indices of marginalized variables."""
        self.einet_layers[0].set_marginalization_idx(idx)

    def get_marginalization_idx(self):
        """Get indices of marginalized variables."""
        return self.einet_layers[0].get_marginalization_idx()

    def forward(self, x):
        """Evaluate the EinsumNetwork feed forward."""

        input_layer = self.einet_layers[0]
        input_layer(x=x)
        for einsum_layer in self.einet_layers[1:]:
            einsum_layer()
        return self.einet_layers[-1].prob[:, :, 0]

    def backtrack(self, num_samples=1, class_idx=0, x=None, mode='sampling', **kwargs):
        """
        Perform backtracking; for sampling or MPE approximation.
        """

        sample_idx = {l: [] for l in self.einet_layers}
        dist_idx = {l: [] for l in self.einet_layers}
        reg_idx = {l: [] for l in self.einet_layers}

        root = self.einet_layers[-1]

        if x is not None:
            self.forward(x)
            num_samples = x.shape[0]

        sample_idx[root] = list(range(num_samples))
        dist_idx[root] = [class_idx] * num_samples
        reg_idx[root] = [0] * num_samples

        for layer in reversed(self.einet_layers):

            if not sample_idx[layer]:
                continue

            if type(layer) == EinsumLayer:

                ret = layer.backtrack(dist_idx[layer],
                                      reg_idx[layer],
                                      sample_idx[layer],
                                      use_evidence=(x is not None),
                                      mode=mode,
                                      **kwargs)
                dist_idx_left, dist_idx_right, reg_idx_left, reg_idx_right, layers_left, layers_right = ret

                for c, layer_left in enumerate(layers_left):
                    sample_idx[layer_left].append(sample_idx[layer][c])
                    dist_idx[layer_left].append(dist_idx_left[c])
                    reg_idx[layer_left].append(reg_idx_left[c])

                for c, layer_right in enumerate(layers_right):
                    sample_idx[layer_right].append(sample_idx[layer][c])
                    dist_idx[layer_right].append(dist_idx_right[c])
                    reg_idx[layer_right].append(reg_idx_right[c])

            elif type(layer) == EinsumMixingLayer:

                ret = layer.backtrack(dist_idx[layer],
                                      reg_idx[layer],
                                      sample_idx[layer],
                                      use_evidence=(x is not None),
                                      mode=mode,
                                      **kwargs)
                dist_idx_out, reg_idx_out, layers_out = ret

                for c, layer_out in enumerate(layers_out):
                    sample_idx[layer_out].append(sample_idx[layer][c])
                    dist_idx[layer_out].append(dist_idx_out[c])
                    reg_idx[layer_out].append(reg_idx_out[c])

            elif type(layer) == FactorizedLeafLayer:

                unique_sample_idx = sorted(list(set(sample_idx[layer])))
                if unique_sample_idx != sample_idx[root]:
                    raise AssertionError("This should not happen.")

                dist_idx_sample = []
                reg_idx_sample = []
                for sidx in unique_sample_idx:
                    dist_idx_sample.append([dist_idx[layer][c] for c, i in enumerate(sample_idx[layer]) if i == sidx])
                    reg_idx_sample.append([reg_idx[layer][c] for c, i in enumerate(sample_idx[layer]) if i == sidx])

                samples = layer.backtrack(dist_idx_sample, reg_idx_sample, mode=mode, **kwargs)

                if self.args.num_dims == 1:
                    samples = torch.squeeze(samples, 2)

                if x is not None:
                    marg_idx = layer.get_marginalization_idx()
                    keep_idx = [i for i in range(self.args.num_var) if i not in marg_idx]
                    samples[:, keep_idx] = x[:, keep_idx]

                return samples

    def sample(self, num_samples=1, class_idx=0, x=None, **kwargs):
        return self.backtrack(num_samples=num_samples, class_idx=class_idx, x=x, mode='sample', **kwargs)

    def mpe(self, num_samples=1, class_idx=0, x=None, **kwargs):
        return self.backtrack(num_samples=num_samples, class_idx=class_idx, x=x, mode='argmax', **kwargs)

    def em_set_hyperparams(self, online_em_frequency, online_em_stepsize, purge=True):
        for l in self.einet_layers:
            l.em_set_hyperparams(online_em_frequency, online_em_stepsize, purge)

    def em_process_batch(self):
        for l in self.einet_layers:
            l.em_process_batch()

    def em_update(self):
        for l in self.einet_layers:
            l.em_update()


def log_likelihoods(outputs, labels=None):
    """Compute the likelihood of EinsumNetwork."""
    if labels is None:
        num_dist = outputs.shape[-1]
        if num_dist == 1:
            lls = outputs
        else:
            num_dist = torch.tensor(float(num_dist), device=outputs.device)
            lls = torch.logsumexp(outputs - torch.log(num_dist), -1)
    else:
        lls = outputs.gather(-1, labels.unsqueeze(-1))
    return lls


def eval_accuracy_batched(einet, x, labels, batch_size):
    """Computes accuracy in batched way."""
    with torch.no_grad():
        idx_batches = torch.arange(0, x.shape[0], dtype=torch.int64, device=x.device).split(batch_size)
        n_correct = 0
        for batch_count, idx in enumerate(idx_batches):
            batch_x = x[idx, :]
            batch_labels = labels[idx]
            outputs = einet.forward(batch_x)
            _, pred = outputs.max(1)
            n_correct += torch.sum(pred == batch_labels)
        return (n_correct.float() / x.shape[0]).item()


def eval_loglikelihood_batched(einet, x, labels=None, batch_size=100):
    """Computes log-likelihood in batched way."""
    with torch.no_grad():
        idx_batches = torch.arange(0, x.shape[0], dtype=torch.int64, device=x.device).split(batch_size)
        ll_total = 0.0
        for batch_count, idx in enumerate(idx_batches):
            batch_x = x[idx, :]
            if labels is not None:
                batch_labels = labels[idx]
            else:
                batch_labels = None
            outputs = einet(batch_x)
            ll_sample = log_likelihoods(outputs, batch_labels)
            ll_total += ll_sample.sum().item()
        return ll_total
