import torch
import torch.nn.functional as F
from EinsumNetwork.Layer import Layer
import functools
from itertools import count
from utils import sample_matrix_categorical
softmax = torch.nn.functional.softmax


class SumLayer(Layer):
    """
    Implements an abstract SumLayer class. Takes care of parameters and EM.
    EinsumLayer and MixingLayer are derived from SumLayer.
    """

    def __init__(self, params_shape, normalization_dims, use_em, params_mask=None):
        """
        :param params_shape: shape of tensor containing all sum weights (tuple of ints).
        :param normalization_dims: the dimensions (axes) of the sum-weights which shall be normalized
                                   (int of tuple of ints)
        :param use_em: use the on-board EM algorithm?
        :param params_mask: binary mask for masking out certain parameters (tensor of shape params_shape).
        """
        super(SumLayer, self).__init__(use_em=use_em)

        self.params_shape = params_shape
        self.params = None
        self.normalization_dims = normalization_dims
        if params_mask is not None:
            params_mask = params_mask.clone().detach()
        self.register_buffer('params_mask', params_mask)

        self.online_em_frequency = None
        self.online_em_stepsize = None
        self._online_em_counter = 0

    # --------------------------------------------------------------------------------
    # The following functions need to be implemented in derived classes.

    def _forward(self, params):
        """
        Implementation of the actual sum operation.

        :param params: sum-weights to use.
        :return: result of the sum layer. Must yield a (batch_size, num_dist, num_nodes) tensor of log-densities.
                 Here, num_dist is the vector length of vectorized sums (K in the paper), and num_nodes is the number
                 of sum nodes in this layer.
        """
        raise NotImplementedError

    def _backtrack(self, dist_idx, node_idx, sample_idx, params, use_evidence=False, mode='sample', **kwargs):
        """
        Helper routine to implement EiNet backtracking, for sampling or MPE approximation.

        dist_idx, node_idx, sample_idx are lists of indices, all of the same length.

        :param dist_idx: list of indices, indexing into vectorized sums.
        :param node_idx: list of indices, indexing into node list of this layer.
        :param sample_idx: list of sample indices; representing the identity of the samples the EiNet is about to
                           generate. We need this, since not every SumLayer necessarily gets selected in the top-down
                           sampling process.
        :param params: sum-weights to use (Tensor).
        :param use_evidence: incorporate the bottom-up evidence (Bool)? For conditional sampling.
        :param mode: 'sample' or 'argmax'; for sampling or MPE approximation, respectively.
        :param kwargs: Additional keyword arguments.
        :return: depends on particular implementation.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------------------

    def default_initializer(self):
        """
        A simple initializer for normalized sum-weights.
        :return: initial parameters
        """
        params = 0.01 + 0.98 * torch.rand(self.params_shape)
        with torch.no_grad():
            if self.params_mask is not None:
                params.data *= self.params_mask
            params.data = params.data / (params.data.sum(self.normalization_dims, keepdim=True))
        return params

    def initialize(self, initializer='default'):
        """
        Initialize the parameters for this SumLayer.

        :param initializer: denotes the initialization method.
               If 'default' (str): use the default initialization, and store the parameters locally.
               If Tensor: provide custom initial parameters.
        :return: None
        """
        if initializer is None:
            self.params = None
        elif type(initializer) == str and initializer == 'default':
            if self._use_em:
                self.params = torch.nn.Parameter(self.default_initializer())
            else:
                self.params = torch.nn.Parameter(torch.randn(self.params_shape))
        elif type(initializer) == torch.Tensor:
            if initializer.shape != self.params_shape:
                raise AssertionError("Incorrect parameter shape.")
            self.params = torch.nn.Parameter(initializer)
        else:
            raise AssertionError("Unknown initializer.")

    def forward(self, x=None):
        """
        Evaluate this SumLayer.

        :param x: unused
        :return: tensor of log-densities. Must be of shape (batch_size, num_dist, num_nodes).
                 Here, num_dist is the vector length of vectorized sum nodes (K in the paper), and num_nodes is the
                 number of sum nodes in this layer.
        """
        if self._use_em:
            params = self.params
        else:
            reparam = self.reparam(self.params)
            params = reparam
        self._forward(params)

    def backtrack(self, dist_idx, node_idx, sample_idx, use_evidence=False, mode='sample', **kwargs):
        """
        Helper routine for backtracking in EiNets, see _sample(...) for details.
        """
        if mode != 'sample' and mode != 'argmax':
            raise AssertionError('Unknown backtracking mode {}'.format(mode))

        if self._use_em:
            params = self.params
        else:
            with torch.no_grad():
                params = self.reparam(self.params)
        return self._backtrack(dist_idx, node_idx, sample_idx, params, use_evidence, mode, **kwargs)

    def em_purge(self):
        """ Discard em statistics."""
        if self.params is not None:
            self.params.grad = None

    def em_process_batch(self):
        """
        Accumulate EM statistics of current batch. This should be called after call to backwards() on the output of
        the EiNet.
        """
        if not self._use_em:
            raise AssertionError("em_process_batch called while _use_em==False.")
        if self.params is None:
            return

        if self.online_em_frequency is not None:
            self._online_em_counter += 1
            if self._online_em_counter == self.online_em_frequency:
                self.em_update(True)
                self._online_em_counter = 0

    def em_update(self, _triggered=False):
        """
        Do an EM update. If the setting is online EM (online_em_stepsize is not None), then this function does nothing,
        since updates are triggered automatically. Thus, leave the private parameter _triggered alone.

        :param _triggered: for internal use, don't set
        :return: None
        """
        if not self._use_em:
            raise AssertionError("em_update called while _use_em==False.")
        if self.params is None:
            return

        if self.online_em_stepsize is not None and not _triggered:
            return

        with torch.no_grad():
            n = self.params.grad * self.params.data

            if self.online_em_stepsize is None:
                self.params.data = n
            else:
                s = self.online_em_stepsize
                p = torch.clamp(n, 1e-16)
                p = p / (p.sum(self.normalization_dims, keepdim=True))
                self.params.data = (1. - s) * self.params + s * p

            self.params.data = torch.clamp(self.params, 1e-16)
            if self.params_mask is not None:
                self.params.data *= self.params_mask
            self.params.data = self.params / (self.params.sum(self.normalization_dims, keepdim=True))
            self.params.grad = None

    def reparam(self, params_in):
        """
        Reparametrization function, transforming unconstrained parameters into valid sum-weight
        (non-negative, normalized).

        :params_in params: unconstrained parameters (Tensor) to be projected
        :return: re-parametrized parameters.
        """
        other_dims = tuple(i for i in range(len(params_in.shape)) if i not in self.normalization_dims)

        permutation = other_dims + self.normalization_dims
        unpermutation = tuple(c for i in range(len(permutation)) for c, j in enumerate(permutation) if j == i)

        numel = functools.reduce(lambda x, y: x * y, [params_in.shape[i] for i in self.normalization_dims])

        other_shape = tuple(params_in.shape[i] for i in other_dims)
        params_in = params_in.permute(permutation)
        orig_shape = params_in.shape
        params_in = params_in.reshape(other_shape + (numel,))
        out = softmax(params_in, -1)
        out = out.reshape(orig_shape).permute(unpermutation)
        return out

    def project_params(self, params):
        """Currently not required."""
        raise NotImplementedError


class EinsumLayer(SumLayer):
    """
    Implements the EinsumLayer.

    Consider the following simplified figure from the main paper:

        S   S   S     S
        |   |   |     |
        P   P   P     P
       / \ / \ / \   / \
      N   N  N   N  N   N

    Figure I


    Here, we see 6 input nodes (indicated with N), 4 product nodes (indicated with P) and 4 sum nodes (indicated
    with S). Recall that we assume that each N and S computes a vector of distributions of length vector_length (K in
    the paper. The N-nodes in this figure have already been computed by previous layers, and are stored in their
    corresponding log-density tensor of shape (batch_size, vector_length, num_nodes), where the last axis indexes nodes.
    Also note that each product has exactly 2 children -- this is an assumption in this paper, which is actually not
    too severe. Consequently, we can, for each product collect the "left" and "right" child nodes and construct two
    tensors of shape (batch_size, vector_length, 4) (4 because we have 4 sums in this example). These tensors are called
    self.left_child_log_prob and self.right_child_log_prob in forward(...).

    Then, the whole layer of sum nodes can be computed with:
        prob = torch.einsum('bip,bjp,ijop->bop', left_prob, right_prob, params)
    where params is a a 4-D tensor storing all sum-weights. This is essentially what happens in this layer, plus the
    so-called logEinsumExp trick, to ensure numerical stability (see paper, and forward(...)).


    Here, in this class, we assume that sum nodes have only 1 child (a product). In order to handle sum nodes with
    multiple children, the EinsumNetwork class constructs a EinsumMixingLayer following this layer, if necessary.
    Consider the example with multiple sum-children from the paper:

           S          S
        /  |  \      / \
       P    P  P    P  P
      /\   /\  /\  /\  /\
     N  N N N N N N N N N

    Figure II


    This example contains 2 sum nodes, 5 product nodes, and 10 input nodes (they might also be shared among products),
    where the first sum has 3 product children, and the second S has 2 product children. The strategy in EiNets is to
    re-write such structure into two layers of sum nodes:

            S          S
        /   |  \      / \
       S    S   S    S  S
       |    |   |    |  |
       P    P   P    P  P
      /\   /\  /\  /\  /\
     N  N N N N N N N N N

    Figure III


    The first sum layer has 5 nodes, each with a single product as child. The second layer then mixes over the sum
    nodes in the first layer (see paper for details).
    The first layer is computed by this class, while the second layer is computed by the EinsumMixingLayer -- see below.

    The important thing to note is that, regardless whether the PC layer has a-priori only sum nodes with single
    children, as in Figure I, or if it gets re-written as in Figure III, the EinsumLayer always computes as many nodes
    as there are *product nodes*. Thus, an argument to the constructor is the list of product nodes in this layer.
    """

    def __init__(self, graph, products, layers, use_em=True):

        self.products = products

        self.num_sums = set([n.num_dist for p in self.products for n in graph.pred[p]])
        if len(self.num_sums) != 1:
            raise AssertionError("Number of distributions must be the same for all parent nodes in one layer.")
        self.num_sums = list(self.num_sums)[0]

        self.num_input_dist = set([n.num_dist for p in self.products for n in graph.succ[p]])
        if len(self.num_input_dist) != 1:
            raise AssertionError("Number of input distributions must be the same for all child nodes in one layer.")
        self.num_input_dist = list(self.num_input_dist)[0]

        if any([len(graph.succ[p]) != 2 for p in self.products]):
            raise AssertionError("Only 2-partitions are currently supported.")

        param_shape = (self.num_input_dist, self.num_input_dist, self.num_sums, len(self.products))
        super(EinsumLayer, self).__init__(param_shape, normalization_dims=(0, 1), use_em=use_em)

        # get pairs of nodes which are input to the products (list of lists)
        # length of the outer list is same as self.products, length of inner lists is 2
        # "left child" has index 0, "right child" has index 1
        self.inputs = [sorted(graph.successors(p)) for p in self.products]

        # collect all layers which contain left/right children
        self.left_layers = [l for l in layers if any([i[0].einet_address.layer == l for i in self.inputs])]
        self.right_layers = [l for l in layers if any([i[1].einet_address.layer == l for i in self.inputs])]

        # The following code does some index bookkeeping, in order that we can gather the required data in forward(...).
        # Recall that in EiNets, each layer implements a log-density tensor of shape
        # (batch_size, vector_length, num_nodes).
        # We iterate over all previous left/right layers, and collect the node indices (corresponding to axis 2) in
        # self.idx_layer_i_child_j, where i indexes into self.left_layers for j==0, and into self.left_layers for j==1.
        # These index collections allow us to simply iterate over the previous layers and extract the required node
        # slices in forward.
        #
        # Furthermore, the following code also generates self.permutation_child_0 and self.permutation_child_1,
        # which are permutations of the left and right input nodes. We need this to get them in the same order as
        # assumed in self.products.
        def do_input_bookkeeping(layers, child_num):
            permutation = [None] * len(self.inputs)
            permutation_counter = count(0)
            for layer_counter, layer in enumerate(layers):
                cur_idx = []
                for c, input in enumerate(self.inputs):
                    if input[child_num].einet_address.layer == layer:
                        cur_idx.append(input[child_num].einet_address.idx)
                        if permutation[c] is not None:
                            raise AssertionError("This should not happen.")
                        permutation[c] = next(permutation_counter)
                self.register_buffer('idx_layer_{}_child_{}'.format(layer_counter, child_num), torch.tensor(cur_idx))
            if any(i is None for i in permutation):
                raise AssertionError("This should not happen.")
            self.register_buffer('permutation_child_{}'.format(child_num), torch.tensor(permutation))

        do_input_bookkeeping(self.left_layers, 0)
        do_input_bookkeeping(self.right_layers, 1)

        # when the EinsumLayer is followed by a EinsumMixingLayer, we produce a dummy "node" which outputs 0 (-inf in
        # log-domain) for zero-padding.
        self.dummy_idx = None

        # the dictionary mixing_component_idx stores which nodes (axis 2 of the log-density tensor) need to get mixed
        # in the following EinsumMixingLayer
        self.mixing_component_idx = {}

        for c, product in enumerate(self.products):
            # each product must have exactly 1 parent (sum node)
            node = list(graph.predecessors(product))
            assert len(node) == 1
            node = node[0]

            if len(graph.succ[node]) == 1:
                node.einet_address.layer = self
                node.einet_address.idx = c
            else:
                if node not in self.mixing_component_idx:
                    self.mixing_component_idx[node] = []
                self.mixing_component_idx[node].append(c)
                self.dummy_idx = len(self.products)

    def _forward(self, params):
        """
        EinsumLayer forward pass.
        """
        def cidx(layer_counter, child_num):
            return self.__getattr__('idx_layer_{}_child_{}'.format(layer_counter, child_num))

        # iterate over all layers which contain "left" nodes, get their indices; then, concatenate them to one tensor
        self.left_child_log_prob = torch.cat([l.prob[:, :,  cidx(c, 0)] for c, l in enumerate(self.left_layers)], 2)
        # get into the same order as assumed in self.products
        self.left_child_log_prob = self.left_child_log_prob[:, :, self.permutation_child_0]
        # ditto, for right "right" nodes
        self.right_child_log_prob = torch.cat([l.prob[:, :, cidx(c, 1)] for c, l in enumerate(self.right_layers)], 2)
        self.right_child_log_prob = self.right_child_log_prob[:, :, self.permutation_child_1]

        # We perform the LogEinsumExp trick, by first subtracting the maxes
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        # this is the central einsum operation
        prob = torch.einsum('bip,bjp,ijop->bop', left_prob, right_prob, params)

        # LogEinsumExp trick, re-add the max
        prob = torch.log(prob) + left_max + right_max

        # zero-padding (-inf in log-domain) for the following mixing layer
        if self.dummy_idx:
            prob = F.pad(prob, [0, 1], "constant", float('-inf'))

        self.prob = prob

    def _backtrack(self, dist_idx, node_idx, sample_idx, params, use_evidence=False, mode='sample', **kwargs):
        """
        Helper routine for backtracking in EiNets.

        Recall that each layer in an EiNet implements a tensor of log-densities of shape (see doc for EinsumNetwork)
            (batch_size, vector_length, num_nodes).
        Here dist_idx, node_idx, and sample_idx are lists of the same length,
        where dist_idx indexes into axis 1 and node_idx into axis 2. sample_idx indicates the sample number of samples
        to be created -- this is required since sampling in PCs by sampling a path from top to bottom (ancestral
        sampling), and not every EinsumLayer does necessarily get all samples.

        :param dist_idx: list of indices into axis 1 of log-density tensor
        :param node_idx: list of indices into axis 2 of log-density tensor
        :param sample_idx: global identifier of the sample to be produced
        :param params: parameters to be used for this layer
        :param use_evidence: using evidence form bottom-up pass? For conditional sampling.
        :param mode: 'sample' or 'argmax'; for sampling or MPE approximation, respectively.
        :param kwargs: other keyword arguments
        :return: selected layers and indices below
        """
        with torch.no_grad():
            if use_evidence:
                log_prior = torch.log(params[:, :, dist_idx, node_idx])
                log_prior = log_prior.permute(2, 0, 1)
                left_log_prob = self.left_child_log_prob[sample_idx, :, node_idx].unsqueeze(2)
                right_log_prob = self.right_child_log_prob[sample_idx, :, node_idx].unsqueeze(1)
                log_posterior = log_prior + left_log_prob + right_log_prob
                log_posterior = log_posterior.reshape(log_posterior.shape[0], -1)
                posterior = torch.exp(log_posterior - torch.logsumexp(log_posterior, 1, keepdim=True))
            else:
                posterior = params[:, :, dist_idx, node_idx]
                posterior = posterior.permute(2, 0, 1)
                posterior = posterior.reshape(posterior.shape[0], -1)

            if mode == 'sample':
                idx = sample_matrix_categorical(posterior)
            elif mode == 'argmax':
                idx = torch.argmax(posterior, -1)

            dist_idx_right = idx % self.num_input_dist
            dist_idx_left = idx // self.num_input_dist
            node_idx_left = [self.inputs[i][0].einet_address.idx for i in node_idx]
            node_idx_right = [self.inputs[i][1].einet_address.idx for i in node_idx]
            layers_left = [self.inputs[i][0].einet_address.layer for i in node_idx]
            layers_right = [self.inputs[i][1].einet_address.layer for i in node_idx]

        return dist_idx_left, dist_idx_right, node_idx_left, node_idx_right, layers_left, layers_right


class EinsumMixingLayer(SumLayer):
    """
    Implements the Mixing Layer, in order to handle sum nodes with multiple children.
    Recall Figure II from above:

           S          S
        /  |  \      / \
       P   P  P     P  P
      /\   /\  /\  /\  /\
     N  N N N N N N N N N

    Figure II


    We implement such excerpt as in Figure III, splitting sum nodes with multiple children in a chain of two sum nodes:

            S          S
        /   |  \      / \
       S    S   S    S  S
       |    |   |    |  |
       P    P   P    P  P
      /\   /\  /\  /\  /\
     N  N N N N N N N N N

    Figure III


    The input nodes N have already been computed. The product nodes P and the first sum layer are computed using an
    EinsumLayer, yielding a log-density tensor of shape
        (batch_size, vector_length, num_nodes).
    In this example num_nodes is 5, since the are 5 product nodes (or 5 singleton sum nodes). The EinsumMixingLayer
    then simply mixes sums from the first layer, to yield 2 sums. This is just an over-parametrization of the original
    excerpt.
    """

    def __init__(self, graph, nodes, einsum_layer, use_em):
        """
        :param graph: the PC graph (see Graph.py)
        :param nodes: the nodes of the current layer (see constructor of EinsumNetwork), which have multiple children
        :param einsum_layer:
        :param use_em:
        """

        self.nodes = nodes

        self.num_sums = set([n.num_dist for n in self.nodes])
        if len(self.num_sums) != 1:
            raise AssertionError("Number of distributions must be the same for all regions in one layer.")
        self.num_sums = list(self.num_sums)[0]

        self.max_components = max([len(graph.succ[n]) for n in self.nodes])
        # einsum_layer is actually the only layer which gives input to EinsumMixingLayer
        # we keep it in a list, since otherwise it gets registered as a torch sub-module
        self.layers = [einsum_layer]
        self.mixing_component_idx = einsum_layer.mixing_component_idx

        if einsum_layer.dummy_idx is None:
            raise AssertionError('EinsumLayer has not set a dummy index for padding.')

        param_shape = (self.num_sums, len(self.nodes), self.max_components)

        # The following code does some bookkeeping.
        # padded_idx indexes into the log-density tensor of the previous EinsumLayer, padded with a dummy input which
        # outputs constantly 0 (-inf in the log-domain), see class EinsumLayer.
        padded_idx = []
        params_mask = torch.ones(param_shape)
        for c, node in enumerate(self.nodes):
            num_components = len(self.mixing_component_idx[node])
            padded_idx += self.mixing_component_idx[node]
            padded_idx += [einsum_layer.dummy_idx] * (self.max_components - num_components)
            if self.max_components > num_components:
                params_mask[:, c, num_components:] = 0.0
            node.einet_address.layer = self
            node.einet_address.idx = c

        super(EinsumMixingLayer, self).__init__(param_shape,
                                                normalization_dims=(2,),
                                                use_em=use_em,
                                                params_mask=params_mask)

        self.register_buffer('padded_idx', torch.tensor(padded_idx))

    def _forward(self, params):
        self.child_log_prob = self.layers[0].prob[:, :, self.padded_idx]
        self.child_log_prob = self.child_log_prob.reshape((self.child_log_prob.shape[0],
                                                           self.child_log_prob.shape[1],
                                                           len(self.nodes),
                                                           self.max_components))

        max_p = torch.max(self.child_log_prob, 3, keepdim=True)[0]
        prob = torch.exp(self.child_log_prob - max_p)

        output = torch.einsum('bonc,onc->bon', prob, params)

        self.prob = torch.log(output) + max_p[:, :, :, 0]

    def _backtrack(self, dist_idx, node_idx, sample_idx, params, use_evidence=False, mode='sample', **kwargs):
        """Helper routine for backtracking in EiNets."""
        with torch.no_grad():
            if use_evidence:
                log_prior = torch.log(params[dist_idx, node_idx, :])
                log_posterior = log_prior + self.child_log_prob[sample_idx, dist_idx, node_idx, :]
                posterior = torch.exp(log_posterior - torch.logsumexp(log_posterior, 1, keepdim=True))
            else:
                posterior = params[dist_idx, node_idx, :]

            if mode == 'sample':
                idx = sample_matrix_categorical(posterior)
            elif mode == 'argmax':
                idx = torch.argmax(posterior, -1)
            dist_idx_out = dist_idx
            node_idx_out = [self.mixing_component_idx[self.nodes[i]][idx[c]] for c, i in enumerate(node_idx)]
            layers_out = [self.layers[0]] * len(node_idx)

        return dist_idx_out, node_idx_out, layers_out
