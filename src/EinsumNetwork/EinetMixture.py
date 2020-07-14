import torch
import numpy as np
from scipy.special import logsumexp
from EinsumNetwork.EinsumNetwork import log_likelihoods


class EinetMixture:
    """A simple class for mixtures of Einets, implemented in numpy."""

    def __init__(self, p, einets):

        if len(p) != len(einets):
            raise AssertionError("p and einets must have the same length.")

        self.num_components = len(p)

        self.p = p
        self.einets = einets

        num_var = set([e.args.num_var for e in einets])
        if len(num_var) != 1:
            raise AssertionError("all EiNet components must have the same num_var.")
        self.num_var = list(num_var)[0]

        num_dims = set([e.args.num_dims for e in einets])
        if len(num_dims) != 1:
            raise AssertionError("all EiNet components must have the same num_dims.")
        self.num_dims = list(num_dims)[0]

    def sample(self, N, **kwargs):
        samples = np.zeros((N, self.num_var, self.num_dims))
        for k in range(N):
            rand_idx = np.sum(np.random.rand() > np.cumsum(self.p[0:-1]))
            samples[k, ...] = self.einets[rand_idx].sample(num_samples=1, **kwargs).cpu().numpy()
        return samples

    def conditional_sample(self, x, marginalize_idx, **kwargs):
        marginalization_backup = []
        component_posterior = np.zeros((self.num_components, x.shape[0]))
        for einet_counter, einet in enumerate(self.einets):
            marginalization_backup.append(einet.get_marginalization_idx())
            einet.set_marginalization_idx(marginalize_idx)
            lls = einet.forward(x)
            lls = lls.sum(1)
            component_posterior[einet_counter, :] = lls.detach().cpu().numpy() + np.log(self.p[einet_counter])

        component_posterior = component_posterior - logsumexp(component_posterior, 0, keepdims=True)
        component_posterior = np.exp(component_posterior)

        samples = np.zeros((x.shape[0], self.num_var, self.num_dims))
        for test_idx in range(x.shape[0]):
            component_idx = np.argmax(component_posterior[:, test_idx])
            sample = self.einets[component_idx].sample(x=x[test_idx:test_idx + 1, :], **kwargs)
            samples[test_idx, ...] = sample.squeeze().cpu().numpy()

        # restore the original marginalization indices
        for einet_counter, einet in enumerate(self.einets):
            einet.set_marginalization_idx(marginalization_backup[einet_counter])

        return samples

    def log_likelihood(self, x, labels=None, batch_size=100):
        with torch.no_grad():
            idx_batches = torch.arange(0, x.shape[0], dtype=torch.int64, device=x.device).split(batch_size)
            ll_total = 0.0
            for batch_count, idx in enumerate(idx_batches):
                batch_x = x[idx, :]
                if labels is not None:
                    batch_labels = labels[idx]
                else:
                    batch_labels = None

                lls = torch.zeros(len(idx), self.num_components, device=x.device)
                for einet_count, einet in enumerate(self.einets):
                    outputs = einet(batch_x)
                    lls[:, einet_count] = log_likelihoods(outputs, labels=batch_labels).squeeze()
                    lls[:, einet_count] -= torch.log(torch.tensor(self.p[einet_count]))
                lls = torch.logsumexp(lls, dim=1)
                ll_total += lls.sum().item()
            return ll_total
