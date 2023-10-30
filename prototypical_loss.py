# coding=utf-8
import torch
import numpy as np
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def contrastive_sensitivity(prototypes, query_samples):
    """
    Calculates the contrastive sensitivity of a prototype network for matching tasks.

    Args:
    prototypes: a N x 2 x D numpy array, where N is the number of prototype pairs, 2 is the number of prototypes in each pair, and D is the dimensionality of each prototype.
    query_samples: a M x 2 x D numpy array, where M is the number of sample pairs, 2 is the number of samples in each pair, and D is the dimensionality of each sample.

    Returns:
    Contrastive sensitivity, a numpy array of length M, where each element corresponds to the contrastive sensitivity of a sample pair.
    """
    # Calculate the center or centroid of each prototype
    prototype_centers = np.mean(prototypes, axis=1)

    # Calculate the Euclidean distance between each sample pair
    sample_distances = np.linalg.norm(query_samples[:,0,:] - query_samples[:,1,:], axis=1)

    # Calculate the Euclidean distance between each prototype pair
    prototype_distances = np.linalg.norm(prototypes[:,0,:] - prototypes[:,1,:], axis=1)

    # Calculate contrastive sensitivity
    epsilon = 1e-6 # to avoid division by zero
    cs = prototype_distances / (sample_distances + epsilon)

    return cs



def prototypical_loss(input, target, n_support, output = False):
    '''

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    - output : evaluate model
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    # support_idxs class prototypes
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs]) 
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]

    ### prototypes compare  set a threshold predict new class
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    
    if output :
        with open('output/labels.txt', 'a') as f:
            for label1, label2 in zip(y_hat.numpy().flatten(), target_inds.squeeze(2).numpy().flatten()):
                f.write(f"{label1}\t{label2}\n")
            f.close()
    return loss_val,  acc_val
