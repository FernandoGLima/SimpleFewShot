import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# DSN: Adaptive Subspaces for Few-Shot Learning
# https://openaccess.thecvf.com/content_CVPR_2020/papers/Simon_Adaptive_Subspaces_for_Few-Shot_Learning_CVPR_2020_paper.pdf
# Based on the open-source implementation of https://github.com/chrysts/dsn_fewshot/tree/master

class DSN(nn.Module):
    def __init__(self, backbone):
        super(DSN, self).__init__()
        self.backbone = backbone

    def forward(self, train_imgs, train_labels, query_imgs, query_labels, normalize=False):
        """
        Constructs the subspace representation of each class(=mean of support vectors of each class) and
        returns the classification score (=L2 distance to each class prototype) on the query set.

            Our algorithm using subspaces here

        Parameters:
            query:  a (tasks_per_batch, n_query, d) Tensor.
            support:  a (tasks_per_batch, n_support, d) Tensor.
            support_labels: a (tasks_per_batch, n_support) Tensor.
            n_way: a scalar. Represents the number of classes in a few-shot classification task.
            n_shot: a scalar. Represents the number of support examples given per class.
            normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """
        self.n_way = train_labels.size(1)
        self.n_shot = train_labels.size(0) // self.n_way
        
        support_features = self.backbone.forward(train_imgs)
        query_features = self.backbone.forward(query_imgs)

        d = query_imgs.size(1)
        
        class_representatives = []
        for nn in range(self.n_way):
            idxss = train_labels[:, nn].bool()
            class_support = support_features[idxss]
            class_support_t = class_support.transpose(0, 1) 
            class_representatives.append(class_support_t)

        dist = []
        for class_support_t in class_representatives:
            uu, _, _ = torch.svd(class_support_t.double())
            uu = uu.float()
            subspace = uu[:, :self.n_shot-1].transpose(0, 1)
            projection = subspace.transpose(0, 1).mm(subspace.mm(query_features.transpose(0, 1))).transpose(0, 1)
            dist_perclass = torch.sum((query_features - projection)**2, dim=-1)
            dist.append(dist_perclass)

        dist = torch.stack(dist, dim=1)
        logits = -dist

        if normalize:
            logits = logits / d

        logits = F.log_softmax(logits.reshape(-1, self.n_way), dim=1)

        return logits