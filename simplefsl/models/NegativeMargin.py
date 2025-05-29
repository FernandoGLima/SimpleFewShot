import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import Parameter, functional as F

class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale_factor: norm of input feature
        margin: margin
    :return： (theta) - m
    """

    def __init__(self, out_features, in_features=2048, scale_factor=30.0, margin=0.40):
        super(AddMarginProduct, self).__init__()
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        device = feature.device
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight.to(device)))

        # when test, no label, just return
        if label is None:
            return cosine * self.scale_factor

        phi = cosine - self.margin
        one_hot_labels = label.to(torch.bool)
        output = torch.where(one_hot_labels, phi, cosine)
        output *= self.scale_factor

        return output


class NegativeMargin(nn.Module):
    def __init__(self, backbone):
        super(NegativeMargin, self).__init__()
        self.backbone = backbone

    def forward(self, train_imgs, train_labels, query_imgs, flag=False):
        self.n_way = train_labels.size(1)
        self.n_support = train_labels.size(0) // self.n_way

        support_features = self.backbone.forward(train_imgs)
        query_features = self.backbone.forward(query_imgs)

        linear_clf = AddMarginProduct(self.n_way).cuda()

        finetune_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        loss_function = nn.CrossEntropyLoss().cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support
         
        for _ in range(50):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                selected_id = torch.from_numpy(rand_id[i: min(i+batch_size, support_size)]).to(support_features.device)
                z_batch = support_features[selected_id]
                y_batch = train_labels[selected_id]

                scores = linear_clf(z_batch, y_batch)
                loss = loss_function(scores, y_batch)

                finetune_optimizer.zero_grad()
                if flag:
                    loss.backward(retain_graph=True)
                finetune_optimizer.step()

        scores = linear_clf(query_features)
        return scores