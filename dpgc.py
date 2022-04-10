import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric as tg
from lib.functions import ReverseLayerF


class PEWE(nn.Module):
    def __init__(self, non_image_dim=3, embedding_dim=256, dropout=0.1):
        super(PEWE, self).__init__()
        hidden = 128
        self.parser = nn.Sequential(
            nn.Linear(non_image_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden, bias=True),
        )

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.non_image_dim = non_image_dim
        self.embedding_dim = embedding_dim
        self.model_init()

    def forward(self, non_image_x, ess):
        # ess.shape = [871, 256]
        x1_non_image = non_image_x[:, 0:self.non_image_dim]  # shape [9149, 128]
        x1_es = ess[:, 0:self.embedding_dim]
        x2_non_image = non_image_x[:, self.non_image_dim:]  # shape [9149, 128]
        x2_es = ess[:, self.embedding_dim:]
        x1_non_image = self.parser(x1_non_image)
        x2_non_image = self.parser(x2_non_image)
        h1 = torch.cat((x1_non_image, x1_es), dim=1)
        h2 = torch.cat((x2_non_image, x2_es), dim=1)
        p = (self.cos(h1, h2) + 1) * 0.5
        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True


class DPGC(torch.nn.Module):
    def __init__(self, config, input_dim=2000, embedding_dim=512, gcn_input_dim=2000, gcn_hidden_filters=16, gcn_layers=4,
                 dropout=0.2, num_classes=2, non_image_dim=3, edge_dropout=0.5):
        super(DPGC, self).__init__()
        self.config = config
        K = 3  # K-nearest neighbors
        self.dropout = dropout  # for non-image feature transformation
        self.edge_dropout = edge_dropout  # for population gcn
        self.gcn_layers = gcn_layers  #
        self.embedding_dim = embedding_dim
        self.relu = nn.ReLU(inplace=True)
        bias = False  # for ChebConv
        # disentangling modules
        self.EI = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.ES = nn.Linear(in_features=input_dim, out_features=embedding_dim)

        # ei 和 es concat
        self.DE = nn.Linear(in_features=embedding_dim * 2, out_features=gcn_input_dim)

        # gcn
        self.gconv = nn.ModuleList()
        hidden = [gcn_hidden_filters for _ in range(gcn_layers)]
        for i in range(gcn_layers):
            in_channels = gcn_input_dim if i == 0 else hidden[i - 1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias))

        # label classification
        cls_input_dim = sum(hidden)
        self.label_clf = nn.Sequential(
            nn.Linear(cls_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout),
            nn.Linear(256, num_classes))

        # site classification
        self.site_clf = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout),
            nn.Linear(256, 20)
        )

        # edge_net for constructing population graph
        self.edge_net = PEWE(non_image_dim=non_image_dim, embedding_dim=embedding_dim, dropout=self.dropout)
        self.model_init()

    def forward(self, image_features, edge_index, non_image_features, enforce_edropout=False):
        if not self.config.device:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda:1")
        if self.edge_dropout > 0:
            if enforce_edropout or self.training:
                one_mask = torch.ones(
                    [non_image_features.shape[0], 1])  # non-image-features.shape = torch.Size([18392, 6])
                drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                bool_mask = torch.squeeze(drop_mask.type(torch.bool))
                edge_index = edge_index[:, bool_mask]  # dropout掉一部分，torch.Size([2, 9159])
                non_image_features = non_image_features[bool_mask]  # torch.Size([9159, 6])
        ei = self.EI(image_features)  # torch.Size([871, 256])
        # todo(tdye): 把es从computation graph中剥离掉，只用于building population graph，就是把es当成一个从图像中抽出的一个“非成像特征”
        es = self.ES(image_features).detach_()  # torch.Size([871, 256]) todo(tdye): add sparsity regularization

        # 制作和non-image特征相似的tensor
        # 可以转化成矩阵运算加快速度
        # print(edge_index)
        flatten_id = 0
        ES_M = torch.zeros((edge_index.shape[1], 2 * self.embedding_dim))
        # todo(tdye): for diff?
        for i in range(edge_index.shape[1]):
            # print(edge_index[:, i])
            source = edge_index[0, i]
            target = edge_index[1, i]
            ES_M[flatten_id] = torch.cat((es[source], es[target]), dim=0)
            flatten_id += 1
        # print(ES_M.shape)
        ES_M = ES_M.to(device)
        edge_weight = torch.squeeze(self.edge_net(non_image_features, ES_M))

        # decoded feature, regularize disentanglement
        concat_features = torch.cat((ei, es), dim=1)
        # todo(tdye): debug 是 decoded后的特征无法代表image_features还是构图效果不好？
        reconstruct_features = self.DE(concat_features)  # [871, 2000]
        # reconstruct_features = image_features

        # dropout todo(tdye): why?
        ei = F.dropout(ei, self.dropout, self.training)
        h = self.relu(self.gconv[0](ei, edge_index, edge_weight))
        h0 = h

        # todo(tdye): 聚合邻居信息时，接受近的邻居的concat_features，接收远的邻居的de——noised message (ei)
        for i in range(1, self.gcn_layers):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_weight))
            jk = torch.cat((h0, h), axis=1)
            h0 = jk
        label_logits = self.label_clf(jk)  # 871 x 2

        # site classification
        # todo(tdye): tuning alpha
        reversed_si = ReverseLayerF.apply(ei, 1)
        site_logits = self.site_clf(reversed_si)
        # return
        # ei to regulate it to sparsity
        return label_logits, site_logits, es, reconstruct_features

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
