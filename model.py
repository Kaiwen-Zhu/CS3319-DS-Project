import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import dgl.function as dglfn


class WeightedGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, device, edge_weight=None):
        super(WeightedGraphConv, self).__init__()
        self.conv = dglnn.GraphConv(in_feats, out_feats, norm='none', allow_zero_in_degree=True)
        self.edge_weight = edge_weight
        self.norm = dglnn.EdgeWeightNorm(norm='right')
        self.device = device

    def forward(self, g, inputs):
        norm_weight = self.norm(g, torch.ones(g.number_of_edges(), device=self.device))
        if self.edge_weight is not None:
            if g.is_block:
                ids = g.edata[dgl.EID]
                weight = self.edge_weight[ids]
            else:
                weight = self.edge_weight
            norm_weight *= weight
        return self.conv(g, inputs, edge_weight=norm_weight)


class RGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat1, hidden_feat2, hidden_feat3, out_feat,
                 rel_names, device, pred_weight):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv(dict(
                {rel[1]: WeightedGraphConv(in_feat, hidden_feat1, device) for rel in rel_names[:4]}, 
                **{rel[1]: WeightedGraphConv(in_feat, hidden_feat1, device, pred_weight[rel[1]]) 
                for rel in rel_names[4:]}))
        self.conv2 = dglnn.HeteroGraphConv(dict(
                {rel[1]: WeightedGraphConv(hidden_feat1, hidden_feat2, device) for rel in rel_names[:4]}, 
                **{rel[1]: WeightedGraphConv(hidden_feat1, hidden_feat2, device, pred_weight[rel[1]]) 
                for rel in rel_names[4:]}))
        self.conv3 = dglnn.HeteroGraphConv(dict(
                {rel[1]: WeightedGraphConv(hidden_feat2, hidden_feat3, device) for rel in rel_names[:4]}, 
                **{rel[1]: WeightedGraphConv(hidden_feat2, hidden_feat3, device, pred_weight[rel[1]]) 
                for rel in rel_names[4:]}))
        self.conv4 = dglnn.HeteroGraphConv(dict(
                {rel[1]: WeightedGraphConv(hidden_feat3, out_feat, device) for rel in rel_names[:4]}, 
                **{rel[1]: WeightedGraphConv(hidden_feat3, out_feat, device, pred_weight[rel[1]]) 
                for rel in rel_names[4:]}))
        self.LeakyReLU = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(hidden_feat1)
        self.bn2 = nn.BatchNorm1d(hidden_feat2)
        self.bn3 = nn.BatchNorm1d(hidden_feat3)

    def forward(self, blocks, x):
        out1 = self.conv1(blocks[0], x)
        for k in out1.keys():
            out1[k] = self.bn1(out1[k])
            out1[k] = self.LeakyReLU(out1[k])

        out2 = self.conv2(blocks[1], out1)
        for k in out2.keys():
            out2[k] = self.bn2(out2[k])
            out2[k] = self.LeakyReLU(out2[k])

        out3 = self.conv3(blocks[2], out2)
        for k in out3.keys():
            out3[k] = self.bn3(out3[k])
            out3[k] = self.LeakyReLU(out3[k])
            
        out4 = self.conv4(blocks[3], out3)
        return out4


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, h):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(dglfn.u_dot_v('h', 'h', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class GNN(nn.Module):
    def __init__(self, in_features, hidden_features1, hidden_features2, hidden_features3, 
                    out_features, etypes, device, pred_weight=None):
        super().__init__()
        self.rgcn = RGCN(
            in_features, hidden_features1, hidden_features2, hidden_features3,
             out_features, etypes, device, pred_weight).to(device)
        self.pred = ScorePredictor()

    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.rgcn(blocks, x)
        pos_score = self.pred(positive_graph, x)
        neg_score = self.pred(negative_graph, x)
        return pos_score, neg_score

    def loss_fn(self, pos_score, neg_score, etype):
        n_edges = pos_score[etype].shape[0]
        return (3 - pos_score[etype].unsqueeze(1) + neg_score[etype].view(n_edges, -1)).clamp(min=0).mean()
