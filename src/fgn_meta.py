import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
import torch.nn as nn

class EdgeModel(torch.nn.Module):
    def __init__(self, node_emb_size: int, edge_emb_size: int, hidden_emb_size: int, fe_layers: int,fe_activation: Callable[[torch.Tensor],torch.Tensor]  = torch.nn.Softplus(beta=1, threshold=20)):
        super().__init__()
        fe_features  =[node_emb_size]+[hidden_emb_size]*fe_layers+[edge_emb_size]
        fe_mlp=[]
        
        for i in range(len(fe_features)-2):
            fe_mlp+=[nn.Linear(in_features=fe_features[i],out_features=fe_features[i+1]),fe_activation]
        fe_mlp+=[nn.Linear(in_features=fe_features[-2],out_features=fe_features[-1])]
        
        self.edge_mlp = Seq(*fe_mlp)
    
    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dst, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp_1 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
        self.node_mlp_2 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0),
                      reduce='mean')
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([
            u,
            scatter(x, batch, dim=0, reduce='mean'),
        ], dim=1)
        return self.global_mlp(out)



op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)