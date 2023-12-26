import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Callable, Optional, Union, List, Dict, Any
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.nn.models import MetaLayer
import torch.nn as nn

class EdgeModel(torch.nn.Module):
    
    def __init__(self, node_emb_size: int, edge_emb_size: int, hidden_emb_size: int, fe_layers: int,fe_activation: Callable[[torch.Tensor],torch.Tensor]  = torch.nn.Softplus(beta=1, threshold=20)
                ):
        super().__init__()
        fe_features  =[node_emb_size]+[hidden_emb_size]*fe_layers+[edge_emb_size]
        fe_mlp=[]
        
        for i in range(len(fe_features)-2):
            fe_mlp+=[nn.Linear(in_features=fe_features[i],out_features=fe_features[i+1]),fe_activation]
        fe_mlp+=[nn.Linear(in_features=fe_features[-2],out_features=fe_features[-1])]
        
        self.fe_mlp=nn.Sequential(*fe_mlp)
    def forward(self, src, dest, edge_attr, u=None, batch=None):
        c2ij = src * dest
        out = self.fe_mlp(c2ij)
        out = out + edge_attr
        return out
    
class NodeModel(torch.nn.Module):
    def __init__(self, node_emb_size: int, edge_emb_size: int, hidden_emb_size: int, fv_layers: int,fv_activation: Callable[[torch.Tensor],torch.Tensor]  = torch.nn.Softplus(beta=1, threshold=20)
                ):
        super(NodeModel, self).__init__()
        
        fv1_features  =[node_emb_size+edge_emb_size]+[hidden_emb_size]*fv_layers+[node_emb_size]
        
        fv1_mlp=[]
        for i in range(len(fv1_features)-2):
            fv1_mlp+=[nn.Linear(in_features=fv1_features[i],out_features=fv1_features[i+1]),fv_activation]
        fv1_mlp+=[nn.Linear(in_features=fv1_features[-2],out_features=fv1_features[-1])]
        
        self.fv1_mlp=nn.Sequential(*fv1_mlp)
        
    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        s, r = edge_index
        c1ij = torch.cat([x[r], edge_attr], dim=1)
        out = self.fv1_mlp(c1ij)
        return x+ scatter(out, r, dim=0, dim_size=x.size(0), reduce='sum')

class HGNN(torch.nn.Module):
    def __init__(self,
                in_edge_feats:int,
                in_node_feats:int,
                in_type_ohe_size: int,
                edge_emb_size:int,
                node_emb_size: int,
                hidden_emb_size: int,
                use_ke_model:bool,
                fa_layers: int,
                fb_layers: int,
                fv_layers: int,
                fe_layers: int,
                kemlp_layers: int,
                MLP1_layers: int,
                message_passing_steps: int,
                fa_activation: Callable[[torch.Tensor],torch.Tensor]  = torch.nn.Softplus(beta=1, threshold=20),
                fb_activation: Callable[[torch.Tensor],torch.Tensor]  = torch.nn.Softplus(beta=1, threshold=20),
                MLP1_activation:   Callable[[torch.Tensor],torch.Tensor] = torch.nn.Softplus(beta=1, threshold=20),
                kemlp_activation:   Callable[[torch.Tensor],torch.Tensor] = torch.nn.Softplus(beta=1, threshold=20),
                MLP2_activation:   Callable[[torch.Tensor],torch.Tensor] = torch.nn.Softplus(beta=1, threshold=20),
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        kem=True
        fa_features  =[in_node_feats]+[node_emb_size]*fa_layers
        fb_features  =[in_edge_feats]+[edge_emb_size]*fb_layers
        kemlp_features=[node_emb_size]+[hidden_emb_size]*kemlp_layers+[1]
        MLP1_features=[edge_emb_size]+[hidden_emb_size]*MLP1_layers+[1]
        MLP2_features=[in_type_ohe_size, 5, 5, 1]
        
        self.message_passing_steps=message_passing_steps
        
        fa_mlp=[]
        for i in range(len(fa_features)-2):
            fa_mlp+=[nn.Linear(in_features=fa_features[i],out_features=fa_features[i+1]),fa_activation]
        fa_mlp+=[nn.Linear(in_features=fa_features[-2],out_features=fa_features[-1])]
        
        fb_mlp=[]
        for i in range(len(fb_features)-2):
            fb_mlp+=[nn.Linear(in_features=fb_features[i],out_features=fb_features[i+1]),fb_activation]
        fb_mlp+=[nn.Linear(in_features=fb_features[-2],out_features=fb_features[-1])]
        
        ke_mlp=[]
        for i in range(len(kemlp_features)-2):
            ke_mlp+=[nn.Linear(in_features=kemlp_features[i], out_features=kemlp_features[i + 1]), kemlp_activation]
        ke_mlp+=[nn.Linear(in_features=kemlp_features[-2],out_features=kemlp_features[-1])]

        MLP1=[]
        for i in range(len(MLP1_features)-2):
            MLP1+=[nn.Linear(in_features=MLP1_features[i],out_features=MLP1_features[i+1]),MLP1_activation]
        MLP1+=[nn.Linear(in_features=MLP1_features[-2],out_features=MLP1_features[-1])]
        
        MLP2=[]
        for i in range(len(MLP2_features)-2):
            MLP2+=[nn.Linear(in_features=MLP2_features[i],out_features=MLP2_features[i+1]),MLP2_activation]
        MLP2+=[nn.Linear(in_features=MLP2_features[-2],out_features=MLP2_features[-1]),MLP2_activation]
        
        self.kem=use_ke_model
        self.fa_mlp=nn.Sequential(*fa_mlp)
        self.fb_mlp=nn.Sequential(*fb_mlp)
        self.ke_mlp=nn.Sequential(*ke_mlp)
        self.MLP1=nn.Sequential(*MLP1)
        self.MLP2=nn.Sequential(*MLP2)
        self.GNNConv=MetaLayer(edge_model=EdgeModel(node_emb_size,edge_emb_size,hidden_emb_size,fe_layers),
                               node_model=NodeModel(node_emb_size,edge_emb_size,hidden_emb_size,fv_layers),
                               global_model=None
                                )
        
    def forward(self,Inp_Graph : Data):
        Graph=Inp_Graph.clone()
        #fa
        def initial_node_emb_fn(nodes):#fb
            return self.fa_mlp(nodes)
        
        #fb
        def initial_edge_emb_fn(edges):#fb
            return self.fb_mlp(edges)
        
        #kemlp
        def node_to_ke(node_attr):
            if self.kem:
                t=self.ke_mlp(node_attr)
            else:
                t=0.5*(Graph["velocity"]**2).sum(axis=1)*Graph["mass"]
            return t

        #MLP1
        def edge_node_to_pe(edge_attr):
            fij = self.MLP1(edge_attr)
            return fij
        
        #1 Create_initial_node_emb
        Graph['x']=initial_node_emb_fn(Graph['x'])
        
        #2 Create initial edge emb
        Graph['edge_attr']=initial_edge_emb_fn(Graph['edge_attr'])
        
        #3 Message Passing
        for k in range(self.message_passing_steps):
            Graph['x'],Graph['edge_attr'],_=self.GNNConv(Graph['x'],Graph['edge_index'],Graph['edge_attr'])

        return edge_node_to_pe(Graph['edge_attr']), node_to_ke(Graph['node_vel_emb'])
