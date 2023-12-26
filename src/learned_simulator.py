import torch
import torch.nn as nn
import numpy as np
from src import graph_network
from typing import Dict
from torch_geometric.data import Data
from typing import Callable

class FGN(nn.Module):
  
  def __init__(
          self,
          in_edge_feats:int,
          in_node_feats:int,
          in_type_ohe_size: int,
          edge_emb_size:int,
          node_emb_size: int,
          hidden_emb_size: int,
          nmlp_layers: int, #fa_layers, fb_layers, fv_layers, fe_layers
          MLP1_layers: int,
          MLP1_F_out_dim: int,  # Dimensionality of the problem.
          message_passing_steps: int,
          MLP1_activation:   Callable[[torch.Tensor],torch.Tensor] = torch.nn.Softplus(beta=1, threshold=20),
          MLP2_activation:   Callable[[torch.Tensor],torch.Tensor] = torch.nn.Softplus(beta=1, threshold=20),
          ):
    """Initializes the model.
    
    Args:
      in_node_feats: Number of node inputs.
      in_edge_feats: Number of edge inputs.
      latent_dim (edge_emb_size): Size of latent dimension
      nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
      MLP1_F_out_dim: Dimensionality of the problem.
      message_passing_steps: Number of message passing steps.
      device: Runtime device (cuda or cpu).
    
    """
    super(FGN, self).__init__()
    
    MLP1_features=[node_emb_size]+[hidden_emb_size]*MLP1_layers+[MLP1_F_out_dim]
    MLP2_features=[in_type_ohe_size, 10, 5, 1]
    
    MLP1=[]
    for i in range(len(MLP1_features)-2):
      MLP1+=[nn.Linear(in_features=MLP1_features[i],out_features=MLP1_features[i+1]),MLP1_activation]
    MLP1+=[nn.Linear(in_features=MLP1_features[-2],out_features=MLP1_features[-1])]
    
    MLP2=[]
    for i in range(len(MLP2_features)-2):
      MLP2+=[nn.Linear(in_features=MLP2_features[i],out_features=MLP2_features[i+1]),MLP2_activation]
    MLP2+=[nn.Linear(in_features=MLP2_features[-2],out_features=MLP2_features[-1]),MLP2_activation]
    
    # Initialize the EncodeProcessDecode
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=in_node_feats,
        nnode_out_features=node_emb_size,
        nedge_in_features=in_edge_feats,
        latent_dim=edge_emb_size,
        nmessage_passing_steps=message_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=hidden_emb_size)
    
    self.MLP1=nn.Sequential(*MLP1)
    self.MLP2=nn.Sequential(*MLP2)
  
  def forward(self, Inp_Graph : Data):
    Graph=Inp_Graph.clone()
    
    def emb_to_force(node_attr):
      return self.MLP1(node_attr)
    
    def species_to_gamma(type):
      return self.MLP2(type)
    
    node_features, edge_index, edge_features = Graph['x'], Graph['edge_index'], Graph['edge_attr']
    out = self._encode_process_decode(node_features, edge_index, edge_features)
    return emb_to_force(out), species_to_gamma(Graph['type'])

