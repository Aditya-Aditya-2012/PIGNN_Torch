import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import sys
MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8
from src.Systems import lennard_jones


class graph_loader:
    def create_G(self, R, V, A, species, pair_cutoffs_G, Disp_Vec_fn, Node_energy_fn):
            """
            R: Node Positions
            R_next: Node Positions at next step
            Disp_vec_fn: Calculates distance between atoms considering periodic boundaries
            species: Node type info 0 and 1
            cutoffs: pair cutoffs (N,N) shape
            sigma  : pair sigma   (N,N)
                """
            #1: Calculate pair distances
            
            dR_pair = Disp_Vec_fn(R, R)
            dr_pair =torch.sqrt(torch.sum(dR_pair ** 2, axis=-1))
            
            #2: Creating neigh_list and senders and receivers
            
            n_list=(dr_pair<pair_cutoffs_G).int()
            n_list.fill_diagonal_(0)
            (senders,receivers)=torch.where(n_list==1)
            
            #3: Node features
            Node_feats = torch.nn.functional.one_hot(species).float().to(dtype=torch.float64)
            vel=torch.sum(torch.square(V), dim=1, keepdim=True)
            emb_vel=torch.cat((Node_feats, vel), dim=1)
            
            #4: Edge Features
            Edge_feats=dR_pair[senders,receivers,:]
            # Edge_feats = dr_pair[senders,receivers].reshape((-1,1))
            
            #5: extra [Energy, type, 
            node_pe=Node_energy_fn(R)
            Energy=torch.sum(node_pe)/2
            
            _type = torch.nn.functional.one_hot(species).float().to(dtype=torch.float64)
            # _type = species.reshape(-1,1).float().to(dtype=torch.float64)
            # _type = torch.ones_like(_type)
            
            G = Data(x=Node_feats, edge_index=torch.stack([senders,receivers]), edge_attr=Edge_feats, type = _type, mass=torch.ones(len(species)), node_vel_emb=emb_vel, velocity=V, position=R, acceleration=A)
            return G



    def create_batched_States(self, Batch_size: int=20):
            Traj=torch.from_numpy(np.load('data/LJ_125/lamp_data/DatasetLJ_ab_3D.npy'))
            species=torch.from_numpy(np.load('data/LJ_125/lamp_data/species.npy'))
        
            N=species.shape[0]
            _box_size=(N/1.2)**(1/3)
            
            Disp_Vec_fn, pair_dist_fn, Node_energy_fn, Total_energy_fn, displacement_fn, shift_fn, pair_cutoffs, pair_sigma, pair_epsilon = lennard_jones(species=species, box_size=_box_size)
            # cutoffs_G:torch.Tensor =torch.Tensor([[1.5 ,1.25],[1.25 ,2.0]]).cuda()
            cutoffs_G:torch.Tensor = torch.Tensor([[2.5 ,2.5],[2.5 ,2.5]])
            pair_cutoffs_G = cutoffs_G[torch.meshgrid(species, species, indexing='xy')]
            
            G_list=[]
            for i in tqdm(range(len(Traj))):
                R = Traj[i][:,1:4]
                V = Traj[i][:,4:7]
                A = Traj[i][:,7:10]
                G_list += [self.create_G(R, V, A, species, pair_cutoffs_G, Disp_Vec_fn, Node_energy_fn)]
        
            N_sample = len(G_list)
            Train_loader = DataLoader(G_list[:int(1.0*N_sample)], batch_size=Batch_size, shuffle=True)
            # Test_loader = DataLoader(G_list[int(0.6*N_sample):int(0.8*N_sample)], batch_size=batch_size,shuffle=False,generator=torch.Generator(device='cuda'))
            # Test_loader = DataLoader(G_list[int(0.2*N_sample):], batch_size=Batch_size, shuffle=True,generator=torch.Generator(device='cuda'))
            
            return Train_loader
        