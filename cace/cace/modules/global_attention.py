import torch
import torch.nn as nn
import torch.nn.functional as F
# from .attention_performer import Attention
# from .residual_mlp import ResidualMLP
from typing import Optional
from performer_pytorch import SelfAttention

class NonlocalInteraction(nn.Module):
    """
    Block for updating atomic features through nonlocal interactions with all
    atoms.

    Arguments:
        num_features (int):
            Dimensions of feature space.
        num_basis_functions (int):
            Number of radial basis functions.
        num_residual_pre_i (int):
            Number of residual blocks applied to atomic features in i branch
            (central atoms) before computing the interaction.
        num_residual_pre_j (int):
            Number of residual blocks applied to atomic features in j branch
            (neighbouring atoms) before computing the interaction.
        num_residual_post (int):
            Number of residual blocks applied to interaction features.
        activation (str):
            Kind of activation function. Possible values:
            'swish': Swish activation function.
            'ssp': Shifted softplus activation function.
    """

    def __init__(
        self,
    ) -> None:
        """ Initializes the NonlocalInteraction class. """
        super(NonlocalInteraction, self).__init__()

        # self.resblock_q = ResidualMLP(
        #     num_features, num_residual_q, activation=activation, zero_init=True
        # # )
        # self.resblock_k = ResidualMLP(
        #     num_features, num_residual_k, activation=activation, zero_init=True
        # )
        # self.resblock_v = ResidualMLP(
        #     num_features, num_residual_v, activation=activation, zero_init=True
        # )

        self.attn = SelfAttention(
            dim = 4,
            heads = 2,
            causal = False,
        )

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def initialize_resq(self, input_size):
        self.resblock_q = ResidualMLP(
            input_size, input_size, zero_init=True
        )

    def initialize_resk(self, input_size):
        self.resblock_k = ResidualMLP(
            input_size, input_size, zero_init=True
        )

    def initialize_resv(self, input_size):
        self.resblock_v = ResidualMLP(
            input_size, input_size, zero_init=True
        )


    def forward(
        self,
        x: torch.Tensor,
        graph_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.

        """
        num_atoms = x.size(0)

        # if not isinstance(self.resblock_q, ResidualMLP):
        #     input_size = x.size(1)
        #     self.initialize_resq(input_size)

        # if not isinstance(self.resblock_k, ResidualMLP):
        #     input_size = x.size(1)
        #     self.initialize_resk(input_size)

        # if not isinstance(self.resblock_v, ResidualMLP):
        #     input_size = x.size(1)
        #     self.initialize_resv(input_size)

        # Q = self.resblock_q(x)  # queries
        # K = self.resblock_k(x)  # keys
        # V = self.resblock_v(x)  # values

        # q_split = torch.split(Q, num_atoms)
        # k_split = torch.split(K, num_atoms)
        # v_split = torch.split(V, num_atoms)

        # att = []

        # for i, q in enumerate(q_split):
        #     k = k_split[i]
        #     v = v_split[i]

        #     qk = torch.matmul(q, k.transpose(0, 1)) / self.feat_dim**0.5
        #     this_att = torch.matmul(torch.softmax(qk, dim=-1), v)
        #     att.append(this_att)

        # att = torch.cat(att)

        grouped_node_properties = []

        for graph_id in torch.unique(graph_ids):
            mask = (graph_ids == graph_id)
            graph_node_properties = x[mask].unsqueeze(0)
            if graph_node_properties.shape[2] == 1:
                grouped_node_properties.append(graph_node_properties)
            else:    
                print("graph_node_properties:", graph_node_properties.shape)
                modified_graph_node = self.attn(graph_node_properties)            
                print("modified_graph_node:", modified_graph_node.squeeze(0).shape)
                grouped_node_properties.append(modified_graph_node.squeeze(0))   

        grouped_node_properties = torch.cat(grouped_node_properties, dim=0)
        print("attention implemented", grouped_node_properties.shape)

        return grouped_node_properties