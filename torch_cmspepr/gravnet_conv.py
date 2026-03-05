from typing import Optional, Tuple
from torch_geometric.typing import OptTensor

import torch
from torch import Tensor
from torch.nn import Linear

# Import the custom GravNetOp
# try:
from fastgraphcompute.gnn_ops import GravNetOp
print("Using GravNetOp from gnn_ops")



class GravNetConv(torch.nn.Module):
    r"""The GravNet operator from the `"Learning Representations of Irregular
    Particle-detector Geometry with Distance-weighted Graph
    Networks" <https://arxiv.org/abs/1902.07987>`_ paper, where the graph is
    dynamically constructed using nearest neighbors.
    
    This implementation can use either:
    1. GravNetOp with custom CUDA operations (more memory-efficient)
    2. PyTorch Geometric's MessagePassing (fallback)

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        space_dimensions (int): The dimensionality of the space used to
           construct the neighbors; referred to as :math:`S` in the paper.
        propagate_dimensions (int): The number of features to be propagated
           between the vertices; referred to as :math:`F_{\textrm{LR}}` in the
           paper.
        k (int): The number of nearest neighbors.
        output_activation: Activation function for output layer (default: ReLU)
        **kwargs (optional): Additional arguments.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        space_dimensions: int,
        propagate_dimensions: int,
        k: int,
        num_workers: int = 1,
        output_activation = torch.nn.Identity(),
        **kwargs
    ):
        super(GravNetConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.space_dimensions = space_dimensions
        self.propagate_dimensions = propagate_dimensions
        self.k = k
        self.num_workers = num_workers
        self.flow = kwargs.get('flow', 'target_to_source')  # Preserve flow parameter
        # Determine which implementation to use
        
        
            
        self.gravnet_op = GravNetOp(
            in_channels=in_channels,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
            output_activation=output_activation,
            optimization_arguments=kwargs.get('optimization_arguments', {})
        )


    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        """
        Forward pass of GravNetConv.
        
        Args:
            x: Input feature tensor of shape (N, in_channels)
            batch: Batch assignment tensor of shape (N,)
        
        Returns:
            Output tensor of shape (N, out_channels)
        """
            # Convert batch tensor to row_splits format required by GravNetOp
        row_splits = self._batch_to_row_splits(batch, x.size(0), x.device)
        
        # Use GravNetOp
        out, _, _, _ = self.gravnet_op(x, row_splits)
        return out


    def _batch_to_row_splits(self, batch: OptTensor, num_nodes: int, device: torch.device) -> Tensor:
        """
        Convert batch tensor to row_splits format.
        
        Args:
            batch: Batch assignment tensor of shape (N,) or None
            num_nodes: Total number of nodes
            
        Returns:
            row_splits: Tensor indicating the start index of each batch
        """
        if batch is None:
            # Single batch case
            return torch.tensor([0, num_nodes], dtype=torch.long, device=device)
        
        # Count nodes per batch
        batch_size = int(batch.max().item()) + 1
        row_splits = torch.zeros(batch_size + 1, dtype=torch.long, device=batch.device)
        
        # Compute cumulative sum
        # ones = torch.ones_like(batch)
        row_splits[1:] = torch.cumsum(
            torch.bincount(batch, minlength=batch_size), dim=0
        )
        
        return row_splits


    def __repr__(self):
        impl = "GravNetOp"
        return '{}({}, {}, k={}, implementation={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, 
            self.k, impl
        )