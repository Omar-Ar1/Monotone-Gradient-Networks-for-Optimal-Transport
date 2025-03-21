import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class M_MGN(nn.Module):
    """
    A PyTorch implementation of the M_MGN (Modular Montone Gradient Network) class.

    Classes:
        M_MGN: A neural network model with custom structured layers and terms.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        hidden_dim (int): The number of units in the hidden layers.
        num_layers (int): The number of custom layers in the network.
        W_k (nn.ModuleList): A list of linear layers (weights W_k) for each layer.
        activations (nn.ModuleList): A list of activation functions (e.g., Tanh) for each layer.
        V (nn.Linear): A learnable linear layer for the term V^T V, initialized to be orthogonal.
        a (nn.Parameter): A learned bias term added to the final output.

    Methods:
        forward(x):
            Executes the forward pass of the network. Computes the output by combining the learned 
            bias term, the PSD term V^T V, and the contributions of all layers, including the 
            activation and logcosh transformations.
            Args:
                x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            Returns:
                torch.Tensor: Output tensor of shape [batch_size, input_dim].

        logcosh(x):
            Computes the element-wise log(cosh(x)) transformation for an input tensor.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Transformed tensor with log(cosh(x)) applied element-wise.
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(M_MGN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define modules (W_k, b_k, and activation functions)
        self.W_k = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, bias=True) for _ in range(num_layers)
        ])

        # Each module has its own activation (e.g., tanh, softplus)
        self.activations = nn.ModuleList([nn.Tanh() for _ in range(num_layers)])

        # V^T V term (PSD by construction)
        self.V = nn.Linear(input_dim, input_dim, bias=False)  # Shape: [input_dim, input_dim]
        nn.init.orthogonal_(self.V.weight)  # Initialize V to be orthogonal

        # Bias term (a)
        self.a = nn.Parameter(torch.randn(input_dim))  # Learned bias


    def forward(self, x):
        batch_size = x.shape[0]

        # Initialize output with bias term (broadcasted to batch)
        out = self.a.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, input_dim]

        # Add V^T V x term (ensures PSD Jacobian)
        V_sq = self.V.weight.t() @ self.V.weight  # Shape: [input_dim, input_dim]
        out = out + x @ V_sq  # Shape: [batch_size, input_dim]

        # Loop over modules and compute terms
        for k in range(self.num_layers):
            # Compute z_k = W_k x + b_k
            z_k = self.W_k[k](x)  # Shape: [batch_size, hidden_dim]

            # Compute s_k(z_k) = sum_i log(cosh(z_k_i)) (scalar per sample)
            s_k = torch.sum(torch.log(torch.cosh(z_k)), dim=1)  # Shape: [batch_size]

            # Compute activation σ_k(z_k)
            sigma_k = self.activations[k](z_k)  # Shape: [batch_size, hidden_dim]

            # Compute s_k(z_k) * W_k^T σ_k(z_k)
            W_k_T = self.W_k[k].weight.t()  # Shape: [input_dim, hidden_dim]
            term = (W_k_T @ sigma_k.t()).t()  # Shape: [batch_size, input_dim]
            term = s_k.unsqueeze(-1) * term  # Broadcast s_k and multiply

            out += term

        return out  # Shape: [batch_size, input_dim]

    def logcosh(self, x):
        return torch.log(torch.cosh(x))

