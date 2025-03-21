import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class C_MGN(nn.Module):
    """
    A PyTorch implementation of the C_MGN (Cascade Monotone Gradient Network) class.

    This model employs a cascading structure with shared parameters across layers and monotonic activation functions to compute output representations. The model ensures flexible and efficient processing of input data by utilizing shared weights, multiple bias terms, and orthogonality constraints when required.

    Classes:
        C_MGN: A neural network model implementing a cascade monotone gradient architecture.

    Attributes:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Number of units in the hidden layers.
        output_dim (int): Dimensionality of the output data.
        num_layers (int): Number of cascading layers in the network.
        W (torch.nn.Parameter): Shared weight matrix for all layers, mapping input to hidden dimensions.
        biases (torch.nn.ParameterList): List of bias vectors (b0 to bL) for each layer.
        bL (torch.nn.Parameter): Bias term added to the final output.
        V (torch.nn.Parameter): A weight matrix for additional transformations, optionally orthogonal.
        activation (function): The activation function used in each layer (e.g., sigmoid, tanh, softplus).

    Methods:
        forward(x):
            Executes the forward pass of the Cascade Monotone Gradient Network. The process includes:
            - Applying shared weights and bias terms across layers.
            - Passing intermediate computations through a monotonic activation function.
            - Combining results from cascading layers with additional transformations.
            Args:
                x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            Returns:
                torch.Tensor: Output tensor of shape [batch_size, output_dim].

    Initialization:
        ortho (bool): Whether to initialize matrix `V` as orthogonal (default is False).
        activation (str): The type of activation function to use ('sigmoid', 'tanh', 'softplus').

    Raises:
        ValueError: If an unsupported activation function is specified.

    Key Features:
        - Shared Weight Matrix: The same matrix `W` is used across all layers, reducing parameter complexity.
        - Cascade Structure: Each layer's output builds upon the previous one, ensuring efficient computation.
        - Orthogonality Constraint: Optionally initializes the `V` matrix to be orthogonal, enhancing stability.
        - Flexibility: Supports multiple activation functions, allowing customization for different tasks.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, ortho=False, activation='sigmoid'):
        super(C_MGN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # the matrix params is shared across layers
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim))
        
        # biases b0 to bL
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_layers)])
        self.bL = nn.Parameter(torch.randn(output_dim))
        
        self.V = nn.Parameter(torch.randn(input_dim, output_dim))
        if ortho and input_dim == output_dim: nn.init.orthogonal_(self.V)  # Initialize V to be orthogonal

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'softplus':
            self.activation = F.softplus
        else:
            raise ValueError("Activation function not supported.")
    
    def forward(self, x):
        # first layer
        z_prev = torch.matmul(x, self.W) + self.biases[0]
        
        for l in range(1, self.num_layers):
            z_l = torch.matmul(x, self.W) + self.activation(z_prev) + self.biases[l]
            z_prev = z_l
        inter_1 = torch.matmul(self.activation(z_prev), self.W.t()) # (batch_size, hidden_dim) * (hidden_dim, input_dim)
        # x@V (b, i) * (i, o) => (b, o)
        # x@V@V.T (b, o) * (o, i) = > (b, i)
        inter_2 = torch.matmul(torch.matmul(x, self.V), self.V.t()) 
        output = inter_1 + inter_2 + self.bL  # (batch_size, input_dim) +  (batch_size, input_dim) + (batch_size, input_dim)
        
        return output

