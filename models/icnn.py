import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class I_CNN(nn.Module):
    """
    A PyTorch implementation of the I_CNN (Input Convex Neural Network) class.

    This model ensures convexity with respect to its input by constraining certain weights 
    to be non-negative and applying ReLU activations. It features a modular architecture where 
    each layer uses shared components for flexibility and scalability.

    Classes:
        I_CNN: A neural network model implementing the input convex property.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        hidden_dim (int): The number of units in the hidden layers.
        num_layers (int): The number of layers in the network.
        relu (torch.nn.ReLU): ReLU activation function for introducing non-linearity.
        modules_x (torch.nn.ModuleList): A list of `Linear` layers for transformations on the input data.
        modules_z (torch.nn.ModuleList): A list of `Linear` layers for transformations on intermediate outputs.

    Methods:
        forward(x):
            Executes the forward pass of the Input Convex Neural Network. This includes:
            - Constraining weights of intermediate layers to be non-negative.
            - Computing intermediate representations using ReLU activations.
            Args:
                x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            Returns:
                torch.Tensor: Final output tensor of shape [batch_size, hidden_dim].

        _initialize_weights():
            Applies Xavier uniform initialization to all layer weights and initializes biases to zero.
            Ensures proper weight initialization for improved training performance.

    Key Features:
        - Convexity: Weight constraints (`clamp_(min=0)`) enforce convexity with respect to the input.
        - Modularity: Separate modules for input transformations (`modules_x`) and intermediate computations (`modules_z`).
        - Weight Initialization: Xavier uniform initialization ensures stable training dynamics.
        - Scalability: The architecture can handle variable numbers of layers and dimensions.

    Notes:
        1. Convexity ensures that the network is well-suited for problems where convex representations 
        are necessary (e.g., optimization, structured predictions).
        2. Non-negative constraints on `modules_z` weights are applied during each forward pass 
        to enforce the convexity property.
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(I_CNN, self).__init__()
        # Define ReLU activation
        self.relu = nn.ReLU()

        # Use nn.ModuleList for layers to ensure they are registered as model parameters
        self.modules_x = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=True) for _ in range(num_layers)])
        self.modules_z = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Initialize zl as a zero tensor with the same device as x
        zl = torch.zeros(x.shape[0], self.modules_x[0].out_features, device=x.device)
        for i in range(len(self.modules_x)):
            self.modules_z[i].weight.data.clamp_(min=0)
            zl = self.relu(self.modules_z[i](zl) + self.modules_x[i](x))  # Apply ReLU activation
        return zl

    def _initialize_weights(self):
        # Loop through all layers and apply Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  # Apply Xavier uniform initialization
                if module.bias is not None:
                    init.zeros_(module.bias)  # Initialize biases to 0