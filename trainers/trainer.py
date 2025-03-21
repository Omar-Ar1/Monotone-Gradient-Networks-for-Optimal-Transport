import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Callable
import matplotlib.pyplot as plt

class Trainer:
    """General-purpose trainer for M-MGN supporting multiple tasks."""
    def __init__(
        self,
        task: str = 'gradient',  # 'gradient' or 'optimal_transport'
        dataset: str = '2D_distribution',
        input_data : Optional[torch.tensor] = None,
        target_data: Optional[torch.tensor] = None,
        target_distribution: Optional[dict] = None,
        n_epochs: int = 50,
        lr: float = 0.01,
        criterion: str = 'L1loss',
        optimizer: str = 'Adam',
        weight_decay: float = 0,
        betas: Tuple[float, float] = (0.9, 0.999),
        model: nn.Module = None,
        model_name: Optional[str] = None,
        true_fx: Optional[Callable] = None,
        batch_size: int = 32,
        device: str = 'cpu',
        verbose: bool = True
    ):
        self.task = task
        self.dataset = dataset
        self.target_distribution = target_distribution
        self.device = device
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.lr = lr
        self.batch_size = batch_size
        self.input_data = input_data
        self.target_data = target_data
        self.metrics = {
            'train_loss': [],  'train_cost': []
        }

        # Initialize model
        self.model = model.to(device)
        self.model_name = model_name

        # Initialize the function to approximate (if applicable)
        self.true_fx = true_fx

        # Initialize optimizer and loss
        self.optimizer = self._get_optimizer(optimizer=optimizer, weight_decay=weight_decay, betas=betas)
        self.criterion = self._get_criterion(criterion=criterion)

    def _get_optimizer(self, optimizer: str, weight_decay: float, betas: Tuple[float, float]):
        """Initialize the optimizer."""
        if optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr, betas=betas, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported.")

    def _get_criterion(self, criterion: str):
        """Initialize the criterion."""
        if criterion.lower() == 'l1loss':
            return nn.L1Loss()
        elif criterion.lower() in ['kld', 'nll']:
            return lambda x, y: self._custom_loss(x, y)
        elif criterion.lower() == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f"Criterion '{criterion}' is not supported.")
    def _get_grad(self, x):
        
        if self.model_name and self.model_name.lower() == 'icnn':
            # Compute gradients
            outputs = self.model(x.squeeze(0))
            grad = torch.autograd.grad(
                outputs=outputs,
                inputs=x,
                grad_outputs=torch.ones_like(outputs),
                create_graph=True
            )[0]  # Extract gradient tensor

            return grad
        return self.model(x)

    def _kld_loss(self, g_x, mu_p, cov_p):
        """
        Compute the KLD between the empirical distribution of g_x and the target Gaussian.
        g_x is assumed to be a batch of samples with shape [batch_size, d].
        """
        # Ensure eps is the same dtype as g_x
        eps = torch.tensor(1e-6, dtype=g_x.dtype, device=self.device)
        
        # Estimate the empirical mean and covariance from g_x
        mu_q = g_x.mean(dim=0)
        diff = g_x - mu_q
        cov_q = (diff.T @ diff) / (g_x.size(0) - 1) + eps * torch.eye(g_x.shape[1], device=self.device, dtype=g_x.dtype)
        
        # Cast target parameters to match g_x dtype
        mu_p = mu_p.to(g_x.dtype)
        cov_p = cov_p.to(g_x.dtype) + eps * torch.eye(mu_p.shape[0], device=self.device, dtype=g_x.dtype)
        d = mu_q.shape[0]
        
        # Compute log determinants in a stable way
        _, logdet_q = torch.linalg.slogdet(cov_q)
        _, logdet_p = torch.linalg.slogdet(cov_p)
        
        epsilon = 1e-6  # Small positive value to prevent singularity
        inv_cov_p = torch.linalg.inv(cov_p + epsilon * torch.eye(cov_p.shape[0], device=cov_p.device))

        trace_term = torch.trace(inv_cov_p @ cov_q)
        mean_diff = (mu_p - mu_q).unsqueeze(0)  # shape [1, d]
        mean_term = mean_diff @ inv_cov_p @ mean_diff.T
        
        kld = 0.5 * (logdet_p - logdet_q - d + trace_term + mean_term.squeeze())
        return kld
    
    def _custom_loss(self, x_batch, g_x):
        """Custom loss for optimal transport (NLL with Jacobian)."""

        # Gaussian log-likelihood
        if not self.target_distribution:
            log_p = -0.5 * (g_x ** 2).sum(dim=1) - torch.log(torch.tensor(2 * torch.pi))
            # Compute Jacobian for each sample in the batch
            batch_size = x_batch.shape[0]
            log_det = torch.zeros(batch_size, device=x_batch.device)

            for i in range(batch_size):
                J = torch.autograd.functional.jacobian(lambda x: self._get_grad(x.unsqueeze(0)), x_batch[i], create_graph=True)
                det = torch.det(J)
                log_det[i] = torch.log(det.abs() + 1e-6)  # Avoid log(0)
            return - (log_p + log_det).mean()
        elif 'weights' not in self.target_distribution:
            return self._kld_loss(g_x, self.target_distribution['mean'], self.target_distribution['cov'])

        else:
            m = self.target_distribution['mean']
            cov = self.target_distribution['cov']
            d = cov.shape[-1]
            weights = self.target_distribution['weights']
            loss = float('inf')
            for i, w in enumerate(weights):
                kld = self._kld_loss(g_x, m[i], cov[i])
                if kld < loss:
                    loss = kld
            return loss



    def compute_transport_cost(self, x, g_x):
        """Compute Brenier's transport cost: E[||x - g(x)||^2]."""
        return torch.mean(torch.sum((x - g_x) ** 2, dim=1))

    def compute_jacobian(self, x):
        """Compute the Jacobian of g(x) w.r.t. x."""
        batch_size = x.shape[0]
        jacobian = torch.zeros(batch_size, x.shape[1], x.shape[1], device=x.device)
        for i in range(batch_size):
            jacobian[i] = torch.autograd.functional.jacobian(lambda x: self.model(x), x[i].unsqueeze(0))
        return jacobian

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        best_val_loss = float('inf')
        if self.input_data != None:
            image = self.input_data.permute(1, 2, 0).view(-1, 3)
        for epoch in (pbar := tqdm(range(self.n_epochs + 1), disable=not self.verbose)):
            self.model.train()
            train_loss = train_cost = 0

            for batch in train_loader: 
                x = batch[0]  # Input data
                if self.task == 'gradient':
                    true_grad = batch[1]  # Ground truth gradient
                else:
                    true_grad = None
                # Forward pass
                x.requires_grad = True
                g_x = self._get_grad(x)
                
                # Compute loss
                if self.task == 'gradient':
                    loss = self.criterion(g_x, true_grad)
                else:  # Optimal transport
                    loss = self.criterion(x, g_x)

                # Compute transport cost
                cost = self.compute_transport_cost(x, g_x)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                train_loss += loss.item() * x.shape[0]
                train_cost += cost.item() * x.shape[0]

            if self.input_data != None and epoch % 10 == 0:
                with torch.no_grad():
                    result = self.model(image)
                result = result.view(self.input_data.permute(1, 2, 0).shape)

                fig, ax = plt.subplots(1, 3, figsize=(12, 6))
                # Original image
                ax[0].imshow(self.input_data.permute(1, 2, 0).detach().cpu().numpy())
                ax[0].set_title('Original Image')
                ax[0].axis('off')

                # Transported image
                ax[1].imshow(result.detach().cpu().numpy())
                ax[1].set_title('Transported Colors')
                ax[1].axis('off')

                # Target colors
                ax[2].imshow(self.target_data.permute(1, 2, 0).detach().cpu().numpy())
                ax[2].set_title('Target colors')
                ax[2].axis('off')
                plt.show()


            # Update metrics
            self.metrics['train_loss'].append(train_loss / len(train_loader.dataset))
            self.metrics['train_cost'].append(train_cost / len(train_loader.dataset))


            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1} | "
                                f"Train Loss: {train_loss / len(train_loader.dataset):.4f} | "
                                f"Train Cost: {train_cost / len(train_loader.dataset):.4f} | ")

    def plot_train_metrics(self, plot_cost=False):
        """Plot training and validation metrics using Plotly."""
        if not self.metrics['train_loss']:
            raise ValueError("Training metrics are empty. Ensure the model has been trained.")
        cols = 2 if plot_cost else 1
        # Create subplots
        fig = make_subplots(rows=1, cols=cols, subplot_titles=(
            "Train Loss", "Train Cost"
        ))

        # Add training and validation loss plot
        fig.add_trace(
            go.Scatter(x=list(range(len(self.metrics['train_loss']))), y=self.metrics['train_loss'], name="Train Loss", mode="lines"),
            row=1, col=1
        )

        # Add training and validation cost plot
        if plot_cost:
            fig.add_trace(
                go.Scatter(x=list(range(len(self.metrics['train_cost']))), y=self.metrics['train_cost'], name="Train Cost", mode="lines"),
                row=1, col=2
            )


        # Update layout
        fig.update_layout(
            title=f"Training Metrics for {self.dataset}",
            showlegend=True,
            width=1200 if plot_cost else 600,
            height=500
        )

        # Show the plot
        fig.show()
        