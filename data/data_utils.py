import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.mixture import GaussianMixture

transform = transforms.Compose([
    transforms.ToTensor(),
])

def plot_rgb_dist(input_image):
    # Convert the image to a NumPy array
    image_np = input_image.numpy()

    # Split the channels (R, G, B)
    red_channel = image_np[:, :, 0].flatten()  # Red
    green_channel = image_np[:, :, 1].flatten()  # Green
    blue_channel = image_np[:, :, 2].flatten()  # Blue

    # Plot histograms for each channel
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(red_channel, bins=256, color="red", alpha=0.7)
    plt.title("Red Channel Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    plt.hist(green_channel, bins=256, color="green", alpha=0.7)
    plt.title("Green Channel Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 3)
    plt.hist(blue_channel, bins=256, color="blue", alpha=0.7)
    plt.title("Blue Channel Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def plot_model_test(model, image_path, target_image):
    test_image = transform(Image.open(image_path))
    with torch.no_grad():
        result = model(test_image.permute(1, 2, 0).view(-1, 3))
    result = result.view(test_image.permute(1, 2, 0).shape)

    # Transport image
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    # Original image
    ax[0].imshow(test_image.permute(1, 2, 0).detach().cpu().numpy())
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Transported image
    ax[1].imshow(result.detach().cpu().numpy())
    ax[1].set_title('Transported Colors')
    ax[1].axis('off')

    # Target colors
    ax[2].imshow(target_image.permute(1, 2, 0).detach().cpu().numpy())
    ax[2].set_title('Target colors')
    ax[2].axis('off')
    plt.show()

def plot_compare_loss(train_loss, train_cost, labels):
    # X-axis values (assumes same length for both metrics)
    x_axis = list(range(len(train_loss[0])))

    # Create subplots with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Train Loss", "Train Cost"])

    # Add traces for train loss
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(x=x_axis, y=train_loss[i], mode='lines', name=f"Loss - {label}"), row=1, col=1)

    # Add traces for train cost
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(x=x_axis, y=train_cost[i], mode='lines', name=f"Cost - {label}"), row=1, col=2)

    # Update layout
    fig.update_layout(
        title="Training Loss and Cost Comparisons",
        xaxis_title="Epochs",
        yaxis_title="Values",
        legend_title="Legend",
        height=400,
        width=1200,
    )

    # Show the plot
    fig.show()


def find_gm_components(target_image):
    # Convert the target image to a NumPy array (example tensor to simulate your setup)
    all_channels = target_image.permute(1, 2, 0).view(-1, 3).numpy()  # Assuming target_image is a torch tensor

    # Range of components to try
    components_range = range(1, 11)  # Testing GMMs with 1 to 10 components
    log_likelihoods = []  # To store log-likelihoods for each model

    for n_components in components_range:
        # Fit the GMM model
        gmm_model = GaussianMixture(n_components=n_components, random_state=42)
        gmm_model.fit(all_channels)
        
        # Store the log-likelihood
        log_likelihoods.append(gmm_model.lower_bound_)

    # Plot the log-likelihoods
    plt.figure(figsize=(8, 6))
    plt.plot(components_range, log_likelihoods, marker='o', linestyle='-', color='b')
    plt.title('Log-Likelihood vs Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)
    plt.show()