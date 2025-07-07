import os
import matplotlib.pyplot as plt
import utils.constants as cons
from utils.load_data import load_autoencoder, load_dataloaders
from utils.autoencoders import CustomLoss, train_autoencoder, evaluate_autoencoder, extract_embeddings

import torch
import torch.nn as nn
import torch.optim as optim


# Establish a seed for reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)           # CPU
    torch.cuda.manual_seed(seed)      # GPU (1 GPU)
    torch.cuda.manual_seed_all(seed)  # GPU (multi-GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

# ---------------------------
# Things to do
# ---------------------------
TRAIN = True
EMBEDS = True

# ---------------------------
# Data parameters
# ---------------------------
type_signal = 'bipolar'
subsampling = 2
window_size = 500
window_overlap = 0
seed = 42
verbose = True
batch_size = 128

# Architectures to try
layers = [[64, 64]]
latent_spaces = [16]
kernels = [3]
strides = [1]
paddings = [1]

# Loop through each combination of architecture and hyperparameters
for layer_channels in layers:
    for s in strides:
        for kernel_size in kernels:
            for latent_dim in latent_spaces:
                for padding in paddings:

                    # Static parameters
                    pooling = True
                    norm = True
                    drop = 0.25

                    # Build unique model identifier string
                    aux_str = (
                        f'ch{layer_channels}_k{kernel_size}_s{s}_p{padding}_pool{pooling}_norm{norm}_drop{drop}'
                        f'_ld{latent_dim}_mse'
                    )

                    # Create output directories for results
                    saving_path = cons.RESULTS_DIR / type_signal
                    training_graph_saving_path = saving_path / aux_str / 'training'
                    metrics_saving_path = saving_path / aux_str / 'metrics'
                    model_saving_path = saving_path / aux_str / 'model'
                    embeddings_saving_path = saving_path / aux_str / 'embeddings'
                    os.makedirs(training_graph_saving_path, exist_ok=True)
                    os.makedirs(metrics_saving_path, exist_ok=True)
                    os.makedirs(model_saving_path, exist_ok=True)
                    os.makedirs(embeddings_saving_path, exist_ok=True)

                    # Load the autoencoder model with the current configuration
                    model = load_autoencoder(
                        type_signal, window_size, subsampling, layer_channels, kernel_size, s,
                        padding, pooling, norm, drop, latent_dim, train=True
                    )

                    # Train the model
                    if TRAIN:
                        # Load data loaders for training
                        train_loader, val_loader, test_loader = load_dataloaders(
                            type_signal, window_size, window_overlap, subsampling, seed, batch_size, True
                        )

                        # Define loss function and optimizer
                        learning_rate = 1e-3
                        # if type_signal == 'unipolar':
                        criterion = nn.MSELoss()
                        # else:
                        #     criterion = CustomLoss(lambda_reg=0, tau=0.05, eta=1e-8)
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        scheduler = None
                        num_epochs = 100
                        patience = 5

                        # Train model with early stopping
                        model, best_val_loss, train_losses, val_losses = train_autoencoder(
                            model, train_loader, val_loader, criterion, optimizer, scheduler,
                            num_epochs=num_epochs, patience=patience
                        )

                        # Save the best model checkpoint
                        torch.save(model.state_dict(), model_saving_path / "best_autoencoder.pth")
                        print("Best model saved as 'best_autoencoder.pth'.")

                        # Plot training and validation loss
                        plt.figure(figsize=(10, 6))
                        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss',
                                 color='blue', marker='o')
                        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss',
                                 color='red', marker='o')
                        plt.xlabel('Epochs', fontsize=16)
                        plt.ylabel('MSE', fontsize=16)
                        plt.title('Training vs Validation Loss', fontsize=18)
                        plt.legend(fontsize=14)
                        plt.xticks(fontsize=14)
                        plt.yticks(fontsize=14)
                        plt.tight_layout()
                        plt.savefig(training_graph_saving_path / 'loss_plot.png')
                        plt.show()

                        # Evaluate model on test set and save test loss
                        _, test_loss = evaluate_autoencoder(model, test_loader)
                        with open(metrics_saving_path / 'metrics.txt', 'a') as f:
                            f.write(f"Test Loss: {test_loss}\n")

                    # Extract embeddings
                    if EMBEDS:
                        # Load data loaders (shuffling disabled for reproducibility)
                        train_loader, val_loader, test_loader = load_dataloaders(
                            type_signal, window_size, window_overlap, subsampling, seed, batch_size, False
                        )

                        # Load best model weights
                        state_dict = torch.load(model_saving_path / "best_autoencoder.pth", weights_only=True)
                        model.load_state_dict(state_dict)

                        # Extract and save embeddings from each split
                        extract_embeddings(model, train_loader, embeddings_saving_path / 'train_embeds.pkl')
                        extract_embeddings(model, val_loader, embeddings_saving_path / 'val_embeds.pkl')
                        extract_embeddings(model, test_loader, embeddings_saving_path / 'test_embeds.pkl')
