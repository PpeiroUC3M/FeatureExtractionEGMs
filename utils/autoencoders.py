import pickle
import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=1, input_size=(15, 500), channels_per_layer=(32, 64, 128, 256), kernel_size=3,
                 stride=2, padding=0, norm=False, pooling=(True,), drop=0, latent_dim=128):
        """
        Initializes a convolutional autoencoder model.

        Parameters:
        - in_channels (int): Number of input channels (e.g., 1 for grayscale images).
        - input_size (tuple): Height and width of the input image or 2D input (e.g., (15, 500)).
        - channels_per_layer (tuple): Number of channels in each convolutional layer of the encoder.
        - kernel_size (int or tuple): Size of the convolutional kernel.
        - stride (int): Stride used in convolutional and transposed convolutional layers.
        - padding (int): Padding applied to convolutional layers.
        - norm (bool): If True, applies Batch Normalization after convolution.
        - pooling (tuple or bool): Whether to apply max pooling in each encoder layer.
        - drop (float): Dropout probability (0 to disable).
        - latent_dim (int): Size of the latent vector in the bottleneck.
        """
        super(ConvAutoencoder, self).__init__()

        # Store input channel size
        self.in_channels = in_channels

        # Number of layers in the encoder/decoder
        self.num_layers = len(channels_per_layer)

        # Ensure pooling is a list to match the number of encoder layers
        if isinstance(pooling, (tuple, list)):
            pooling = pooling
        else:
            pooling = [pooling] * self.num_layers
        self.pooling = pooling

        # Containers for encoder and pooling layers
        self.encoder = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        # Track input channels for each convolutional layer
        in_ch = in_channels

        # Build encoder
        for n, out_ch in enumerate(channels_per_layer):
            layer = nn.ModuleList()

            # Add a Conv2D layer
            layer.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding))

            # Optional BatchNorm layer
            if norm:
                layer.append(nn.BatchNorm2d(out_ch))

            # ReLU activation
            layer.append(nn.ReLU(inplace=True))

            # Optional Dropout layer
            if drop > 0:
                layer.append(nn.Dropout(p=drop))

            # Optional MaxPooling layer with index return (for unpooling)
            if pooling[n]:
                self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            else:
                self.pool_layers.append(None)

            # Combine the encoder layer components into a Sequential module
            self.encoder.append(nn.Sequential(*layer))

            # Update input channel size for the next layer
            in_ch = out_ch

        # Compute the final feature size after the encoder flattened and not flattened (spatial dims)
        self.flatten_size, self.spatial_dims, self.intermediate_shapes = self._get_shapes(input_size)

        # Fully connected layers for the bottleneck (encoder/decoder interface)
        self.fc1 = nn.Linear(self.flatten_size, latent_dim)   # Encoder FC
        self.fc2 = nn.Linear(latent_dim, self.flatten_size)   # Decoder FC

        # Calculate output padding to ensure decoder output matches encoder input size
        self.output_padding = self._get_output_padding(self.intermediate_shapes, kernel_size, stride, padding)

        # Containers for decoder and unpooling layers
        self.decoder = nn.ModuleList()
        self.unpool_layers = nn.ModuleList()

        # Build decoder by reversing the encoder configuration
        for i in reversed(range(len(channels_per_layer))):
            layer = nn.ModuleList()

            # Determine the output channel size for this layer
            out_ch = channels_per_layer[i - 1] if i > 0 else in_channels

            # Optional MaxUnpooling layer to reverse pooling
            if pooling[i]:
                self.unpool_layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
            else:
                self.unpool_layers.append(None)

            # Transposed convolution layer
            layer.append(nn.ConvTranspose2d(channels_per_layer[i], out_ch, kernel_size, stride, padding,
                                            output_padding=self.output_padding[i]))

            # If not the final output layer, apply normalization, activation, dropout
            if i > 0:
                if norm:
                    layer.append(nn.BatchNorm2d(out_ch))
                layer.append(nn.ReLU(inplace=True))
                if drop > 0:
                    layer.append(nn.Dropout(p=drop))
            else:
                # Final output activation: Tanh for [-1, 1] range
                layer.append(nn.Tanh())

            # Add this decoder stage
            self.decoder.append(nn.Sequential(*layer))

    def _get_shapes(self, input_size):
        """
        Runs a dummy input through the encoder to determine the final flattened size and record all intermediate
        spatial shapes for later reconstruction.
        """
        x = torch.zeros(1, self.in_channels, *input_size)
        intermediate_shapes = [x.shape[2:]]

        # Run the encoder and add intermediate spatial dimensions
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            intermediate_shapes.append(x.shape[2:])
            if self.pooling[i]:
                x, _ = self.pool_layers[i](x)
                intermediate_shapes.append(x.shape[2:])

        # Save the spatial dimensions and flattened size after all encoding
        spatial_dims = x.shape[1:]
        flatten_size = x.numel()

        return flatten_size, spatial_dims, intermediate_shapes

    def _get_output_padding(self, intermediate_shapes, kernel, stride, padding):
        """
        Calculates the necessary output padding for each transposed convolution layer so the decoder can correctly
        reconstruct the original shape.
        """
        output_paddings = []

        if isinstance(kernel, tuple):
            h_kernel, w_kernel = kernel
        else:
            h_kernel = w_kernel = kernel

        indexes = range(0, len(intermediate_shapes) - 1)
        for i in indexes:
            # Get the shapes of the corresponding layer
            h_decoded, w_decoded = intermediate_shapes[i]
            h_encoded, w_encoded = intermediate_shapes[i + 1]

            # Calculate height and width padding
            h_output_padding = h_decoded - ((h_encoded - 1) * stride + h_kernel - 2 * padding)
            w_output_padding = w_decoded - ((w_encoded - 1) * stride + w_kernel - 2 * padding)

            # Append if padding is within kernel bounds
            if h_output_padding <= h_kernel and w_output_padding <= w_kernel:
                output_paddings.append((h_output_padding, w_output_padding))

        return output_paddings

    def forward(self, x_input):
        """
        Forward pass of the autoencoder.
        - Encodes input to latent space.
        - Decodes it back to reconstructed output.
        """
        pool_indices = []
        x = x_input

        # Encoder pass
        for layer, pool in zip(self.encoder, self.pool_layers):
            x = layer(x)
            if pool:
                # Apply pooling and store indices for unpooling
                x, indices = pool(x)
                pool_indices.append(indices)
            else:
                pool_indices.append([])

        # Flatten and encode into latent space
        x = x.view(x.size(0), -1)
        encoded = self.fc1(x)

        # Decode from latent space
        x = self.fc2(encoded)
        x = x.view(x.size(0), *self.spatial_dims)

        # Prepare output sizes for unpooling
        pool_sizes = []
        n = 0
        for j in self.pooling:
            if j:
                pool_sizes.append(self.intermediate_shapes[n+1])
                n += 2
            else:
                pool_sizes.append([])
                n += 1

        # Decoder pass
        for i, (layer, unpool) in enumerate(zip(self.decoder, self.unpool_layers)):
            if unpool:
                # Apply unpooling using saved indices and target sizes
                x = unpool(x, pool_indices[::-1][i], output_size=pool_sizes[::-1][i])
            # Apply transposed convolution and activation
            x = layer(x)

        decoded = x
        return encoded, decoded, pool_indices


class CustomLoss(nn.Module):
    def __init__(self, lambda_reg=0.1, tau=0.1, eta=1e-8):
        """
        Custom loss function combining MSELoss with L1 regularization based on peak detection.

        Args:
            lambda_reg (float): Weight for the regularization term.
            tau (float): Threshold for peak detection.
            eta (float): Small constant to avoid division by zero.
        """
        super(CustomLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.tau = tau
        self.eta = eta

    def forward(self, y_pred, y_true):
        """
        Compute the custom loss.

        Args:
            y_pred (torch.Tensor): Predicted values (N, ).
            y_true (torch.Tensor): Ground truth values (N, ).

        Returns:
            torch.Tensor: Computed loss value.
        """
        # Ensure y_pred and y_true have the same shape
        assert y_pred.shape == y_true.shape, "y_pred and y_true must have the same shape"

        # Compute the MSE term
        mse_loss = F.mse_loss(y_pred, y_true)

        # Compute the regularization term
        abs_diff = torch.abs(y_true - y_pred)  # |y_n - \hat{y}_n|
        relu_peaks = F.relu(torch.abs(y_true) - self.tau)  # ReLU(|y_n| - \tau)

        # Numerator of the regularization term
        reg_numerator = torch.sum(relu_peaks * abs_diff)

        # Denominator of the regularization term
        reg_denominator = torch.sum(relu_peaks) + self.eta

        # Regularization term
        reg_term = (self.lambda_reg / y_true.size(0)) * (reg_numerator / reg_denominator)

        # Combine MSE and regularization term
        loss = mse_loss + reg_term

        return loss


def train_autoencoder(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=20, patience=5):
    """
    Trains an autoencoder model with early stopping and optional learning rate scheduling.

    Parameters:
    - model: the autoencoder model to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - criterion: loss function (e.g., MSELoss).
    - optimizer: optimizer for model parameters (e.g., Adam).
    - scheduler: optional learning rate scheduler.
    - num_epochs: maximum number of training epochs.
    - patience: number of epochs with no improvement before early stopping.

    Returns:
    - model: the trained model (with best weights restored).
    - best_val_loss: the lowest validation loss achieved.
    - training_losses: list of training loss values per epoch.
    - validation_losses: list of validation loss values per epoch.
    """

    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize variables for early stopping
    best_model = None
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Setup logging for progress reporting
    logging.basicConfig(level=logging.INFO)

    # Store loss history
    training_losses = []
    validation_losses = []

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        # Iterate over training batches
        for data in train_loader:
            # Move input data to device
            inputs = data.to(dtype=torch.float).to(device)

            # Forward pass through the model
            _, outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)  # Reconstruction loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Train loss
        model.eval()
        train_loss = 0.0
        total_train_samples = 0

        with torch.no_grad():
            for data in train_loader:
                inputs = data.to(dtype=torch.float).to(device)
                batch_size = inputs.size(0)
                total_train_samples += batch_size

                _, outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)
                train_loss += loss.item() * batch_size

        # Validation loss
        model.eval()
        val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for data in val_loader:
                inputs = data.to(dtype=torch.float).to(device)
                batch_size = inputs.size(0)
                total_val_samples += batch_size

                _, outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * batch_size

        # Average losses
        train_loss /= total_train_samples
        val_loss /= total_val_samples

        # Log progress for this epoch
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        # Step the learning rate scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # Reduce on plateau uses val loss
            else:
                scheduler.step()  # Others usually step per epoch

        # Early stopping condition (tolerance buffer)
        if val_loss + 0.00025 < best_val_loss:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save best model weights if current val loss is lowest
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())

        # Trigger early stopping if no improvement for `patience` epochs
        if epochs_no_improve >= patience:
            logging.info("Early stopping triggered.")
            break

    # Restore best model weights before returning
    if best_model is not None:
        model.load_state_dict(best_model)
        logging.info(f"Best model loaded with validation loss: {best_val_loss:.4f}")

    return model, best_val_loss, training_losses, validation_losses


def evaluate_autoencoder(model, test_loader):
    """
    Evaluates a trained autoencoder on test data and returns the latent embeddings and average reconstruction loss.

    Parameters:
    - model: the trained autoencoder model.
    - test_loader: DataLoader for test data.

    Returns:
    - all_embeddings (Tensor): stacked latent vectors for the entire test set.
    - test_loss (float): average reconstruction loss (MSE) on the test set.
    """

    # List to store all latent representations from the encoder
    all_embeddings = []

    # Choose GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    # Initialize test loss and define loss function
    test_loss = 0.0
    criterion = nn.MSELoss()

    # Disable gradient tracking for efficiency
    with torch.no_grad():
        for data in test_loader:
            # Move inputs to correct device and data type
            inputs = data.to(dtype=torch.float).to(device)

            # Forward pass through the model to get encoded and decoded output
            embeddings, outputs, _ = model(inputs)

            # Calculate reconstruction loss
            loss = criterion(outputs, inputs)
            test_loss += loss.item()

            # Save the embeddings (latent representations)
            all_embeddings.append(embeddings)

    # Stack all latent vectors into a single tensor
    all_embeddings = torch.vstack(all_embeddings)

    # Compute average loss over all batches
    test_loss /= len(test_loader)

    # Print test loss for monitoring
    print("Test Loss:", test_loss)

    return all_embeddings, test_loss


def extract_embeddings(model, data_loader, saving_path):
    """
    Extracts latent embeddings from an autoencoder model using a given data loader,
    and saves them to a file using pickle.

    Parameters:
    - model: trained autoencoder from which to extract embeddings.
    - data_loader: DataLoader providing the data for which embeddings are to be computed.
    - saving_path (str): File path where the extracted embeddings will be saved.
    """

    # Run evaluation to extract latent vectors (embeddings) from the encoder
    embeddings, _ = evaluate_autoencoder(model, data_loader)

    # Move embeddings from GPU to CPU, detach them from the computation graph,
    # and convert to NumPy array for saving
    embeddings = embeddings.cpu().detach().numpy()

    # Save embeddings to a binary file using Python's pickle module
    with open(saving_path, "wb") as f:
        pickle.dump(embeddings, f)
