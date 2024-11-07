import torch
from torch.utils.data import DataLoader
from helpers import MCSims
from models.auto_encoder import AutoEncoder
import os

# Set the environment variable to manage CUDA memory more efficiently
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Prepare dataset and dataloader with smaller batch size
dataset = MCSims()
dataloader = DataLoader(dataset, batch_size=64,
                        shuffle=False)  # Reduced batch size

# Load model
autoencoder = AutoEncoder([[7], [7], [5], [3], [3]]).to(device)
autoencoder.load_state_dict(torch.load('model.pth'))
autoencoder.eval()

# Initialize lists for storing encoded and decoded data
encoded_data = []
decoded_data = []

with torch.no_grad():
    for i, batch in enumerate(dataloader):
        print("encoding batch: ", i, "of", len(dataloader), end='\r')

        # Move batch to device, encode, and store results
        batch = batch.to(device)
        encoded_batch = autoencoder.encoder(batch)
        # Move to CPU to free GPU memory
        encoded_data.append(encoded_batch.cpu())
        torch.cuda.empty_cache()  # Clear cached memory after encoding

    for i, encoded_batch in enumerate(encoded_data):
        print("decoding batch: ", i, "of", len(dataloader), end='\r')

        # Move encoded batch back to GPU for decoding
        encoded_batch = encoded_batch.to(device)
        decoded_batch = autoencoder.decoder(encoded_batch)
        # Move to CPU to free GPU memory
        decoded_data.append(decoded_batch.cpu())
        torch.cuda.empty_cache()  # Clear cached memory after decoding

# Concatenate encoded and decoded data
encoded_data = torch.cat(encoded_data, dim=0)
decoded_data = torch.cat(decoded_data, dim=0)

# Save encoded and decoded data
torch.save(encoded_data, 'encoded_data.pth')
torch.save(decoded_data, 'decoded_data.pth')
