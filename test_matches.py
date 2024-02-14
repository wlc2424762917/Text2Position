import torch

# Assuming desc0 is your tensor with shape [B, num_obj, dim]
# and matches1 is a tensor with indices, with shape [B, num_hints]

# Example data (replace these with your actual tensors)
desc0 = torch.randn(B, num_obj, dim)  # Replace B, num_obj, dim with actual values
matches1 = torch.tensor(outputs.matches1)  # Use your matches1 tensor here

# Initialize a list to store the extracted objects
extracted_objects = []

for batch_idx in range(matches1.shape[0]):
    # Extract indices for this batch
    indices = matches1[batch_idx]

    # Filter out -1 (or any invalid index)
    valid_indices = indices[indices >= 0]

    # Extract objects corresponding to valid indices
    objs = desc0[batch_idx, valid_indices]

    # Append the extracted objects to the list
    extracted_objects.append(objs)

# extracted_objects now contains the extracted objects for each batch
