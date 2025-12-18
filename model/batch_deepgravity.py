"""
DeepGravity PyTorch Implementation (Batched)
============================================================

Description:
    This script implements the Deep Gravity model for variable-length batches
    using standard PyTorch design patterns (Logits + CrossEntropyLoss).

    - Model Output: Returns 'Masked Logits'. Padding positions are set to
      -Infinity so they have 0 impact on the Softmax denominator.
    - Loss: Uses nn.CrossEntropyLoss (which fuses LogSoftmax+NLL) for
      maximum stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

class BatchDeepGravityNN(nn.Module):
    """
    A Deep Learning implementation of the Gravity Model for variable-length batches.

    Returns:
        Logits (Scores).
        The padding positions are masked with -inf to ensure numerical correctness
        in downstream Loss or Softmax functions.
    """
    def __init__(self, input_dim: int = 39):
        super(BatchDeepGravityNN, self).__init__()

        # Architecture hyper-parameters
        hidden_layer_sizes = [256] * 6 + [128] * 9

        layers = []
        current_dim = input_dim

        # --- Shared Feature Extractor ---
        for next_dim in hidden_layer_sizes:
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.LeakyReLU())
            current_dim = next_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Final projection to scalar score
        self.output_layer = nn.Linear(current_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning MASKED LOGITS.

        Args:
            x: Input tensor (Batch, Max_Rows, Features).
            mask: Boolean tensor (Batch, Max_Rows). True = Real Data.

        Returns:
            logits: Tensor (Batch, Max_Rows).
                    Real data contains scores.
                    Padding data contains -inf.
        """
        # 1. Feature Extraction (Batch, Max_Rows, 39) -> (Batch, Max_Rows, 128)
        features = self.feature_extractor(x)

        # 2. Score Projection (Batch, Max_Rows, 128) -> (Batch, Max_Rows, 1)
        scores = self.output_layer(features)

        # Flatten to (Batch, Max_Rows)
        logits = scores.squeeze(-1)

        # 3. Masking (Critical for Logits)
        # We fill padding with -Infinity.
        # Why? Because in Loss/Softmax: e^(-inf) = 0.
        # This ensures padding doesn't affect the Softmax denominator.
        logits = logits.masked_fill(~mask, float('-inf'))

        # Return RAW LOGITS (No Softmax here!)
        return logits


# --- Helper Functions for Clean Workflow ---

def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Handles padding for variable length sequences in the batch.
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = torch.tensor([x.shape[0] for x in inputs])

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0.0)

    # Generate Mask (True = Real Data)
    max_len = inputs_padded.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    return inputs_padded, targets_padded, mask

def train_step(model, optimizer, criterion, x, mask, y_target):
    """
    Training step using Logits and Stable Cross Entropy.
    """
    model.train()
    optimizer.zero_grad()

    # 1. Get Masked Logits (-inf at padding)
    logits = model(x, mask)

    # 2. Calculate Loss
    # nn.CrossEntropyLoss works with (Logits, Soft_Targets).
    # Since padding logits are -inf and targets are 0, they contribute 0 to loss.
    loss = criterion(logits, y_target)

    loss.backward()
    optimizer.step()

    return loss.item()

def predict(model, x, mask):
    """
    Inference step: Explicitly applies Softmax to convert Logits -> Probs.
    """
    model.eval()
    with torch.no_grad():
        logits = model(x, mask)

        # Apply Softmax across the rows (dim=1)
        probs = F.softmax(logits, dim=1)

    return probs


# --- Execution Block ---

if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. Setup
    model = BatchDeepGravityNN(input_dim=39)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Standard PyTorch Loss (Stable)
    # Note: Handles Softmax internally.
    criterion = nn.CrossEntropyLoss()

    # 2. Create Dummy Data (Batch of 3, variable lengths)
    data = []
    for _ in range(3):
        rows = torch.randint(low=5, high=20, size=(1,)).item()
        X = torch.randn(rows, 39)
        y = torch.rand(rows)
        y = y / y.sum()
        data.append((X, y))

    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=3, collate_fn=custom_collate_fn)

    print("--- Starting Training (Logits Pipeline) ---")

    # 3. Training Loop
    for epoch in range(3):
        total_loss = 0
        for X_batch, y_batch, mask_batch in loader:
            loss = train_step(model, optimizer, criterion, X_batch, mask_batch, y_batch)
            total_loss += loss
        print(f"Epoch {epoch+1}: Avg Batch Loss = {total_loss/len(loader):.6f}")

    print("\n--- Running Inference (Softmax Applied Externally) ---")

    # 4. Inference
    X_sample, y_sample, mask_sample = next(iter(loader))
    final_probs = predict(model, X_sample, mask_sample)

    # 5. Verification
    # Check Sample 0
    valid_len = mask_sample[0].sum()
    print(f"Sample 0 Valid Length: {valid_len}")
    print(f"Sample 0 Total Prob:   {final_probs[0].sum().item():.6f} (Should be 1.0)")
    print(f"Sample 0 Padding Prob: {final_probs[0, valid_len:].sum().item():.6f} (Should be 0.0)")