"""
DeepGravity PyTorch Implementation
==================================================================

Description:
    This script implements the Deep Gravity model (Simini et al., 2021) using
    standard PyTorch design patterns.

    - Training: The model returns raw 'logits' (scores). The Loss function
      (nn.CrossEntropyLoss) handles the LogSoftmax internally for maximum
      numerical stability.
    - Inference: We explicitly apply Softmax to the logits to generate
      interpretable probabilities.

Architecture:
    - Input: Matrix (N_destinations, 39_features) representing one Origin.
    - Hidden: 15-layer Shared MLP (Pointwise 1x1 Conv).
    - Output: Logits (Raw scores, range -inf to +inf).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
	INPUT_DIM: int = 39
	HIDDEN_LAYERS: List[int] = field(default_factory = lambda: [256] * 6 + [128] * 9)
	LEARNING_RATE: float = 0.001


class DeepGravityNN(nn.Module):
	"""
	A row-wise deep neural network for predicting flow scores (logits).
	"""

	def __init__(self, config: ModelConfig = ModelConfig()):
		super().__init__()

		# --- Architecture Configuration ---
		# 15 hidden layers: Bottom 6 (256 units), Top 9 (128 units)
		hidden_layer_sizes = config.HIDDEN_LAYERS

		layers = []
		current_dim = config.INPUT_DIM

		# --- Shared Feature Extractor ---
		for next_dim in hidden_layer_sizes:
			layers.append(nn.Linear(current_dim, next_dim))
			layers.append(nn.LeakyReLU())
			current_dim = next_dim

		self.feature_extractor = nn.Sequential(*layers)

		# Final projection: (Hidden_Dim) -> Scalar Logit
		self.output_layer = nn.Linear(current_dim, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass returning RAW LOGITS.

		Args:
			x: Tensor of shape (N, 39).
		Returns:
			logits: Tensor of shape (N,).
					Values are NOT normalized (can be negative or positive).
		"""
		# 1. Feature Extraction (N, 39) -> (N, 128)
		features = self.feature_extractor(x)

		# 2. Score Projection (N, 128) -> (N, 1)
		scalars = self.output_layer(features)

		# 3. Flatten to 1D vector (N,)
		logits = scalars.view(-1)

		# CRITICAL: We return logits. We do NOT apply Softmax here.
		# This allows CrossEntropyLoss to work safely.
		return logits


# --- Helper Functions for Clean Workflow ---

def train_step(model, optimizer, criterion, x, y_target):
	"""
	Performs one step of training using Logits + CrossEntropyLoss.
	"""
	model.train()  # Enable Dropout/BatchNorm (if present)
	optimizer.zero_grad()

	# 1. Get Raw Logits
	logits = model(x)

	# 2. Calculate Loss
	# nn.CrossEntropyLoss takes (Logits, Soft_Targets) automatically
	loss = criterion(logits, y_target)

	# 3. Backprop
	loss.backward()
	optimizer.step()

	return loss.item()


def predict(model, x):
	"""
	Performs inference by applying Softmax to the model's logits.
	"""
	model.eval()  # Disable Dropout/BatchNorm

	with torch.no_grad():
		logits = model(x)

		# Apply Softmax HERE for the user
		probs = F.softmax(logits, dim = 0)

	return probs


# --- Execution Block ---

if __name__ == "__main__":
	torch.manual_seed(42)

	# 1. Setup
	model = DeepGravityNN()

	# Standard PyTorch Loss (Handles Softmax internally via LogSumExp trick)
	# Note: Requires PyTorch >= 1.10 for soft target support
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = 0.001)

	print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

	# 2. Create Dummy Data (Single Example)
	N_destinations = 10
	X = torch.randn(N_destinations, 39)

	# Targets (must sum to 1.0)
	y_target = torch.rand(N_destinations)
	y_target = y_target / y_target.sum()

	print("\n--- Starting Training ---")

	# 3. Training Loop Simulation
	for epoch in range(5):
		loss = train_step(model, optimizer, criterion, X, y_target)
		print(f"Epoch {epoch + 1}: Loss = {loss:.6f}")

	print("\n--- Running Inference ---")

	# 4. Prediction (Inference Mode)
	final_probs = predict(model, X)

	# 5. Verification
	print(f"Predicted Probabilities: \n{final_probs[:5]} ...")
	print(f"Sum of Probabilities: {final_probs.sum().item():.6f} (Should be 1.0)")
