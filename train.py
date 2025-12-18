from model.mobility_data_manager import MobilityDataManager, DataConfig
from model.deepgravity import DeepGravityNN

from pathlib import Path
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch.nn.functional as F

import geopandas as gpd
import numpy as np
import argparse
import logging
import torch
import pickle

data_config = DataConfig()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type = int, default = 42, help = "Random seed for reproducibility")
parser.add_argument("--split_ratio", type = float, default = 0.5, help = "Train/Validation split ratio")
parser.add_argument("--info", action = "store_true", help = "Enable info verbose logging")
parser.add_argument("--batch_size", type = int, default = 16, help = "Batch size for training")
parser.add_argument("--epochs", type = int, default = 20, help = "Number of training epochs")
parser.add_argument("--learning_rate", type = float, default = 5e-6, help = "Learning rate for optimizer")
parser.add_argument("--momentum", type = float, default = 0.9, help = "Momentum for RMSprop optimizer")
parser.add_argument("--log_interval", type=int, default=10, help="Print progress every N batches")
args = parser.parse_args()

args.info = True


@dataclass
class TrainConfig:
	TRAIN_SECTION_PATH = "data/processed_data/sections_features_train.parquet"
	VAL_SECTION_PATH = "data/processed_data/sections_features_val.parquet"
	TRAIN_TESSELLATION_PATH = "data/processed_data/tessellation_25km_train.parquet"
	VAL_TESSELLATION_PATH = "data/processed_data/tessellation_25km_val.parquet"

	SECTIONS_PATH = "data/processed_data/sections_features.parquet"
	TESSELLATION_PATH = "data/processed_data/tessellation_25km.parquet"

	SAVE_MODEL_PATH = "models/deepgravity_model.pth"

	#  Paths to save processed data (AT THE MOMENT I CANNOT SAVE IN PICKLE DUE TO INCOMPATIBILITY ISSUES)
	TRAIN_DATA_PATH = "data/processed_data/data_train.pkl"
	VAL_DATA_PATH = "data/processed_data/data_val.pkl"

	BASE_DIR: Path = Path(__file__).resolve().parent
	TRAIN_SECTION_PATH = (BASE_DIR / TRAIN_SECTION_PATH).resolve()
	VAL_SECTION_PATH = (BASE_DIR / VAL_SECTION_PATH).resolve()
	TRAIN_TESSELLATION_PATH = (BASE_DIR / TRAIN_TESSELLATION_PATH).resolve()
	VAL_TESSELLATION_PATH = (BASE_DIR / VAL_TESSELLATION_PATH).resolve()
	TRAIN_DATA_PATH = (BASE_DIR / TRAIN_DATA_PATH).resolve()
	VAL_DATA_PATH = (BASE_DIR / VAL_DATA_PATH).resolve()
	SAVE_MODEL_PATH = (BASE_DIR / SAVE_MODEL_PATH).resolve()

	DTYPE: torch.dtype = torch.float16
	FLOAT_DTYPE: np.dtype = np.float16
	CRS_METRIC: int = 3003  # Italy TM (EPSG:3003)
	INFO_VERBOSE: bool = False
	SEED: int = args.seed
	SPLIT_RATIO: float = args.split_ratio
	BATCH_SIZE: int = args.batch_size
	EPOCHS: int = args.epochs
	LEARNING_RATE: float = args.learning_rate
	MOMENTUM: float = args.momentum
	LOG_INTERVAL: int = args.log_interval


config = TrainConfig()


def logger_setup():
	logging.basicConfig(
		level = logging.INFO,
		# %(name)s will print "MobilityDataManager" or "root"
		format = "%(asctime)s - [%(name)s] - %(message)s",
		datefmt = "%H:%M:%S",
		force = True
	)
	logger = logging.getLogger("Training")  # Give your main script a name too!
	return logger


logger = logger_setup()


def get_device():
	if torch.cuda.is_available():
		return torch.device("cuda")
	elif torch.backends.mps.is_available():
		return torch.device("mps")  # <--- Apple M1/M2/M3 GPU
	else:
		return torch.device("cpu")


device = get_device()
logger.info(f"Using device: {device}")


# --- Helper Functions ---

def get_cpc(y_true, y_pred):
	"""
	Common Part of Commuters (CPC) metric.
	CPC = 2 * sum(min(y_true, y_pred)) / (sum(y_true) + sum(y_pred))
	Since both are probabilities summing to 1, CPC = sum(min(y_true, y_pred)).
	"""
	return torch.sum(torch.min(y_true, y_pred)).item()


def mobility_collate_fn(batch):
	"""
	Custom collate function to handle variable numbers of destinations.

	Args:
		batch: List of tuples (X, y) from MobilityDataManager

	Returns:
		X_cat: Concatenated features (Total_Dests, Features)
		y_cat: Concatenated targets (Total_Dests,)
		batch_indices: Index tensor mapping each row to its origin in the batch (Total_Dests,)
	"""
	X_list = []
	y_list = []
	batch_indices_list = []

	for i, (X, y) in enumerate(batch):
		# X shape: (N_dests, Features)
		# y shape: (N_dests,)

		X_list.append(X)
		y_list.append(y)

		# Create an index vector [i, i, i...] of length N_dests
		# This tells the model which origin these destinations belong to
		batch_indices_list.append(torch.full((X.shape[0],), i, dtype = torch.long))

	# Concatenate everything into one massive tensor
	X_cat = torch.cat(X_list, dim = 0)
	y_cat = torch.cat(y_list, dim = 0)
	batch_indices = torch.cat(batch_indices_list, dim = 0)

	return X_cat, y_cat, batch_indices

# --- Main Execution Flow ---

if not (config.TRAIN_SECTION_PATH.exists() and
        config.VAL_SECTION_PATH.exists() and
        config.TRAIN_TESSELLATION_PATH.exists() and
        config.VAL_TESSELLATION_PATH.exists()
):
	# 1. Prepare Training and Validation Data
	logger.info("⚙️  Preparing training and validation datasets...")

	# we select 50% tiles for training and 50% for validation
	sections_gdf = gpd.read_parquet(config.SECTIONS_PATH)
	tessellation_gdf = gpd.read_parquet(config.TESSELLATION_PATH)

	# Perform a spatial join to filter sections and tessellation to only those that intersect
	joined = sections_gdf.sjoin(tessellation_gdf, how = "inner", predicate = "intersects")
	tessellation_gdf = tessellation_gdf.loc[joined['index_right'].unique()]

	# Split tessellation into train and val
	tile_train, tile_val = train_test_split(
		tessellation_gdf,
		test_size = config.SPLIT_RATIO,
		random_state = config.SEED,
	)

	# Get sections for train and val based on the tiles
	sections_train = joined[joined['index_right'].isin(tile_train.index)].drop(columns = ['index_right'])
	sections_val = joined[joined['index_right'].isin(tile_val.index)].drop(columns = ['index_right'])

	# Save the datasets
	sections_train.to_parquet(config.TRAIN_SECTION_PATH)
	sections_val.to_parquet(config.VAL_SECTION_PATH)
	tile_train.to_parquet(config.TRAIN_TESSELLATION_PATH)
	tile_val.to_parquet(config.VAL_TESSELLATION_PATH)

	logger.info("✅ Training and validation datasets prepared and saved.")

# 2. Load Data using MobilityDataManager
logger.info("📥 Loading training and validation data...")

data_config.SECTIONS_PATH = config.TRAIN_SECTION_PATH
data_config.TESSELLATION_PATH = config.TRAIN_TESSELLATION_PATH
train_data = MobilityDataManager(data_config)

data_config.SECTIONS_PATH = config.VAL_SECTION_PATH
data_config.TESSELLATION_PATH = config.VAL_TESSELLATION_PATH
data_config.SCALER = train_data.scaler  # Use the same scaler for validation
val_data = MobilityDataManager(data_config)

logger.info("✅ Training and validation data loaded and saved.")

# 3. Initialize Dataloaders
train_loader = DataLoader(
	train_data,
	batch_size = config.BATCH_SIZE,
	shuffle = True
)
val_loader = DataLoader(
	val_data,
	batch_size = config.BATCH_SIZE,
	shuffle = False
)

# 4. Initialize and Train DeepGravityNN Model
logger.info("🚀 Initializing and training DeepGravityNN model...")
model = DeepGravityNN()
model.to(device)

# 5. RMSprop with momentum 0.9 and criterion CrossEntropyLoss
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=config.LEARNING_RATE,
    momentum=config.MOMENTUM
)

# Softmax is handled internally by CrossEntropyLoss when targets are probabilities
criterion = torch.nn.CrossEntropyLoss()

# --- Training Loop ---
print(f"\n🚀 Starting Training: {config.EPOCHS} Epochs | Batch: {config.BATCH_SIZE} | Device: {device}")
print(f"ℹ️  Logging every {config.LOG_INTERVAL} batches")

for epoch in range(config.EPOCHS):
	model.train()

	# Global metrics for the entire epoch
	total_loss = 0.0
	total_cpc = 0.0

	# Running metrics for the print interval
	running_loss = 0.0
	running_cpc = 0.0

	for batch_idx, (X, y, batch_indices) in enumerate(train_loader):
		X = X.to(device).float()
		y = y.to(device).float()
		batch_indices = batch_indices.to(device)

		optimizer.zero_grad()
		logits = model(X).squeeze()

		# --- Compute Loss & CPC per Origin ---
		batch_loss_accum = 0.0
		batch_cpc_accum = 0.0
		unique_origins = torch.unique(batch_indices)
		n_origins = len(unique_origins)

		for o_idx in unique_origins:
			mask = (batch_indices == o_idx)
			local_logits = logits[mask]
			local_targets = y[mask]

			# Softmax for CPC
			local_probs = F.softmax(local_logits, dim = 0)
			batch_cpc_accum += get_cpc(local_targets, local_probs)

			# LogSoftmax for Loss
			log_probs = F.log_softmax(local_logits, dim = 0)
			loss_i = -torch.sum(local_targets * log_probs)
			batch_loss_accum += loss_i

		# Average over the batch (number of origins)
		loss = batch_loss_accum / n_origins
		avg_batch_cpc = batch_cpc_accum / n_origins

		# Backward pass
		loss.backward()
		optimizer.step()

		# Update Totals (For Epoch Summary)
		total_loss += loss.item()
		total_cpc += batch_cpc_accum  # Sum of CPCs (will divide by total dataset size later)

		# Update Running (For Interval Logging)
		running_loss += loss.item()
		running_cpc += avg_batch_cpc

		# --- Interval Logging ---
		if (batch_idx + 1) % config.LOG_INTERVAL == 0:
			current_loss = running_loss / config.LOG_INTERVAL
			current_cpc = running_cpc / config.LOG_INTERVAL

			print(
				f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(train_loader)} | "
				f"Loss: {current_loss:.5f} | CPC: {current_cpc:.4f}"
				)

			# Reset running counters
			running_loss = 0.0
			running_cpc = 0.0

	# --- Epoch Summary ---
	avg_train_loss = total_loss / len(train_loader)
	avg_train_cpc = total_cpc / len(train_data)  # Divide by total N of origins

	# --- Validation Loop ---
	model.eval()
	val_loss = 0.0
	val_cpc = 0.0

	with torch.no_grad():
		for X, y, batch_indices in val_loader:
			X, y, batch_indices = X.to(device).float(), y.to(device).float(), batch_indices.to(device)
			logits = model(X).squeeze()
			unique_origins = torch.unique(batch_indices)

			batch_loss_val = 0.0

			for o_idx in unique_origins:
				mask = (batch_indices == o_idx)
				local_logits = logits[mask]
				local_targets = y[mask]

				local_probs = F.softmax(local_logits, dim = 0)
				val_cpc += get_cpc(local_targets, local_probs)

				log_probs = F.log_softmax(local_logits, dim = 0)
				batch_loss_val += -torch.sum(local_targets * log_probs)

			val_loss += (batch_loss_val / len(unique_origins)).item()

	avg_val_loss = val_loss / len(val_loader)
	avg_val_cpc = val_cpc / len(val_data)

	print(
		f"✅ Epoch {epoch + 1} DONE | "
		f"Train Loss: {avg_train_loss:.5f} CPC: {avg_train_cpc:.4f} | "
		f"Val Loss: {avg_val_loss:.5f} CPC: {avg_val_cpc:.4f}"
		)

print("\n🏁 Training Complete!")

# Save the trained model
config.SAVE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
logger.info(f"💾 Model saved to {config.SAVE_MODEL_PATH}")

