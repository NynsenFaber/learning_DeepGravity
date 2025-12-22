from model.mobility_data_manager import MobilityDataManager, DataConfig
from model.deepgravity import DeepGravityNN

from pathlib import Path
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import geopandas as gpd
import numpy as np
import argparse
import logging
import torch
import time

data_config = DataConfig()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type = int, default = 42, help = "Random seed for reproducibility")
parser.add_argument("--split_ratio", type = float, default = 0.5, help = "Train/Validation split ratio")
parser.add_argument("--info", action = "store_true", help = "Enable info verbose logging")
parser.add_argument("--batch_size", type = int, default = 64, help = "Batch size for training")
parser.add_argument("--val_batch_size", type = int, default = 50, help = "Number of random batches for fast validation")
parser.add_argument("--epochs", type = int, default = 20, help = "Number of training epochs")
parser.add_argument("--learning_rate", type = float, default = 5e-4, help = "Learning rate for optimizer")
parser.add_argument("--momentum", type = float, default = 0.9, help = "Momentum for RMSprop optimizer")
parser.add_argument("--log_interval", type = int, default = 10, help = "Print progress every N batches")
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
	LOG_FILE_PATH: Path = (BASE_DIR / "training_log.txt").resolve()

	DTYPE: torch.dtype = torch.float16
	FLOAT_DTYPE: np.dtype = np.float16
	CRS_METRIC: int = 3003  # Italy TM (EPSG:3003)
	INFO_VERBOSE: bool = False
	SEED: int = args.seed
	SPLIT_RATIO: float = args.split_ratio
	BATCH_SIZE: int = args.batch_size
	VAL_BATCH_SIZE: int = args.val_batch_size  # How many batches to check during validation
	EPOCHS: int = args.epochs
	LEARNING_RATE: float = args.learning_rate
	MOMENTUM: float = args.momentum
	LOG_INTERVAL: int = args.log_interval


config = TrainConfig()


# --- Logger Setup ---
def setup_logger():
	logger = logging.getLogger("DeepGravityTrain")
	logger.setLevel(logging.INFO)
	logger.handlers = []  # Clear existing handlers

	formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt = "%H:%M:%S")

	# 1. Console Handler
	ch = logging.StreamHandler()
	ch.setFormatter(formatter)
	logger.addHandler(ch)

	# 2. File Handler (Appends to file)
	fh = logging.FileHandler(config.LOG_FILE_PATH, mode = 'a')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger


logger = setup_logger()


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

def cpc_loss(y_true, y_pred, mask):
	"""
	Computes Common Part of Commuters (Sorensen Index) in a vectorized way.
	y_true: (Batch, Samples) - True Probabilities
	y_pred: (Batch, Samples) - Predicted Probabilities (Softmaxed)
	mask:   (Batch, Samples) - Boolean Mask (False = Padding)
	"""
	# 1. Zero out padding in predictions just to be safe (though softmax handles -inf)
	y_pred = y_pred * mask

	# 2. Minimum of true flow vs predicted flow per destination
	min_flow = torch.min(y_true, y_pred)

	# 3. Sum over destinations (dim=1) to get CPC per origin
	cpc_per_origin = 2.0 * torch.sum(min_flow, dim = 1) / (
			torch.sum(y_true, dim = 1) + torch.sum(y_pred, dim = 1) + 1e-8)

	# 4. Average over the batch
	return torch.mean(cpc_per_origin)


def validate(model, loader, device):
	model.eval()  # Set to evaluation mode
	val_loss = 0.0
	val_cpc = 0.0

	with torch.no_grad():
		for X, y, mask in loader:
			X, y, mask = X.to(device).float(), y.to(device).float(), mask.to(device)
			B, S, F_dim = X.shape

			# Forward
			logits = model(X.view(-1, F_dim)).view(B, S)
			logits = logits.masked_fill(~mask, float('-inf'))

			# Probs & Loss
			log_probs = F.log_softmax(logits, dim = 1)
			probs = F.softmax(logits, dim = 1)

			# Loss Calculation
			loss_elements = -(y * log_probs).masked_fill(~mask, 0.0)
			batch_loss = torch.mean(torch.sum(loss_elements, dim = 1))

			# Metric Calculation
			batch_cpc = cpc_loss(y, probs, mask)

			val_loss += batch_loss.item()
			val_cpc += batch_cpc.item()

	return val_loss / len(loader), val_cpc / len(loader)


def validate_fast(model, loader, device, num_batches = 50):
	"""
	Validates on a random subset of batches to save time.
	"""
	model.eval()
	val_loss = 0.0
	val_cpc = 0.0

	# Create an iterator from the loader
	loader_iter = iter(loader)
	total_batches = len(loader)

	# We validate on min(num_batches, total_batches)
	batches_to_check = min(num_batches, total_batches)

	with torch.no_grad():
		for _ in range(batches_to_check):
			try:
				# Try to get next batch
				X, y, mask = next(loader_iter)
			except StopIteration:
				# If iterator runs out (shouldn't happen with correct length check but safe to handle)
				break

			X, y, mask = X.to(device).float(), y.to(device).float(), mask.to(device)
			B, S, F_dim = X.shape

			logits = model(X.view(-1, F_dim)).view(B, S)
			logits = logits.masked_fill(~mask, float('-inf'))

			log_probs = F.log_softmax(logits, dim = 1)
			probs = F.softmax(logits, dim = 1)

			# Loss
			loss_elements = -(y * log_probs).masked_fill(~mask, 0.0)
			batch_loss = torch.mean(torch.sum(loss_elements, dim = 1))

			# CPC
			batch_cpc = cpc_loss(y, probs, mask)

			val_loss += batch_loss.item()
			val_cpc += batch_cpc.item()

	return val_loss / batches_to_check, val_cpc / batches_to_check


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

# --- 3. Initialize Model ---
# We pass the detected dimension to the model config
model = DeepGravityNN()
model = model.to(device).float()  # Ensure Model is Float32

optimizer = torch.optim.RMSprop(
	model.parameters(),
	lr = config.LEARNING_RATE,
	momentum = config.MOMENTUM
)

logger.info(f"🏁 Start Training: {config.EPOCHS} Epochs | Val Sample: {config.VAL_BATCH_SIZE} batches")

# --- NEW: Learning Rate Scheduler ---
# This will monitor 'val_loss'. If it doesn't improve for 'patience' epochs,
# it reduces the Learning Rate by a factor of 'factor'.
scheduler = ReduceLROnPlateau(
	optimizer,
	mode = 'min',
	factor = 0.5,  # Cut LR in half
	patience = 2,  # Wait 2 validation checks before cutting
)

logger.info(f"🧠 Model Initialized. LR: {config.LEARNING_RATE} | Batch: {config.BATCH_SIZE}")
logger.info(f"🏁 Start Training: {config.EPOCHS} Epochs")

# --- 4. Main Training Loop ---
for epoch in range(config.EPOCHS):
	model.train()
	running_loss = 0.0
	running_cpc = 0.0
	t0 = time.time()

	for batch_idx, (X, y, mask) in enumerate(train_loader):
		X, y, mask = X.to(device).float(), y.to(device).float(), mask.to(device)

		optimizer.zero_grad()

		# Forward
		B, S, F_dim = X.shape
		logits = model(X.view(-1, F_dim)).view(B, S)
		logits = logits.masked_fill(~mask, float('-inf'))

		# Loss
		log_probs = F.log_softmax(logits, dim = 1)
		loss_elements = -(y * log_probs).masked_fill(~mask, 0.0)
		loss = torch.mean(torch.sum(loss_elements, dim = 1))

		# CPC (for tracking)
		probs = F.softmax(logits, dim = 1)
		cpc = cpc_loss(y, probs, mask)

		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()

		running_loss += loss.item()
		running_cpc += cpc.item()

		# --- LOGGING ---
		if (batch_idx + 1) % config.LOG_INTERVAL == 0:
			avg_train_loss = running_loss / config.LOG_INTERVAL
			avg_train_cpc = running_cpc / config.LOG_INTERVAL

			# Fast Validation
			t_val_start = time.time()
			val_loss, val_cpc = validate_fast(model, val_loader, device, num_batches = config.VAL_BATCH_SIZE)
			t_val_end = time.time()

			# --- STEP SCHEDULER ---
			# We step the scheduler based on Validation Loss
			scheduler.step(val_loss)

			current_lr = optimizer.param_groups[0]['lr']

			log_msg = (
				f"Ep {epoch + 1:02d} | Bt {batch_idx + 1:04d} | "
				f"Tr Loss: {avg_train_loss:.4f} CPC: {avg_train_cpc:.4f} | "
				f"Val Loss: {val_loss:.4f} CPC: {val_cpc:.4f} | "
				f"LR: {current_lr:.1e} | "
				f"T: {time.time() - t0:.1f}s"
			)

			logger.info(log_msg)

			# Reset
			running_loss = 0.0
			running_cpc = 0.0
			t0 = time.time()
			model.train()

	# End of Epoch
	logger.info(f"✅ Epoch {epoch + 1} Completed.")

print("\n🏁 Training Complete!")
config.SAVE_MODEL_PATH.parent.mkdir(parents = True, exist_ok = True)
torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
logger.info(f"💾 Model saved to {config.SAVE_MODEL_PATH}")
