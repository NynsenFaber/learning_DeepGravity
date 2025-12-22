import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import tqdm

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from itertools import combinations

import geopandas as gpd
import numpy as np
import torch
import bbhash
import functools

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import shapely.geometry


def not_implemented(func):
	@functools.wraps(func)  # <--- Preserves name and docstring
	def wrapper(*args, **kwargs):
		raise NotImplementedError(f"The method '{func.__name__}' is disabled/not implemented.")

	return wrapper


# --- Configuration ---
@dataclass
class DataConfig:
	"""
	Configuration for data paths, processing parameters, and logging.
	"""
	# Base Paths
	BASE_DIR: Path = Path(__file__).resolve().parent

	# Input File Paths
	COMMUTE_PATH = "../data/processed_data/commute_data.pickle"
	SECTIONS_PATH = "../data/processed_data/sections_features.parquet"
	TESSELLATION_PATH = "../data/processed_data/tessellation_25km.parquet"

	# Adjust these relative paths as needed for your project structure
	COMMUTE_PATH: Path = (BASE_DIR / COMMUTE_PATH).resolve()
	SECTIONS_PATH: Path = (BASE_DIR / SECTIONS_PATH).resolve()
	TESSELLATION_PATH: Path = (BASE_DIR / TESSELLATION_PATH).resolve()

	# Processing Constants
	EXCLUDE_COLS: Tuple[str, ...] = ('geometry', 'SEZ2011')

	DTYPE: torch.dtype = torch.float16
	FLOAT_DTYPE: np.dtype = np.float16
	CRS_METRIC: int = 3003  # Italy TM (EPSG:3003)

	# Logging Control
	VERBOSE: bool = True

	# Negative Sampling
	SAMPLES_PER_ORIGIN: int = 512

	# Scaler for normalization
	SCALER: Optional[MinMaxScaler] = None


class MobilityDataManager(Dataset):
	"""
	Manages loading, cleaning, normalization, and spatial indexing of mobility data.

	Optimized for PyTorch training:
	- Pre-computes coordinate tensors for vectorized distance calc.
	- Pre-sorts destination lists for deterministic X/y alignment.
	- Uses Set intersection for fast commute filtering.
	"""

	def __init__(self, config: DataConfig = DataConfig()):
		self.config = config
		self._setup_logger()

		## ---- Data Containers ----
		self.features: Optional[torch.Tensor] = None
		self.tiles: Dict[int, shapely.geometry.Polygon] = {}  # tile_idx -> geometry
		self.sections: Dict[int, shapely.geometry.Polygon] = {}  # origin_idx -> geometry
		self.y: Dict[int, torch.Tensor] = {}  # (origin_idx, dest_idx) -> prob
		self.scaler: Optional[MinMaxScaler] = None  # to rescale features
		self.probability: Dict[Tuple[int, int], float] = {}  # (origin_idx, dest_idx) -> prob (does not store zeros)
		self._d_max: Optional[int] = None  # to rescale distance
		self._d_min: Optional[int] = None  # to rescale distance

		# (POSSIBLE UPDATE) for better GPU performance during training and testing
		# at the price of larger space. Store a distance tensor for each tile
		# self.tile_distance_matrices: List[torch.Tensor] = [] # tile_idx -> distance matrix of the tile

		# (COMMENT OUT) If using _process_distance_bbhash
		self._mph: Optional[bbhash.PyMPHF] = None  # Perfectly Minimal Hash for (origin_idx, dest_idx) -> distance
		self.distance: Optional[np.ndarray] = None  # Container for distances

		## ---- Temporary Raw Data ----
		self.commute_data: Dict[str, Dict[str, float]] = {}
		self.sections_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame()
		self.tile_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame()

		## ---- Spatial Indexing ----
		self._tile_idx_to_idx: Dict[int, Set[int]] = defaultdict(set)  # Grid ID -> Set of Section IDx
		self._idx_to_tile_idx: Dict[int, int] = {}  # Section IDx -> Grid ID
		self._id_to_idx: Dict[str, int] = {}  # Section ID (str) -> Section IDx (int)

		## ---- Tensor Artifacts ----
		self.feature_names: List[str] = []
		self._valid_idx: List[int] = []  # Valid idx (Sections that have non zero commute data)

		## ---- Pipeline Execution ----

		# upload section_gdf, tesselation_gdf, commute_data
		self._load_raw_data()

		# spatial join + filtering, creates:
		# self.sections, self._tiles_idx_to_idx, self._idx_to_tile_idx
		self._process_sections_and_tessellations()
		del self.tile_gdf  # free memory

		# store the features in section_gdf as a tensor and normalize it
		self._process_features()
		del self.sections_gdf  # free memory

		self._process_distances_bbhash()

		# It filters and normalizes commute flows within the same tessellation grid
		self._process_commutes()
		del self.commute_data

	def _setup_logger(self):
		# Use the class name so logs are traceable
		self.logger = logging.getLogger(self.__class__.__name__)

		# It's okay to set the Level here if you want to silence this specific class
		# by default, but generally, even this is often done in the main script.
		self.logger.setLevel(logging.INFO if self.config.VERBOSE else logging.WARNING)

	def _load_raw_data(self):
		"""Loads external datasets with validation."""
		self.logger.info("📥  Loading raw datasets...")

		# 1. Commute Data
		if not self.config.COMMUTE_PATH.exists():
			raise FileNotFoundError(f"Missing Commute Data: {self.config.COMMUTE_PATH}")

		with open(self.config.COMMUTE_PATH, 'rb') as f:
			self.commute_data = pickle.load(f)

		# 2. Sections Features
		if not self.config.SECTIONS_PATH.exists():
			raise FileNotFoundError(f"Missing Sections Data: {self.config.SECTIONS_PATH}")

		self.sections_gdf = gpd.read_parquet(self.config.SECTIONS_PATH)
		self.sections_gdf.to_crs(epsg = self.config.CRS_METRIC, inplace = True)

		# 3. Tessellation
		if not self.config.TESSELLATION_PATH.exists():
			raise FileNotFoundError(f"Missing Tessellation Data: {self.config.TESSELLATION_PATH}")

		self.tile_gdf = gpd.read_parquet(self.config.TESSELLATION_PATH)
		# set index of tessellation to consecutive integers for easier mapping
		self.tile_gdf.reset_index(drop = True, inplace = True)
		self.tile_gdf.to_crs(epsg = self.config.CRS_METRIC, inplace = True)

		self.logger.info(f"✅  Data Loaded: {len(self.sections_gdf)} sections.")

	def _process_sections_and_tessellations(self):
		"""
		Performs spatial join and filters dataset to only include valid sections.
		"""
		self.logger.info("⚙️  Building Spatial Index...")

		# 1. GeoDataFrame of points, for spatial join
		points: gpd.GeoDataFrame = gpd.GeoDataFrame(
			geometry = self.sections_gdf.geometry,
			index = self.sections_gdf.index,
			crs = self.config.CRS_METRIC,
		)
		initial_number_of_sections = len(points)
		initial_number_of_tessellations = len(self.tile_gdf)

		# 2. Spatial Join: Points WITHIN Polygons
		joined = points.sjoin(
			self.tile_gdf,
			how = "inner",
			predicate = "within",
		)

		# 3. Remove sections and tessellation that do not appear in the join
		self.sections_gdf = self.sections_gdf.loc[joined.index.unique()]
		self.tile_gdf = self.tile_gdf.loc[joined["index_right"].unique()].reset_index(drop = True)

		# 4. Get spatial indexing maps
		self._id_to_idx = {str(id): idx for idx, id in enumerate(self.sections_gdf.index.astype(str))}
		self._idx_to_id = {idx: str(id) for id, idx in self._id_to_idx.items()}
		for section_id, row in joined.iterrows():
			tile_idx = int(row["index_right"])
			self._tile_idx_to_idx[tile_idx].add(self._id_to_idx[str(section_id)])
			self._idx_to_tile_idx[self._id_to_idx[str(section_id)]] = tile_idx

		# 5. Update data
		self.sections = {self._id_to_idx[id]: row.geometry for id, row in self.sections_gdf.iterrows()}
		self.tiles = {idx: row.geometry for idx, row in self.tile_gdf.iterrows()}

		self.logger.info(
			f"✅  Spatial Index Built: {initial_number_of_sections} sections mapped to {initial_number_of_tessellations} grids.\n"
			f"     After filtering: {len(self.sections_gdf)} sections and {len(self.tile_gdf)} tessellations remain.",
		)

	def _process_features(self):
		"""Standardizes features and creates Coordinate Tensor."""
		if self.sections_gdf.empty:
			self.logger.warning("⚠️  Sections DataFrame is empty.")
			return

		self.logger.info("⚙️  Preprocessing Features & Coordinates...")

		feature_df = self.sections_gdf.drop(columns = 'geometry', errors = 'ignore')

		# 1. Feature Selection
		self.feature_names = [
			c for c in feature_df.columns
			if c not in self.config.EXCLUDE_COLS
		]

		# 2. Feature Tensor Construction
		matrix = feature_df[self.feature_names].values
		if np.isnan(matrix).any():
			matrix = np.nan_to_num(matrix, nan = 0.0)

		# 3. Min-Max Scaling
		if self.config.SCALER is not None:
			self.scaler = self.config.SCALER
		else:
			self.scaler = MinMaxScaler(feature_range = (0, 1))
		matrix = self.scaler.fit_transform(matrix)
		self.features = torch.tensor(matrix, dtype = self.config.DTYPE)

		self.logger.info(
			f"✅  Tensors Ready. Features: {self.features.shape}",
		)

	def _process_distances_bbhash(self):
		"""
		Precompute pairwise distances using Minimal Perfect Hashing (BBHash).
		Drastically reduces memory usage compared to a dictionary.
		"""
		self.logger.info("⚙️  Precomputing Pairwise Distances (BBHash)...")

		# 1. OPTIMIZATION: Extract coordinates.
		# Structure: {id: (x, y)}
		coord_lookup = {
			idx: (poly.centroid.x, poly.centroid.y)
			for idx, poly in self.sections.items()
		}

		# Buffers to collect raw data chunks (List of arrays is faster than resizing one array)
		chunk_keys = []
		chunk_dists = []

		# 2. Iterate Tiles and Collect Data
		for tile_idx, section_idx_set in tqdm.tqdm(
				self._tile_idx_to_idx.items(),
				desc = "Processing Tiles",
				total = len(self._tile_idx_to_idx),
		):
			sorted_idx = sorted(list(section_idx_set))
			n = len(sorted_idx)

			# Skip tiles with only one section in it
			# Zero distance d(u, u) are hard coded in the query distance function
			if n < 2:
				continue

			# Extract points: Shape (N, 2)
			points = np.array([coord_lookup[i] for i in sorted_idx])

			# A. Calculate Distances (Vectorized)
			# Returns flat array: [dist(0,1), dist(0,2)...]
			dists = pdist(points, metric = 'euclidean')

			# B. Generate Keys matching pdist order (Vectorized)
			# pdist order matches upper triangle indices row-by-row
			ids = np.array(sorted_idx, dtype = np.uint64)

			# Generate row/col indices: (0,1), (0,2), (0,3)... (1,2)...
			rows, cols = np.triu_indices(n, k = 1)

			# Map indices back to actual IDs
			id_rows = ids[rows]
			id_cols = ids[cols]

			# Pack into 64-bit int: (u << 32) | v (Bit Packing)
			# First 32 bits for origin (u), last 32 bits for destination (v)
			# Since input is sorted, u < v is guaranteed.
			packed_keys = (id_rows << 32) | id_cols

			# Store chunks
			chunk_keys.append(packed_keys)
			chunk_dists.append(dists)

		# 3. Consolidation (Merge & Deduplicate)
		self.logger.info("⚙️  Merging and deduplicating data...")
		if not chunk_keys:
			self.logger.warning("No pairs found.")
			return

		# Concatenate all chunks
		keys = np.concatenate(chunk_keys)
		dists = np.concatenate(chunk_dists)

		# Clear massive temporary buffers
		del chunk_keys, chunk_dists

		# 3. Normalize Manually (Faster/Lighter than sklearn)
		# Compute min/max on the unique set
		self._d_min = dists.min()
		self._d_max = dists.max()

		# Avoid division by zero if all distances are identical
		if self._d_max > self._d_min:
			# In-place operation (-= and /=) saves memory
			dists -= self._d_min
			dists /= (self._d_max - self._d_min)

		# Cast to float16 for final storage
		# This is the perfect time to drop to 16-bit
		dists = dists.astype(self.config.FLOAT_DTYPE)

		# 4. Build BBHash
		count = len(keys)
		self.logger.info(f"⚙️  Building MPH for {count} unique pairs...")

		# gamma=1.0 is the most memory compact (slower construction)
		# 4 threads for faster build
		self._mph = bbhash.PyMPHF(keys.tolist(), count, 4, 1.0)

		# 5. Populate Value Array
		# We must place distances into the slot bbhash assigns to their key.
		self.distance = np.zeros(count, dtype = self.config.FLOAT_DTYPE)

		# Note: We must loop here because bbhash.lookup is not vectorized.
		# For 100M items, this might take 1-2 minutes, but it's a one-time cost.
		for key, val in tqdm.tqdm(zip(keys, dists), total = count, desc = "Populating MPH"):
			idx = self._mph.lookup(key)
			self.distance[idx] = val

		self.logger.info("✅  Pairwise Distances Computed.")

	def get_distance_bbhash(self, u: int, v: int) -> float:
		""" O(1) Lookup Helper """
		# 1. Handle Self-Distance
		if u == v:
			return 0.0

		# 2. Enforce u < v (because we sorted indices during construction)
		if u > v:
			u, v = v, u

		# 3. Pack Key
		key = (u << 32) | v

		# 4. Lookup
		# Note: If (u,v) was never computed, this returns a garbage value.
		# Ensure you only query valid edges.
		idx = self._mph.lookup(key)
		return self.distance[idx]

	def renormalize_distance(self, raw_distance: float) -> float:
		"""
		Return the original distance from the normalized value.
		"""
		return raw_distance * (self._d_max - self._d_min) + self._d_min

	def _process_commutes(self):
		"""
		Normalizes commute flows constrained to the same tessellation grid.
		"""
		self.logger.info("⚙️  Filtering into Tiles & Normalizing Commute Flows...")

		for tile_idx, tile_sections_idx in tqdm.tqdm(
				self._tile_idx_to_idx.items(),
				desc = "Processing Tiles",
				total = len(self._tile_idx_to_idx),
		):
			# skip empty tiles (This should not happen as we filtered them before)
			if not tile_sections_idx:
				self.logger.warning("⚠️  Empty tile detected during commute processing.")
				continue

			# Process each origin in this tile
			for origin_idx in tile_sections_idx:
				try:
					commutes = self.commute_data[self._idx_to_id[origin_idx]]
				except KeyError:
					# No commute data for this origin
					continue

				non_zero_destinations_idx = set(
					self._id_to_idx[key] for key in commutes.keys()
					if key in self._id_to_idx.keys()  # filtering was not applied to commute data
				)
				destinations = tile_sections_idx.intersection(non_zero_destinations_idx)
				# get commutes only within the same tessellation, set to zero if missing
				filtered_counts = {k: commutes.get(self._idx_to_id[k], 0) for k in destinations}
				total_flow = sum(filtered_counts.values())

				if total_flow > 0:
					for destination_idx, count in filtered_counts.items():
						key = (origin_idx, destination_idx)
						self.probability[key] = count / total_flow

					self._valid_idx.append(origin_idx)
		percentage = len(self._valid_idx) / len(self.sections) * 100
		self.logger.info(f"✅  Commute Flows Processed. The number of valid origins is {percentage:.2f}%. of the total.")

	# --- Dataset Interface Methods ---

	def __len__(self):
		return len(self._valid_idx)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# 1. Get valid origin
		origin_idx = self._valid_idx[idx]
		origin_feat = self.features[origin_idx]

		# 2. Get destinations (Candidates)
		tile_idx = self._idx_to_tile_idx[origin_idx]
		destination_ids = sorted(self._tile_idx_to_idx[tile_idx])

		# 3. Apply Negative Sampling if needed
		target_size = self.config.SAMPLES_PER_ORIGIN  # e.g. 512

		if len(destination_ids) > target_size:
			positives = []
			negatives = []
			for dest_id in destination_ids:
				if (origin_idx, dest_id) in self.probability:
					positives.append(dest_id)
				else:
					negatives.append(dest_id)

			n_pos = len(positives)
			if n_pos >= target_size:
				destination_ids = sorted(positives[:target_size])
			else:
				n_neg_needed = target_size - n_pos
				if len(negatives) > n_neg_needed:
					sampled_negatives = np.random.choice(negatives, size = n_neg_needed, replace = False).tolist()
				else:
					sampled_negatives = negatives
				destination_ids = sorted(positives + sampled_negatives)

		# 4. Prepare Fixed-Size Containers
		# Dimensions
		current_len = len(destination_ids)
		feature_dim = self.features.shape[1]
		# Total Features = Origin_Feat (N) + Dest_Feat (N) + Distance (1)
		total_feature_dim = (feature_dim * 2) + 1

		# Initialize Tensors with Zeros (Padding is implicitly 0)
		X = torch.zeros((target_size, total_feature_dim), dtype = self.config.DTYPE)
		y = torch.zeros(target_size, dtype = self.config.DTYPE)
		mask = torch.zeros(target_size, dtype = torch.bool)  # False = Padding, True = Real

		# 5. Fill Real Data
		if current_len > 0:
			# A. Fetch Features & Distances for real destinations
			dest_feats = self.features[destination_ids]  # (current_len, F)

			distances = torch.zeros(current_len, dtype = self.config.DTYPE)
			for i, dest_idx in enumerate(destination_ids):
				distances[i] = float(self.get_distance_bbhash(origin_idx, dest_idx))

			# B. Construct X components
			origin_repeated = origin_feat.unsqueeze(0).expand(current_len, -1)
			dist_col = distances.unsqueeze(1)

			# C. Concatenate Real Data
			X_real = torch.cat([origin_repeated, dest_feats, dist_col], dim = 1)

			# D. Fill the Fixed-Size Tensors (Top part)
			X[:current_len, :] = X_real

			# Fill y
			for i, dest_idx in enumerate(destination_ids):
				y[i] = self.probability.get((origin_idx, dest_idx), 0.0)

			# Fill Mask
			mask[:current_len] = True

		return X, y, mask

	def __getstate__(self):
		"""Called when pickling: remove the logger."""
		state = self.__dict__.copy()
		if 'logger' in state:
			del state['logger']
		return state

	def __setstate__(self, state):
		"""Called when unpickling: restore state and re-init logger."""
		self.__dict__.update(state)
		# We must re-run setup because the logger was deleted
		self._setup_logger()

	# ----------- Functions that are not optimized for large datasets --------------
	# These are kept for reference or smaller datasets.

	@not_implemented
	def _process_distances_dynamic_dict(self):
		""" Precompute pairwise distances as a dictionary. """
		#### self.distance has 10GB of data! need something better

		self.logger.info("⚙️  Precomputing Pairwise Distances...")

		# 1. OPTIMIZATION: Extract coordinates to a dict of floats once.
		# Accessing .centroid and .x/.y repeatedly in a loop is very expensive.
		# Structure: {id: (x, y)}
		coord_lookup = {
			idx: (poly.centroid.x, poly.centroid.y)
			for idx, poly in self.sections.items()
		}

		for tile_idx, section_idx_set in tqdm.tqdm(
				self._tile_idx_to_idx.items(),
				desc = "Computing Tile: ",
				total = len(self._tile_idx_to_idx),
		):
			# 2. OPTIMIZATION: Sort indices immediately.
			# If we iterate over a sorted list, every combination (a, b)
			# automatically guarantees a < b. No need for min()/max() later.
			sorted_idx_list = sorted(list(section_idx_set))

			# Skip tiles with 0 or 1 items (no pairs to compute)
			if len(sorted_idx_list) < 2:
				for i in sorted_idx_list: self.distance[(i, i)] = 0.0
				continue

			# 3. OPTIMIZATION: Vectorized Calculation
			# Extract points for this tile into a NumPy array
			# Shape: (N_items, 2)
			points = np.array([coord_lookup[i] for i in sorted_idx_list])

			# Calculate Euclidean distance for all pairs instantly (C-speed)
			# pdist returns a flat array of distances: [dist(0,1), dist(0,2), ... dist(1,2) ...]
			dists = pdist(points, metric = 'euclidean')

			# 4. OPTIMIZATION: Bulk Dictionary Update
			# Generate keys corresponding to pdist order
			pair_keys = combinations(sorted_idx_list, 2)

			# Zip them together and update the dict in one go
			# Note: We overwrite duplicates (overlaps) which is safe and faster than checking 'if in'
			self.distance.update(zip(pair_keys, dists))

			# Set diagonals (Distance to self is 0)
			for i in sorted_idx_list:
				self.distance[(i, i)] = 0.0

		self.logger.info("✅  Pairwise Distances Computed.")

	@not_implemented
	def _process_distances_sparse_matrix(self):
		### TO DEBUG
		self.logger.info("⚙️  Precomputing Pairwise Distances (Sparse Matrix)...")

		# 1. MAPPING: Real ID -> Matrix Index
		# We only need one way mapping (Real ID -> 0..N)
		unique_ids = sorted(self.sections.keys())
		id_to_matrix_idx = {real_id: i for i, real_id in enumerate(unique_ids)}
		num_items = len(unique_ids)

		# 2. DATA CONTAINERS
		# We will accumulate data in lists and convert once at the end
		rows = []
		cols = []
		data = []

		# Pre-fetch coordinates: {Matrix_Index: (x, y)}
		# Only store coordinates for IDs that actually exist in the mapping
		coord_lookup = {
			id_to_matrix_idx[idx]: (poly.centroid.x, poly.centroid.y)
			for idx, poly in self.sections.items()
			if idx in id_to_matrix_idx
		}

		# Loop using '_' for the unused tile key
		for _, section_idx_set in self._tile_idx_to_idx.items():
			# Convert to matrix indices
			# Filter valid IDs and Sort (Sorting is crucial for pdist alignment)
			matrix_indices = sorted(
				[
					id_to_matrix_idx[uid] for uid in section_idx_set
					if uid in id_to_matrix_idx
				],
			)

			n = len(matrix_indices)
			if n < 2:
				continue

			# A. Vectorized Distance Calculation (Float Array)
			points = np.array([coord_lookup[i] for i in matrix_indices])
			# pdist returns distances in the order of "upper triangle" of the matrix
			dists = pdist(points, metric = 'euclidean')

			# B. Generate Indices directly (Integer Arrays)
			# Instead of creating a list of tuples with combinations(),
			# we generate the row/col arrays directly using numpy.
			# triu_indices_from generates indices for the upper triangle: (0,1), (0,2)...
			# We assume matrix_indices corresponds to a sub-matrix of size n*n
			r_local, c_local = np.triu_indices(n, k = 1)

			# Map local indices (0..n) back to the actual matrix indices
			# We use numpy indexing to "broadcast" the real indices
			# matrix_indices is a list like [105, 999], r_local is like [0]
			# So rows_mapped becomes [105]
			matrix_indices_arr = np.array(matrix_indices)
			rows.extend(matrix_indices_arr[r_local])
			cols.extend(matrix_indices_arr[c_local])
			data.extend(dists)

		# 3. CONSTRUCT MATRIX
		# Create CSR matrix from the accumulated buffers
		self.distance_matrix = csr_matrix(
			(data, (rows, cols)),
			shape = (num_items, num_items),
			dtype = np.float32,
		)

		# Store the mapper so we can look things up later
		self.id_mapper = id_to_matrix_idx

		self.logger.info(f"✅ Computed. Matrix uses {self.distance_matrix.data.nbytes / 1e6:.2f} MB")

	# ----------- (POSSIBLE UPDATE) Tile-Local Distance Matrices  --------------
	# for faster GPU performance during training and testing.

	@not_implemented
	def _process_distances_tile_local(self):
		"""
		Computes Tile-Local Dense Matrices with Global Normalization.
		Replaces the BBHash method for O(1) access and better CPU performance.
		"""
		self.logger.info("⚙️  Precomputing Tile-Local Dense Matrices...")

		# 1. Coordinate Lookup
		coord_lookup = {
			idx: (poly.centroid.x, poly.centroid.y)
			for idx, poly in self.sections.items()
		}

		# Temp storage for Pass 1
		raw_tile_dists: Dict[int, np.ndarray] = {}
		global_min = float('inf')
		global_max = float('-inf')

		# --- PASS 1: Calculate Raw Distances & Find Global Min/Max ---
		for tile_idx, section_idx_set in tqdm.tqdm(
				self._tile_idx_to_idx.items(),
				desc = "Calc Raw Distances",
				total = len(self._tile_idx_to_idx)
		):
			sorted_idx = self._tile_idx_to_idx[tile_idx]
			if len(sorted_idx) < 2: continue

			# Create Map: Global ID -> Local Matrix Row Index
			for local_row, global_id in enumerate(sorted_idx):
				self._global_to_local_idx[global_id] = local_row

			points = np.array([coord_lookup[i] for i in sorted_idx])

			# Calculate 1D compressed distance array
			dists = pdist(points, metric = 'euclidean')

			if dists.size > 0:
				current_min = dists.min()
				current_max = dists.max()
				if current_min < global_min: global_min = current_min
				if current_max > global_max: global_max = current_max

			raw_tile_dists[tile_idx] = dists

		self._d_min = global_min
		self._d_max = global_max

		denom = self._d_max - self._d_min
		if denom == 0: denom = 1.0

		# --- PASS 2: Normalize & Store as Dense Matrices ---
		for tile_idx, raw_dists in tqdm.tqdm(
				raw_tile_dists.items(), desc = "Normalize & Store", total = len(raw_tile_dists)
		):
			# Normalize in-place
			raw_dists -= self._d_min
			raw_dists /= denom

			# Convert to Square Matrix (N x N)
			dense_matrix = squareform(raw_dists)

			# Store as float16 to save RAM (casted to float32 in __getitem__)
			self.tile_distance_matrices[tile_idx] = torch.tensor(
				dense_matrix, dtype = torch.float16
			)

		del raw_tile_dists
		self.logger.info("✅  Tile-Local Matrices Computed.")


# --- TEST ---
if __name__ == "__main__":
	import random
	import math
	import os


	def test_distance(data: MobilityDataManager, num_tests: int = 5, seed: int = 42, tolerance: float = 10.0):
		"""
		Randomly verifies that the stored hash-based distances match the actual Euclidean
		distances calculated from section centroids.

		Args:
			data: The MobilityDataManager instance.
			num_tests: Number of random pairs to check.
			seed: Random seed for reproducibility.
			tolerance: Absolute tolerance for floating point comparison (e.g., 10 meters).

		Raises:
			AssertionError: If any of the tested distances do not match the ground truth.
		"""
		data.logger.info(f"🧪  Starting Distance Verification (n={num_tests}, seed={seed}) ")

		random.seed(seed)
		failures = []

		# 1. OPTIMIZATION: Filter valid tiles first.
		# We only want tiles that actually have at least 2 sections to form a pair.
		# This avoids "wasting" a test iteration on an empty tile.
		valid_tiles = [
			tid for tid, sections in data._tile_idx_to_idx.items()
			if len(sections) >= 2
		]

		if not valid_tiles:
			data.logger.warning("No tiles with >= 2 sections found. Skipping test.")
			return

		# Select random tiles
		selected_tiles = random.choices(valid_tiles, k = num_tests)

		for i, tile_id in enumerate(selected_tiles):
			sections_in_tile = list(data._tile_idx_to_idx[tile_id])

			# 2. Sample random pair
			u, v = random.sample(sections_in_tile, 2)

			# 3. Calculate Ground Truth (using Shapely attributes)
			p_u = data.sections[u].centroid
			p_v = data.sections[v].centroid
			ground_truth = math.hypot(p_u.x - p_v.x, p_u.y - p_v.y)

			# 4. Retrieve and Renormalize Stored Distance
			try:
				raw_dist = data.get_distance_bbhash(u, v)
				stored_dist = data.renormalize_distance(raw_dist)
			except Exception as e:
				msg = f"Lookup failed for pair ({u}, {v}): {str(e)}"
				data.logger.error(msg)
				failures.append(msg)
				continue

			# 5. Verification
			if not np.isclose(ground_truth, stored_dist, atol = tolerance):
				error_msg = (
					f"Mismatch Test #{i + 1} [Tile {tile_id}]: "
					f"Pair ({u}, {v}) | "
					f"Truth: {ground_truth:.4f} != Stored: {stored_dist:.4f} "
					f"(Diff: {abs(ground_truth - stored_dist):.4f})"
				)
				data.logger.error(error_msg)
				failures.append(error_msg)
			else:
				data.logger.debug(f"Test #{i + 1} Passed.")

		# 6. Final Assertion
		# If failures list is not empty, crash the test so the suite knows something is wrong.
		if failures:
			error_report = "\n".join(failures)
			data.logger.debug(f"❌  Distance verification failed on {len(failures)}/{num_tests} tests:\n{error_report}")
		else:
			data.logger.debug("✅  Success: All distance lookups matched ground truth.")


	def test_probability_distribution(
		data: MobilityDataManager, num_tests: int = 5, seed: int = 42, tolerance: float = 1e-4,
	):
		"""
		Randomly verifies that the probability distributions for selected origins sum to 1.

		Args:
			data: The MobilityDataManager instance.
			num_tests: Number of random origins to check.
			seed: Random seed for reproducibility.
			tolerance: Absolute tolerance for floating point comparison.
		"""

		data.logger.debug("🧪  Starting Probability Distribution Verification")

		random.seed(seed)
		failures = []

		for i in range(num_tests):
			# 1. Randomly select a valid origin
			origin_idx = random.choice(data._valid_idx)

			tile_idx = data._idx_to_tile_idx[origin_idx]
			destination_idx_set = data._tile_idx_to_idx[tile_idx]

			keys = [(origin_idx, dest_idx) for dest_idx in destination_idx_set]
			# 2. Retrieve all probabilities for this origin
			total_prob = sum(data.probability.get(key, 0.0) for key in keys)

			# 3. Verification
			if not np.isclose(total_prob, 1.0, atol = tolerance):
				error_msg = (
					f"Mismatch Test #{i + 1}: "
					f"Origin {origin_idx} | "
					f"Total Probability: {total_prob:.6f} (Expected: 1.0)"
				)
				data.logger.error(error_msg)
				failures.append(error_msg)
			else:
				data.logger.debug(f"Test #{i + 1} Passed.")

		# 4. Final Assertion
		if failures:
			error_report = "\n".join(failures)
			data.logger.debug(
				f"❌  Probability distribution verification failed on {len(failures)}/{num_tests} tests:\n{error_report}"
			)
		else:
			data.logger.debug("✅  Probability distribution verification passed.")


	def check_size(data: MobilityDataManager):
		"""Check the size of the MPH and distance values in MB."""
		data.logger.info("🧪  Checking MPH and Distance Values Size...")

		# Save the hash function to a temp file
		data._mph.save("../data/temporary/temp_mph.bin")
		# Check size in MB
		size_mb = os.path.getsize("../data/temporary/temp_mph.bin") / (1024 * 1024)
		data.logger.info(f"MPH Size: {size_mb:.2f} MB")
		# check size of distance values
		size_mb = data.distance.nbytes / (1024 * 1024)
		data.logger.info(f"Distance Values Size: {size_mb:.2f} MB")


	# Initialize
	try:
		dm = MobilityDataManager()

		# Basic Validation
		dm.logger.info("🚀  MobilityDataManager Initialized.")
		dm.logger.info("🧪  Running Basic Validations...")
		dm.logger.info(f"\n--- Dataset Summary --- \n")
		dm.logger.info(f"Total Samples: {len(dm)}")
		dm.logger.info(f"Feature Dimension: {dm.features.shape[1]}")

		check_size(dm)
		test_distance(dm, num_tests = 100, seed = 42)
		test_probability_distribution(
			dm,
			num_tests = 5,
			seed = 42,
			tolerance = 1e-4,
		)

	except FileNotFoundError as e:
		print(f"\n❌ Error: {e}")
		print("Please check your file paths in DataConfig.")
