"""
Sliding Window Time Series Dataloader for Stage 1 Pre-training

This module provides a dataloader for continuous time series data using sliding windows.
It is designed to be used alongside the event-based dataloader for the three-stage
training pipeline in GS-FUSE.

Stage 1 uses this sliding window approach to pre-train the decoder on pure time series
forecasting without event-based segmentation.

Usage:
    from sliding_window_dataloader import SlidingWindowDataset, sliding_window_set

    # Create sliding window data loaders
    sw_data = sliding_window_set(
        seq_len=512,
        pred_len=16,
        series_id='SP500',
        series_dir='data/series',
        stride=1,
        batch_size=32
    )

    # Access loaders
    train_loader = sw_data.train_loader
    vali_loader = sw_data.vali_loader
"""

from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class SlidingWindowDataset(Dataset):
    """
    Dataset for sliding window time series forecasting.

    Creates dense, overlapping windows from continuous time series data.
    Each sample consists of:
    - Input: [t : t + seq_len]
    - Target: [t + seq_len : t + seq_len + pred_len]

    This is different from event-based segmentation where samples are
    anchored to specific event dates.
    """

    def __init__(self, data, scaled_data, seq_len, pred_len, indices):
        """
        Args:
            data: Raw numpy array of shape [T, d] (time steps x features)
            scaled_data: Scaled numpy array of shape [T, d]
            seq_len: Length of input sequence (L)
            pred_len: Length of prediction horizon
            indices: List of valid starting indices for windows
        """
        self.data = data
        self.scaled_data = scaled_data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]

        # Extract sequence and prediction windows
        # Shape: [seq_len, d] -> transpose to [d, seq_len] to match event_set format
        seq_data = self.data[start_idx:start_idx + self.seq_len]
        pred_data = self.data[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len]

        scaled_seq_data = self.scaled_data[start_idx:start_idx + self.seq_len]
        scaled_pred_data = self.scaled_data[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len]

        # Return as tensors, transposed to [d, L] format (same as event_set)
        return (
            torch.from_numpy(seq_data.astype(np.float32)).T,
            torch.from_numpy(pred_data.astype(np.float32)).T,
            torch.from_numpy(scaled_seq_data.astype(np.float32)).T,
            torch.from_numpy(scaled_pred_data.astype(np.float32)).T
        )


class sliding_window_set(object):
    """
    Sliding window data manager for Stage 1 time series pre-training.

    Creates train and validation data loaders using sliding windows
    over continuous time series data.

    Attributes:
        train_set: Training dataset
        train_loader: Training data loader
        vali_set: Validation dataset
        vali_loader: Validation data loader
        scaler: Fitted StandardScaler
        d: Number of features/channels
    """

    def __init__(self,
                 seq_len,
                 pred_len,
                 series_id='SP500',
                 series_dir='data/series',
                 stride=1,
                 batch_size=32,
                 scale=True,
                 train_ratio=0.6,
                 vali_ratio=0.2,
                 num_workers=4):
        """
        Args:
            seq_len: Length of input sequence (L)
            pred_len: Length of prediction horizon
            series_id: Identifier for the time series file (e.g., 'SP500')
            series_dir: Directory containing time series CSV files
            stride: Stride for sliding window (default: 1 for maximum overlap)
            batch_size: Batch size for data loaders
            scale: Whether to apply standardization
            train_ratio: Ratio of data for training (default: 0.6)
            vali_ratio: Ratio of data for validation (default: 0.2)
                        (remaining 0.2 is reserved for test, not used in Stage 1)
            num_workers: Number of data loading workers
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

        # Find and load the time series file
        series_file = None
        for file in os.listdir(series_dir):
            if file.split('.')[0] == series_id:
                series_file = os.path.join(series_dir, file)
                break

        if series_file is None:
            raise FileNotFoundError(f"No time series file found for series_id='{series_id}' in {series_dir}")

        print(f"[SlidingWindowSet] Loading time series from: {series_file}")
        raw_df = pd.read_csv(series_file)

        # Store dates for reference
        if 'date_int' in raw_df.columns:
            self.dates = raw_df['date_int'].tolist()
        else:
            self.dates = None

        # Drop date columns
        cols_to_drop = [c for c in ['date', 'date_int'] if c in raw_df.columns]
        self.data = raw_df.drop(columns=cols_to_drop).to_numpy().astype(np.float32)

        self.d = self.data.shape[1]  # Number of features
        total_len = len(self.data)

        print(f"[SlidingWindowSet] Time series shape: {self.data.shape} (T={total_len}, d={self.d})")

        # Calculate split boundaries based on time (not shuffled)
        # This respects temporal ordering: train -> vali -> test
        train_end = int(total_len * train_ratio)
        vali_end = int(total_len * (train_ratio + vali_ratio))

        # Generate valid indices for each split
        # A valid index i means we can extract [i : i+seq_len] and [i+seq_len : i+seq_len+pred_len]
        train_indices = []
        vali_indices = []

        # Training indices: windows that end before train_end
        for i in range(0, train_end - seq_len - pred_len + 1, stride):
            train_indices.append(i)

        # Validation indices: windows that start after train_end and end before vali_end
        for i in range(train_end, vali_end - seq_len - pred_len + 1, stride):
            vali_indices.append(i)

        print(f"[SlidingWindowSet] Train windows: {len(train_indices)}, Vali windows: {len(vali_indices)}")
        print(f"[SlidingWindowSet] Train range: [0, {train_end}), Vali range: [{train_end}, {vali_end})")
        print(f"[SlidingWindowSet] Stride: {stride}")

        # Fit scaler on training data only
        self.scaler = StandardScaler()
        if scale:
            train_data_for_scaling = self.data[:train_end]
            self.scaler.fit(train_data_for_scaling)
            self.scaled_data = self.scaler.transform(self.data)
            print(f"[SlidingWindowSet] Scaler fitted on training data (first {train_end} time steps)")
        else:
            self.scaled_data = self.data

        # Create datasets
        self.train_set = SlidingWindowDataset(
            self.data, self.scaled_data, seq_len, pred_len, train_indices
        )
        self.vali_set = SlidingWindowDataset(
            self.data, self.scaled_data, seq_len, pred_len, vali_indices
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=num_workers,
            pin_memory=True
        )
        self.vali_loader = DataLoader(
            self.vali_set,
            batch_size=batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"[SlidingWindowSet] Data Loaded: TRAIN: {len(self.train_set)}, VALI: {len(self.vali_set)}")
        print(f"[SlidingWindowSet] Batches per epoch: Train={len(self.train_loader)}, Vali={len(self.vali_loader)}")


def get_sliding_window_loaders(seq_len, pred_len, series_id='SP500', series_dir='data/series',
                               stride=1, batch_size=32, scale=True,
                               train_ratio=0.6, vali_ratio=0.2, num_workers=4):
    """
    Convenience function to create sliding window data loaders.

    Returns:
        train_loader, vali_loader, scaler, d (number of features)
    """
    sw_set = sliding_window_set(
        seq_len=seq_len,
        pred_len=pred_len,
        series_id=series_id,
        series_dir=series_dir,
        stride=stride,
        batch_size=batch_size,
        scale=scale,
        train_ratio=train_ratio,
        vali_ratio=vali_ratio,
        num_workers=num_workers
    )
    return sw_set.train_loader, sw_set.vali_loader, sw_set.scaler, sw_set.d


if __name__ == "__main__":
    # Example usage and testing
    print("Testing SlidingWindowDataset...")

    # Create a dummy time series for testing
    import tempfile

    # Create dummy data
    T, d = 1000, 5
    dummy_data = np.random.randn(T, d).astype(np.float32)
    dummy_df = pd.DataFrame(dummy_data, columns=[f'feature_{i}' for i in range(d)])
    dummy_df['date'] = pd.date_range('2020-01-01', periods=T)
    dummy_df['date_int'] = range(T)

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_file = os.path.join(tmpdir, 'TEST.csv')
        dummy_df.to_csv(dummy_file, index=False)

        # Test sliding window set
        sw_set = sliding_window_set(
            seq_len=64,
            pred_len=16,
            series_id='TEST',
            series_dir=tmpdir,
            stride=1,
            batch_size=8
        )

        # Test data loading
        for batch in sw_set.train_loader:
            seq, pred, scaled_seq, scaled_pred = batch
            print(f"Batch shapes: seq={seq.shape}, pred={pred.shape}")
            print(f"  scaled_seq={scaled_seq.shape}, scaled_pred={scaled_pred.shape}")
            break

        print("\nTest passed!")
