# dataset.py
import os
import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

def resample_contour(contour, num_points):
    # Re-sample the contour to fixed size
    n_points = len(contour)
    if n_points == 0:
        raise ValueError("Contour has zero points.")
    if n_points < num_points:
        # if not enough points, repeat the last one
        padding_needed = num_points - n_points
        repeated_last_point = np.tile(contour[-1], (padding_needed, 1))
        contour = np.vstack([contour, repeated_last_point])
    elif n_points > num_points:
        # 如果点数过多，使用线性插值下采样
        old_indices = np.arange(n_points)
        new_indices = np.linspace(0, n_points - 1, num_points)
        fx = interp1d(old_indices, contour[:, 0], kind='linear')
        fy = interp1d(old_indices, contour[:, 1], kind='linear')
        contour = np.column_stack((fx(new_indices), fy(new_indices)))
    return contour

def normalize_contour(contour):
    # 轮廓中心化和归一化
    center = contour.mean(axis=0)
    contour_centered = contour - center
    max_dist = np.max(np.linalg.norm(contour_centered, axis=1))
    if max_dist == 0:
        max_dist = 1e-8  # Avoid division by zero
    contour_normalized = contour_centered / max_dist
    return contour_normalized

def compute_scalars(contour):
    # 计算轮廓标量特征
    cnt_int = contour.astype(np.int32).reshape(-1, 1, 2)
    area = cv2.contourArea(cnt_int)
    perimeter = cv2.arcLength(cnt_int, closed=True)
    # Diameter based on bounding box diagonal
    min_coords = contour.min(axis=0)
    max_coords = contour.max(axis=0)
    diameter = np.linalg.norm(max_coords - min_coords)
    return np.array([area, perimeter, diameter], dtype=np.float32)

class FiberContourDataset(Dataset):
    def __init__(self, data_dir, num_points=64, split='train', test_size=0.2, random_state=42):
        self.num_points = num_points
        self.split = split
        
        data_file = os.path.join(data_dir, "labeled_contours.pkl")
        with open(data_file, 'rb') as f:
            full_data = pickle.load(f)
        
        # Filter out any un-labeled items (-1) just in case
        filtered_data = [item for item in full_data if item['label'] != -1]
        
        # Split into train/val
        train_data, val_data = train_test_split(
            filtered_data, test_size=test_size, random_state=random_state
        )
        
        if split == 'train':
            self.data = train_data
        else:
            self.data = val_data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        original_contour = item['contour']  # Shape (N, 2)
        label = item['label']               # int (0 or 1)
        
        # 1. Resample
        contour_resampled = resample_contour(original_contour, self.num_points)
        # 2. Normalize
        contour_normalized = normalize_contour(contour_resampled)
        # 3. Compute scalars (ensure it's 3 elements)
        scalars = compute_scalars(contour_resampled)
        
        # Convert to tensors
        points_tensor = torch.from_numpy(contour_normalized).float()  # (N, 2)
        scalars_tensor = torch.from_numpy(scalars).float()           # (3,)
        label_tensor = torch.tensor(label, dtype=torch.long)         # ()
        
        return points_tensor, scalars_tensor, label_tensor

# For testing the dataset class
if __name__ == "__main__":
    dataset_train = FiberContourDataset("data", num_points=64, split='train')
    print(f"Training samples: {len(dataset_train)}")
    if len(dataset_train) > 0:
        pts, scals, lbl = dataset_train[0]
        print(f"Sample Points shape: {pts.shape}")  # Should be [64, 2]
        print(f"Sample Scalars shape: {scals.shape}") # Should be [3]
        print(f"Sample Label: {lbl}")              # Should be 0 or 1