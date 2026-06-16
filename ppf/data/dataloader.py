from pathlib import Path
import scipy
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
import numpy as np

class UTDMHADDataset:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.mean = None
        self.std = None

        for filepath in (self.root_dir / "Inertial").glob("*.mat"):
            filename = filepath.name

            parts = filename.split("_")

            activity = int(parts[0][1:])
            subject = int(parts[1][1:])

            modality = parts[3].split(".")[0]
            if modality == "inertial":
                self.samples.append(
                {
                    "file": filepath,
                    "activity": activity,
                    "subject": subject,
                    "modality": "accel",
                })
                self.samples.append(
                {
                    "file": filepath,
                    "activity": activity,
                    "subject": subject,
                    "modality": "gyro",
                })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        W = 100
        sample = self.samples[idx]
        data = scipy.io.loadmat(sample["file"])
        raw = data["d_iner"]
        accel = raw[:, :3]
        gyro = raw[:, 3:]
        accel_window = accel[:W, :]
        gyro_window = gyro[:W, :]
        accel_window_tensor = torch.from_numpy(accel_window).float()
        gyro_window_tensor = torch.from_numpy(gyro_window).float()
        if self.mean is not None:
            accel_window_tensor = (accel_window_tensor - self.mean[:3]) / self.std[:3]
            gyro_window_tensor = (gyro_window_tensor - self.mean[3:]) / self.std[3:]
        if sample["modality"] == "accel":
            input_tensor = accel_window_tensor
        elif sample["modality"] == "gyro":
            input_tensor = gyro_window_tensor
            
        return {
            "modality": sample["modality"],
            "activity": sample["activity"] - 1,
            "subject": sample["subject"] - 1,
            "input_tensor": input_tensor,
        }
    
def compute_normalization_stats(dataset, train_indices):
    all_rows = []
    for i in train_indices:
        sample = dataset.samples[i]
        data = scipy.io.loadmat(sample["file"])
        all_rows.append(data["d_iner"])
    stacked = np.concatenate(all_rows, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-8
    return torch.tensor(mean).float(), torch.tensor(std).float()

def build_utdmhad_dataloaders(root_dir, val_split=0.2, batch_size=32):
    dataset = UTDMHADDataset(root_dir)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def build_loso_dataloaders(root_dir, val_subject=8, batch_size=32):
    train_indices = []
    val_indices = []
    dataset = UTDMHADDataset(root_dir)

    for i, sample in enumerate(dataset.samples):
        if sample["subject"] == val_subject:
            val_indices.append(i)
        else:
            train_indices.append(i)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    mean, std = compute_normalization_stats(dataset, train_indices)
    dataset.mean, dataset.std = mean, std

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    print("building dataset...")
    train_loader, val_loader = build_utdmhad_dataloaders("UTD-MHAD")
    print(f"dataset built, train batches: {len(train_loader)}")
    print("fetching batches...")
    for batch in range(10):
        batch = next(iter(train_loader))
        print(batch["input_tensor"].shape, batch["modality"][0], batch["activity"][0])


        