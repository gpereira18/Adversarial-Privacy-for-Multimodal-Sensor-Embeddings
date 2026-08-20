from pathlib import Path
import scipy
import cv2
import torch
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

SEQ_LEN = 128
NUM_FRAMES = 32
FRAME_SIZE = 112
KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)

class UTDMHADDataset:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.mean = None
        self.std = None
        self.rgb_cache = {}
        self.iner_cache = {}

        for filepath in (self.root_dir / "Inertial").glob("*.mat"):
            activity, subject = self._parse(filepath.name)
            raw = scipy.io.loadmat(filepath)["d_iner"]
            self.iner_cache[filepath] = raw
            self.samples.append({
                "file": filepath,
                "activity": activity,
                "subject": subject,
                "modality": "imu",
            })

        for part in ["RGB-part1", "RGB-part2", "RGB-part3", "RGB-part4"]:
            for filepath in (self.root_dir / part).glob("*.avi"):
                activity, subject = self._parse(filepath.name)
                self.samples.append({"file": filepath, "activity": activity, "subject": subject, "modality": "rgb"})

    def _parse(self, filename):
        parts = filename.split("_")
        return int(parts[0][1:]), int(parts[1][1:])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if sample["modality"] == "imu":
            input_tensor = self._load_inertial(sample)
        else:
            input_tensor = self._load_rgb(sample)
        return {
            "modality": sample["modality"],
            "activity": sample["activity"] - 1,
            "subject": sample["subject"] - 1,
            "input_tensor": input_tensor,
        }

    def _load_inertial(self, sample):
        raw = self.iner_cache[sample["file"]]
        idxs = np.linspace(0, raw.shape[0] - 1, SEQ_LEN).astype(int)
        seq = torch.from_numpy(raw[idxs]).float()
        if self.mean is not None:
            seq = (seq - self.mean) / self.std
        return seq

    def _load_rgb(self, sample):
        if sample["file"] in self.rgb_cache:
            return self.rgb_cache[sample["file"]]
        cap = cv2.VideoCapture(str(sample["file"]))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        idxs = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
        clip = [cv2.cvtColor(cv2.resize(frames[i], (FRAME_SIZE, FRAME_SIZE)), cv2.COLOR_BGR2RGB) for i in idxs]
        clip = torch.from_numpy(np.stack(clip)).float() / 255.0
        clip = clip.permute(0, 3, 1, 2)
        clip = ((clip - KINETICS_MEAN) / KINETICS_STD).contiguous()
        self.rgb_cache[sample["file"]] = clip
        return clip


def mixed_collate(batch):
    return {
        "modality": [b["modality"] for b in batch],
        "activity": torch.tensor([b["activity"] for b in batch]),
        "subject": torch.tensor([b["subject"] for b in batch]),
        "input_tensor": [b["input_tensor"] for b in batch],
    }


def compute_normalization_stats(dataset, train_indices):
    all_rows = []
    seen = set()
    for i in train_indices:
        sample = dataset.samples[i]
        if sample["modality"] != "imu" or sample["file"] in seen:
            continue
        seen.add(sample["file"])
        all_rows.append(dataset.iner_cache[sample["file"]])
    stacked = np.concatenate(all_rows, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-8
    return torch.tensor(mean).float(), torch.tensor(std).float()

def build_utdmhad_dataloaders(root_dir, val_split=0.2, batch_size=32):
    dataset = UTDMHADDataset(root_dir)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=mixed_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=mixed_collate)

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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=mixed_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=mixed_collate)

    return train_loader, val_loader

if __name__ == "__main__":
    print("building dataset...")
    train_loader, val_loader = build_utdmhad_dataloaders("UTD-MHAD")
    print(f"dataset built, train batches: {len(train_loader)}")
    print("fetching batches...")
    for batch in range(10):
        batch = next(iter(train_loader))
        print(batch["input_tensor"][0].shape, batch["modality"][0], batch["activity"][0])


        