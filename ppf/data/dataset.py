import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from ppf.data.pii_detector import PIIDetector


def _spans_to_token_labels(
    spans: list,
    offset_mapping: list[tuple[int, int]],
) -> list[int]:
    """
    Map presidio character-level PII spans to token-level binary labels.

    For each token, if its character range overlaps with ANY PII span → label 1.
    Special tokens (offset (0,0)) and padding → label 0 (masked out in loss anyway).
    """
    labels = []
    for start, end in offset_mapping:
        if start == 0 and end == 0:
            # Special token ([CLS], [SEP], [PAD])
            labels.append(0)
            continue
        is_pii = any(
            span.start < end and span.end > start
            for span in spans
        )
        labels.append(1 if is_pii else 0)
    return labels


class TextWithPII(Dataset):
    """
    Text classification dataset with token-level PII labels.

    Each sample returns:
        input_ids:        (seq_len,)   tokenized text
        attention_mask:   (seq_len,)   1 = real token, 0 = padding
        task_label:       int          class label (e.g. sentiment)
        pii_token_labels: (seq_len,)   1 = PII token, 0 = non-PII token
    """

    def __init__(
        self,
        dataset_name: str = "sst2",
        split: str = "train",
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        max_samples: int | None = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        if dataset_name == "sst2":
            ds = load_dataset("stanfordnlp/sst2", split=split)
            text_col, label_col = "sentence", "label"
        elif dataset_name == "ag_news":
            ds = load_dataset("ag_news", split=split)
            text_col, label_col = "text", "label"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        self.texts = list(ds[text_col])
        self.task_labels = list(ds[label_col])

        # Pre-compute PII spans and token-level labels
        print(f"Detecting PII for {len(self.texts)} samples ({split} split)...")
        detector = PIIDetector()

        self.all_spans = []
        pii_sample_count = 0
        total_pii_tokens = 0
        total_real_tokens = 0

        for text in self.texts:
            spans = detector.detect(text)
            self.all_spans.append(spans)
            if spans:
                pii_sample_count += 1

        # Tokenize all texts with offset mapping for span alignment
        self.encodings = self.tokenizer(
            self.texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Build token-level PII labels
        self.pii_token_labels = []
        for i in range(len(self.texts)):
            offsets = self.encodings["offset_mapping"][i].tolist()
            token_labels = _spans_to_token_labels(self.all_spans[i], offsets)
            self.pii_token_labels.append(token_labels)
            mask = self.encodings["attention_mask"][i]
            total_pii_tokens += sum(
                l for l, m in zip(token_labels, mask.tolist()) if m == 1
            )
            total_real_tokens += mask.sum().item()

        pii_pct = 100 * pii_sample_count / len(self.texts)
        tok_pct = 100 * total_pii_tokens / total_real_tokens if total_real_tokens > 0 else 0
        print(f"  Samples with PII: {pii_sample_count}/{len(self.texts)} ({pii_pct:.1f}%)")
        print(f"  PII tokens: {total_pii_tokens}/{total_real_tokens} ({tok_pct:.1f}%)")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "task_label": self.task_labels[idx],
            "pii_token_labels": torch.tensor(self.pii_token_labels[idx], dtype=torch.long),
        }


def _compute_pii_pos_weight(dataset: TextWithPII) -> float:
    """Compute pos_weight = n_non_pii / n_pii for token-level BCE."""
    n_pii = 0
    n_non_pii = 0
    for i in range(len(dataset)):
        mask = dataset.encodings["attention_mask"][i].tolist()
        labels = dataset.pii_token_labels[i]
        for m, l in zip(mask, labels):
            if m == 1:  # real token only
                if l == 1:
                    n_pii += 1
                else:
                    n_non_pii += 1
    if n_pii == 0:
        print("  WARNING: no PII tokens found — pos_weight defaults to 1.0")
        return 1.0
    raw_pw = n_non_pii / n_pii
    pw = min(raw_pw, 20.0)  # cap to prevent probe oscillation
    print(f"  PII pos_weight: {pw:.1f} (raw={raw_pw:.1f}, {n_non_pii} non-PII / {n_pii} PII tokens)")
    return pw


def build_dataloaders(
    dataset_name: str = "sst2",
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, float]:
    """Returns (train_loader, test_loader, pii_pos_weight)."""
    test_split = "validation" if dataset_name == "sst2" else "test"

    train_ds = TextWithPII(dataset_name, "train", tokenizer_name, max_length, max_train_samples)
    test_ds = TextWithPII(dataset_name, test_split, tokenizer_name, max_length, max_test_samples)

    pii_pos_weight = _compute_pii_pos_weight(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, pii_pos_weight
