import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, num_labels, max_length=128):
        if len(texts) != len(labels):
            raise ValueError(f"Texts length ({len(texts)}) does not match labels length ({len(labels)})")

        texts = [str(text) if not isinstance(text, str) else text for text in texts]

        unique_labels = np.unique(labels)
        self.num_labels = num_labels

        print(f"Unique labels in current batch: {unique_labels}")
        print(f"Expected number of labels: {self.num_labels}")

        if len(unique_labels) > self.num_labels:
            raise ValueError(f"Found {len(unique_labels)} classes, but model expects {self.num_labels} classes")

        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print(f"Label mapping for current batch: {label_mapping}")

        self.labels = torch.tensor([label_mapping[label] for label in labels], dtype=torch.long)

        if self.labels.min() < 0 or self.labels.max() >= self.num_labels:
            raise ValueError(f"Labels must be in range [0, {self.num_labels-1}], "
                           f"got range [{self.labels.min()}, {self.labels.max()}]")

        self.encodings = self._parallel_encode(texts, tokenizer, max_length=max_length)

    def _parallel_encode(self, texts, tokenizer, max_length=128, batch_size=128):
        num_workers = min(4, os.cpu_count() - 1) if os.cpu_count() > 1 else 0

        all_input_ids = []
        all_attention_masks = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                futures.append(executor.submit(
                    self._encode_batch,
                    batch_texts,
                    tokenizer,
                    max_length
                ))

            for future in tqdm(futures, total=len(futures), desc="Encoding texts"):
                batch_input_ids, batch_attention_masks = future.result()
                all_input_ids.extend(batch_input_ids)
                all_attention_masks.extend(batch_attention_masks)

        return {
            'input_ids': torch.stack(all_input_ids),
            'attention_mask': torch.stack(all_attention_masks)
        }

    def _encode_batch(self, texts, tokenizer, max_length):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return encodings['input_ids'], encodings['attention_mask']

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)


class CachedDataset:
    def __init__(self):
        self.cache = {}

    def get_or_create(self, texts, labels, tokenizer, key='test', num_labels=None, max_length=128):
        cache_key = f"{key}_{len(texts)}"

        if cache_key not in self.cache:
            if num_labels is None:
                num_labels = len(np.unique(labels))
                print(f"Automatically determined number of labels for {key} set: {num_labels}")

            print(f"Creating new dataset for {key} split with {len(texts)} samples")
            self.cache[cache_key] = TextDataset(texts, labels, tokenizer, num_labels, max_length=max_length)

        return self.cache[cache_key]

    def clear(self):
        self.cache.clear()


def validate_dataset(dataset, max_length=128):
    try:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        sample_size = min(100, len(dataset))
        sample_indices = np.random.choice(len(dataset), sample_size, replace=False)

        for idx in sample_indices:
            item = dataset[idx]

            required_fields = ['input_ids', 'attention_mask', 'labels']
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing required field '{field}' at index {idx}")

            if not isinstance(item['input_ids'], torch.Tensor):
                raise ValueError(f"input_ids at index {idx} is not a torch.Tensor")
            if not isinstance(item['attention_mask'], torch.Tensor):
                raise ValueError(f"attention_mask at index {idx} is not a torch.Tensor")
            if not isinstance(item['labels'], torch.Tensor):
                raise ValueError(f"labels at index {idx} is not a torch.Tensor")

            if item['input_ids'].shape != item['attention_mask'].shape:
                raise ValueError(f"Shape mismatch at index {idx}")

            if item['labels'].min() < 0:
                raise ValueError(f"Negative label found at index {idx}")

            if item['input_ids'].shape[0] > max_length:
                raise ValueError(f"Input sequence length exceeds max_length at index {idx}")

        logging.info("Dataset validation passed successfully")
        return True

    except Exception as e:
        logging.error(f"Dataset validation failed: {str(e)}")
        raise


class PretrainingDataset(Dataset):
    def __init__(self, features, labels, tokenizer, max_length=128, mlm_probability=0.15):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = str(self.features[idx])

        encoding = self.tokenizer(
            feature,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels, already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
