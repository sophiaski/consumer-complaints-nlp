from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class ComplaintsDataset(Dataset):
    def __init__(
        self, dataframe: pd.DataFrame, max_length: int,
    ):
        self.labels = torch.tensor(dataframe["target"].values)
        self.text = dataframe["sequence"].values
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.text[idx]
        # Encode a batch of sentences with dynamic padding.
        encoded_dict = tokenizer.encode_plus(
            data,  # Sentence to encode
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            padding="max_length",  # Pad to longest in batch.
            truncation=True,  # Truncate sentences to `max_length`.
            max_length=self.max_length,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
        )
        encoded_dict["labels"] = self.labels[idx]
        return encoded_dict
