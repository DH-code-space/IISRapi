import torch
import json
from torch.utils.data import Dataset
from typing import Tuple
class myDataset(Dataset):
    """
    Custom Dataset class for loading and processing data for training, validation, and testing.

    This class handles loading data from JSON files, tokenizing text data using a provided tokenizer,
    and preparing it for use with PyTorch DataLoader.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer for processing the text data.
        json_list (list): List of JSON strings representing the data.
        len (int): Number of data points in the dataset.
        max_len (int): Maximum length of tokenized sequences.
        s1_list (list): List of first sentences in the data points.
        s2_list (list): List of second sentences in the data points.
    """

    def __init__(self,tokenizer, input) -> None:
        """
        Initializes the myDataset class with tokenizer and input.

        Args:
            tokenizer (AutoTokenizer): Tokenizer for processing the text data.
            input (str): input sentences
        """
        self.max_len = 512
        self.json_list = input
        self.len = len(self.json_list)
        self.tokenizer = tokenizer
        self.read_data()
        
    def read_data(self) -> None:
        """
        Reads and processes the data from the loaded JSON strings.
        """
        self.s1_list, self.s2_list = [], []
        for json_str in self.json_list:
            result = json.loads(json_str)
            self.s1_list.append(result["s1"])
            self.s2_list.append(result["s2"])

    def __getitem__(self, idx):
        """
        Retrieves a single data point from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the tokenized text tensor, segments tensor, and label tensor.
        """
        texta, textb = self.s1_list[idx], self.s2_list[idx]
        texta = "".join(texta[:250])
        textb = "".join(textb[:250])
        label_tensor = None

        tokensa = self.tokenizer.tokenize(str(texta))
        tokensb = self.tokenizer.tokenize(str(textb))
        word_pieces = ["[CLS]"]
        word_pieces += tokensa + ["[SEP]"]
        lena = len(word_pieces)

        word_pieces += tokensb + ["[SEP]"]
        lenb = len(word_pieces) - lena

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        segments_tensor = torch.tensor([0] * lena + [1] * lenb, dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self) -> int:
        """
        Returns the number of data points in the dataset.

        Returns:
            int: Number of data points in the dataset.
        """
        return self.len


if __name__ == '__main__':
    dataset = myDataset()
    print(dataset)