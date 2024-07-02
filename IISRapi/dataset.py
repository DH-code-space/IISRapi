import torch
import json
from torch.utils.data import Dataset
'''
class myDataset(Dataset):
    """Custom Dataset class for loading and processing data for training, validation, and testing.

    This class handles loading data from JSON files, tokenizing text data using a provided tokenizer,
    and preparing it for use with PyTorch DataLoader.

    Attributes:
        mode (str): Mode in which the dataset is used ('train', 'dev', 'test').
        tokenizer (AutoTokenizer): Tokenizer for processing the text data.
        args (Namespace): Arguments containing various configurations and file paths.
        train_path (str): Path to the training data file.
        dev_path (str): Path to the validation data file.
        test_path (str): Path to the testing data file.
        json_list (list): List of JSON strings representing the data.
        len (int): Number of data points in the dataset.
        max_len (int): Maximum length of tokenized sequences.
        label_list (list): List of labels for the data points.
        s1_list (list): List of first sentences in the data points.
        s2_list (list): List of second sentences in the data points.
        id_list (list): List of IDs for the data points (used if slicing is enabled).
    """

    def __init__(self, mode, tokenizer, args):
        """Initializes the myDataset class with mode, tokenizer, and arguments.

        Args:
            mode (str): Mode in which the dataset is used ('train', 'dev', 'test').
            tokenizer (AutoTokenizer): Tokenizer for processing the text data.
            args (Namespace): Arguments containing various configurations and file paths.
        """
        if args.task_type == 'train':
            assert mode in ["train", "dev", "test"]
            self.train_path = args.input_folder + args.train_file_name
            self.dev_path = args.input_folder + args.dev_file_name
        else:
            assert mode in ["test"]
        
        self.test_path = args.input_folder + args.test_file_name
        self.mode = mode
        self.max_len = 512

        # Load the appropriate data file based on the mode
        if mode == "train":
            with open(self.train_path, "r", encoding = "utf-8") as f1:
                self.json_list = list(f1)
        elif mode == "dev":
            with open(self.dev_path, "r", encoding = "utf-8") as f1:
                self.json_list = list(f1)       
        elif mode == "test":
            with open(self.test_path, "r", encoding = "utf-8") as f1:
                self.json_list = list(f1)
        self.len = len(self.json_list)
        self.args = args
        self.tokenizer = tokenizer
        self.read_data(self.args)

    def read_data(self, args):
        """Reads and processes the data from the loaded JSON strings.

        Args:
            args (Namespace): Arguments containing various configurations.
        """
        self.label_list, self.s1_list, self.s2_list, self.id_list = [], [], [], []

        for json_str in self.json_list:
            result = json.loads(json_str)
            self.label_list.append(int(result["label"]))
            self.s1_list.append(result["s1"])
            self.s2_list.append(result["s2"])
            if args.slice == "True":
                self.id_list.append(result["id"])

    def __getitem__(self, idx):
        """Retrieves a single data point from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the tokenized text tensor, segments tensor, and label tensor.
        """
        if self.mode == "train" or self.mode == "dev":
            label, texta, textb = self.label_list[idx], self.s1_list[idx], self.s2_list[idx]
            texta = "".join(texta[:250])
            textb = "".join(textb[:250])
            label_tensor = torch.tensor(label)
        else:
            label, texta, textb = self.label_list[idx], self.s1_list[idx], self.s2_list[idx]
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

    def __len__(self):
        """Returns the number of data points in the dataset.

        Returns:
            int: Number of data points in the dataset.
        """
        return self.len
    
'''
import torch
import json
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self,tokenizer, input):
        self.max_len = 512
        self.json_list = input
        self.len = len(self.json_list)
        self.tokenizer = tokenizer
        self.read_data()
        
    def read_data(self):
        self.label_list, self.s1_list, self.s2_list, self.id_list = [], [], [], []
        for json_str in self.json_list:
            result = json.loads(json_str)
            self.label_list.append(int(result["label"]))
            self.s1_list.append(result["s1"])
            self.s2_list.append(result["s2"])

    def __getitem__(self, idx):
        
        label, texta, textb = self.label_list[idx], self.s1_list[idx], self.s2_list[idx]
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

    def __len__(self):
        return self.len


if __name__ == '__main__':
    dataset = myDataset()
    print(dataset)