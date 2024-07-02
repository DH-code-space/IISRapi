import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

def create_mini_batch(samples):
    """Creates mini-batches for training from the provided samples.

    Pads the input sequences to the same length, creates segment tensors, 
    and generates attention masks for the input sequences.

    Args:
        samples (list of tuples): Each tuple contains token tensors, segment tensors,
            and optionally label tensors.

    Returns:
        tuple: A tuple containing padded token tensors, segment tensors, 
            attention masks, and label tensors.
    """
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    mask_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    mask_tensors = mask_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, mask_tensors, label_ids

class MaxPooling(nn.Module):
    """A custom MaxPooling layer for processing model outputs.

    This class inherits from torch.nn.Module and implements a custom
    max pooling operation.

    Methods:
        forward(last_hidden_state, attention_mask): Applies max pooling 
        to the input tensor based on the attention mask.
    """

    def __init__(self):
        """Initializes the MaxPooling layer."""
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        """Applies max pooling to the input tensor.

        Args:
            last_hidden_state: Tensor of hidden states from the model.
            attention_mask: Tensor of attention masks.

        Returns:
            max_embeddings: Tensor after applying max pooling.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)

        return max_embeddings
    
def get_recall_precision(label, pred):
    """Calculates recall and precision from true labels and predictions.

    Args:
        label (list of int): List of true labels.
        pred (list of int): List of predicted labels.

    Returns:
        tuple: A tuple containing recall and precision values.
    """
    error_0 = 0
    error_1 = 0
    yes_0 = 0
    yes_1 = 0

    for l, p in zip(label, pred):
        if l != p:
            if l == 0:
                error_0 += 1
            elif l == 1:
                error_1 += 1
        elif l == p:
            if l == 0:
                yes_0 += 1
            elif l == 1:
                yes_1 += 1

    recall = yes_1 / (yes_1 + error_1)
    precision = yes_1 / (yes_1 + error_0)

    return recall, precision