from torch import nn
from transformers import AutoModel
from .utils import MaxPooling

    
class MyGujiBert(nn.Module):
    """Custom BERT model for specific tasks.

    This class implements a BERT-based model with additional dropout,
    max pooling, and a fully connected layer for classification tasks.

    Attributes:
        mymodel: The BERT model loaded from a pre-trained checkpoint.
        drop: Dropout layer to prevent overfitting.
        pooler: Custom max pooling layer.
        fc: Fully connected layer for outputting logits.

    Methods:
        forward(ids, seg, mask): Defines the forward pass of the model.
    """

    def __init__(self, pretrained_model_name,dropout):
        """Initializes MyGujiBert with the specified arguments.

        Args:
            pretrained_model_name: Name for the pretrained model
            dropout: Dropout probability.
        """
        super(MyGujiBert, self).__init__()

        self.mymodel = AutoModel.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(p=dropout)
        self.pooler = MaxPooling()
        self.fc = nn.Linear(self.mymodel.config.hidden_size, 2)
    def forward(self, ids, seg, mask):
        """Defines the forward pass of the model.

        Args:
            ids: Tensor of input IDs.
            seg: Tensor of segment IDs (token type IDs).
            mask: Tensor of attention masks.

        Returns:
            logits: Tensor of logits representing the model's predictions.
        """
        out = self.mymodel(input_ids=ids, attention_mask=mask, token_type_ids=seg,
                         output_hidden_states=False)

        pooled_output = self.pooler(out.last_hidden_state, mask)
        pooled_output = self.drop(pooled_output)
        logits = self.fc(pooled_output)

        return logits