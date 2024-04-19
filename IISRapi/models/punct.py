import os
import torch
from flair.models import SequenceTagger
from flair.data import Sentence
import flair
from IISRapi.data import Data
from typing import Union
from IISRapi.utils import combine_tags

class IISRpunctuation:
    def __init__(self,model,dev):
        self.model_path=model
        if(dev>=0 and torch.cuda.is_available()):
             flair.device = torch.device('cuda:' + str(dev))
        else:
            flair.device = torch.device('cpu')
        if not os.path.exists(self.model_path):
            raise RuntimeError("Model file not found. You can download it at https://drive.google.com/file/d/1XaLt9skosuU-3VOqMCFyub6drf1SiQyz/view")
        elif self.model_path=="best-model-ner.pt":
            print("You loaded wrong model, changing to the punctuation model...")
            self.model_path="best-model-pun.pt"
            self.model=self.load_model()
            
        else:
            print("Model found")
            
        self.model = self.load_model()

    def load_model(self):
        return SequenceTagger.load(self.model_path)
        
    def __call__(self, input: Union[str, Data]):
        if isinstance(input, str):
            ret_txt, annotations = self.tokenize(input)
            return Data(ori_txt=input, ret_txt=ret_txt, punct=annotations)
        elif isinstance(input, Data):
            _, annotations = self.tokenize(input.ori_txt)
            ret_txt = combine_tags(input.ori_txt, input.ner_tags, annotations)
            return Data(ori_txt=input, ret_txt=ret_txt, ner_tags=input.ner_tags, punct=annotations)
    
    def tokenize(self,sentences):
        WINDOW_SIZE = 256
        tokenized_sentences=[]
        punct_tags=[]
        for text in sentences:
            text = text.strip().replace(' ', '')
            if text == "":
                continue
            with_punctuation = []
            paragraph = list(text)
            curr_seg = 0
            end_flag = False
            while curr_seg < len(paragraph) - 1:
                start = curr_seg
                end = curr_seg + WINDOW_SIZE
                if curr_seg + WINDOW_SIZE > len(paragraph):
                    end = len(paragraph)
                    end_flag = True
                tokens = Sentence(paragraph[start : end], use_tokenizer=False)
                self.model.predict(tokens)
                curr_pos = curr_seg
                for token in tokens:
                    with_punctuation.append(text[curr_pos])
                    if token.get_label("ner").value != 'C':
                        if curr_pos != end - 1:
                            with_punctuation.append(token.get_label("ner").value)
                            punct_tags.append((token.get_label("ner").value,curr_pos))
                            if not end_flag:
                                curr_seg = curr_pos + 1
                    curr_pos += 1
                if end_flag and curr_seg != len(paragraph):
                    curr_seg = len(paragraph)
                    with_punctuation.append('\u3002')
                    punct_tags.append(('\u3002',curr_pos))
                while curr_pos > curr_seg:
                    with_punctuation.pop()
                    curr_pos -= 1
            tokenized_sentences.append(''.join(with_punctuation))
            tokenized_string=''.join(tokenized_sentences)
        return tokenized_string, punct_tags