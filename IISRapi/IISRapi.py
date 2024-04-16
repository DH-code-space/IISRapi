import os
import torch
from flair.models import SequenceTagger
from flair.data import Sentence
import re
import flair
from typing import NamedTuple,List,Tuple
class Data(NamedTuple):
    ori_txt: str
    ret_txt: str
    ner_tags: List[Tuple[str, int, int]] = None
    punct: List[Tuple[str, int]] = None

class IISRpunctuation:
    def __init__(self,model,dev):
        self.yes_model=True
        self.pos=[]
        self.model_path=model
        if(dev>=0 and torch.cuda.is_available()):
             flair.device = torch.device('cuda:' + str(dev))
        else:
            flair.device = torch.device('cpu')
        if not os.path.exists(self.model_path):
            print("Model file not found. You can download it at https://drive.google.com/file/d/1XaLt9skosuU-3VOqMCFyub6drf1SiQyz/view")
            self.yes_model=False
        elif self.model_path=="best-model-ner.pt":
            print("You loaded wrong model, changing to the punctuation model...")
            self.model_path="best-model-pun.pt"
            self.model=self.load_model()
            
        else:
            print("Model found")
            
        self.model = self.load_model()

    def load_model(self):
        return SequenceTagger.load(self.model_path)
        
    def __call__(self,text):
        if(self.yes_model):
            temp=text.split('\n')
            return self.tokenize(temp)
        else:
            return self.no_model()
    
    def no_model():
        return "You don't have the model"
    
    def tokenize(self,sentences):
        WINDOW_SIZE = 256
        tokenized_sentences=[]
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
                            self.pos.append((token.get_label("ner").value,curr_pos))
                            if not end_flag:
                                curr_seg = curr_pos + 1
                    curr_pos += 1
                if end_flag and curr_seg != len(paragraph):
                    curr_seg = len(paragraph)
                    with_punctuation.append('\u3002')
                    self.pos.append(('\u3002',curr_pos))
                while curr_pos > curr_seg:
                    with_punctuation.pop()
                    curr_pos -= 1
            tokenized_sentences.append(''.join(with_punctuation))
            tokenized_string=''.join(tokenized_sentences)
        return tokenized_string
    
class IISRner:
    def __init__(self,model,dev):
        self.pos=[]
        self.yes_model=True
        self.model_path=model
        if(dev>=0 and torch.cuda.is_available()):
             flair.device = torch.device('cuda:' + str(dev))
        else:
            flair.device = torch.device('cpu')
        if not os.path.exists(self.model_path):
            print("Model file not found. You can download it at https://drive.google.com/file/d/1XaLt9skosuU-3VOqMCFyub6drf1SiQyz/view")
            self.yes_model=False
            
        elif self.model_path=="best-model-pun.pt":
            print("You loaded wrong model, changing to the ner model...")
            self.model_path="best-model-ner.pt"
            self.model=self.load_model()
            
        else:
            print("Model found")
        self.model = self.load_model()
        
    def load_model(self):
        return SequenceTagger.load(self.model_path)
        
    def __call__(self,texts):
        if(self.yes_model):
            return self.ner(texts)
        else:
            return self.no_model()
        
    def no_model():
        return "You don't have the model"
            
    def ner(self,text):
        seg = text.strip().replace(' ', '　')  # replace whitespace with special symbol
        sent = Sentence(' '.join([i for i in seg.strip()]), use_tokenizer=False)
        self.model.predict(sent)
        temp = []
        for ne in sent.get_labels():
            se = re.search("(?P<s>[0-9]+):(?P<e>[0-9]+)", str(ne))
            la = re.search("(?P<l> ? [A-Z]+)", str(ne))
            start = int(se.group("s"))
            end = int(se.group("e"))
            label = la.group("l")
            texttemp=text[start:end]
            temp.append((start, end, label.strip(),texttemp))
        temp.reverse()
        self.pos=temp
        temp.sort(key=lambda a: a[0], reverse=True)
        for start, end, label, texttemp in temp:
            if len(text[start:end].replace('　', ' ').strip()) != 0:
                text = text[:start] + "<" + label + ">" + text[start:end] + "</" + label + ">" + text[end:]
        result=self.post_processing(text)
        self.pos.reverse()
        return result.strip().replace('　', ' ')
    
    def post_processing(self,word):
        whole = word.split('\n')
        for line in whole:
            for match in reversed(list(re.finditer("<LOC>(.?)</LOC><WEI>(.?)</WEI>", line))):
                start, end = match.start(), match.end()
                line = line[:start] + line[start:end].replace("</LOC><WEI>", "").replace("</WEI>", "</LOC>") + line[end:]
            for match in reversed(list(re.finditer("<WEI>(.?)</WEI><LOC>(.?)</LOC>", line))):
                start, end = match.start(), match.end()
                line = line[:start] + line[start:end].replace("</WEI><LOC>", "").replace("<WEI>", "<LOC>") + line[end:]
            for match in reversed(list(re.finditer("<ORG>(.?)</ORG><(LOC|WEI)>(.?)</(LOC|WEI)><ORG>", line))):
                start, end = match.start(), match.end()
                line = line[:start] + "<ORG>" + re.sub("<[A-Z/]+>", "", line[start:end]) + line[end:]
            for match in reversed(list(re.finditer("<(LOC|WEI|ORG)>(.?)</(LOC|WEI|ORG)><", line))):
                start, end = match.start(), match.end()
                line = line[:start] + line[end - 1:end + 4] + re.sub("<[A-Z/]+>", "", line[start:end - 1]) + line[end + 4:]
            for match in re.finditer("王</PER>", line):
                start, end = match.start(), match.end()
                while line[start] != "<":
                    start -= 1
                line = line[:start] + "<OFF>" + re.sub("<[A-Z/]+>", "", line[start:end]) + "</OFF>" + line[end:]
            for match in re.finditer("[王侯公伯]</(LOC|WEI|ORG)>", line):
                start, end = match.start(), match.end()
                while line[start] != "<":
                    start -= 1
                line = line[:start] + "<OFF>" + re.sub("<[A-Z/]+>", "", line[start:end]) + "</OFF>" + line[end:]
            for match in re.finditer("[王侯公伯]</(LOC|WEI|ORG)>", line):
                start, end = match.start(), match.end()
                while line[start] != "<":
                    start -= 1
                line = line[:start] + "<OFF>" + re.sub("<[A-Z/]+>", "", line[start:end]) + "</OFF>" + line[end:]
            for match in re.finditer("殿</(WEI|ORG)>", line):
                start, end = match.start(), match.end()
                while line[start] != "<":
                    start -= 1
                line = line[:start] + "<LOC>" + re.sub("<[A-Z/]+>", "", line[start:end]) + "</LOC>" + line[end:]
            for match in reversed(list(re.finditer("<(WEI|ORG)>(等|各)", line))):
                start, end = match.start(), match.end()
                while line[end] != ">":
                    end += 1
                line = line[:start] + re.sub("<[A-Z/]+>", "", line[start:end]) + line[end:]
            for match in re.finditer("司</OFF>", line):
                start, end = match.start(), match.end()
                while line[start] != "<":
                    start -= 1
                line = line[:start] + "<ORG>" + re.sub("<[A-Z/]+>", "", line[start:end]) + "</ORG>" + line[end:]
            line = line.replace("<ORG>司</ORG>", "司")
            return line + '\n'