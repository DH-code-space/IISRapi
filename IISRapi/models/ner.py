import os
import torch
from flair.models import SequenceTagger
from flair.data import Sentence
import re
import flair
from IISRapi.data import Data
from IISRapi.utils import combine_tags
from typing import Union

class IISRner:
    def __init__(self,model,dev):
        self.model_path=model
        if(dev>=0 and torch.cuda.is_available()):
             flair.device = torch.device('cuda:' + str(dev))
        else:
            flair.device = torch.device('cpu')
        if not os.path.exists(self.model_path):
            raise RuntimeError("Model file not found. You can download it at https://drive.google.com/file/d/1XaLt9skosuU-3VOqMCFyub6drf1SiQyz/view")
                        
        elif self.model_path=="best-model-pun.pt":
            print("You loaded wrong model, changing to the ner model...")
            self.model_path="best-model-ner.pt"
            self.model=self.load_model()
            
        else:
            print("Model found")
        self.model = self.load_model()
        
    def load_model(self):
        return SequenceTagger.load(self.model_path)
        
    def __call__(self, input: Union[str, Data]):
        if isinstance(input, str):
            ret_txt, annotations = self.ner(input)
            return Data(ori_txt=input, ret_txt=ret_txt, ner_tags=annotations)
        elif isinstance(input, Data):
            _, annotations = self.ner(input.ori_txt)
            ret_txt = combine_tags(input.ori_txt, annotations, input.punct)
            return Data(ori_txt=input, ret_txt=ret_txt, ner_tags=annotations, punct=input.punct)
            
    def ner(self,text):
        seg = text.strip().replace(' ', '　')  # replace whitespace with special symbol
        sent = Sentence(' '.join([i for i in seg.strip()]), use_tokenizer=False)
        self.model.predict(sent)
        annotations = []
        for ne in sent.get_labels():
            se = re.search("(?P<s>[0-9]+):(?P<e>[0-9]+)", str(ne))
            la = re.search("(?P<l> ? [A-Z]+)", str(ne))
            start = int(se.group("s"))
            end = int(se.group("e"))
            label = la.group("l")
            texttemp=text[start:end]
            annotations.append((start, end, label.strip(),texttemp))
        annotations.reverse()
        annotations.sort(key=lambda a: a[0], reverse=True)
        for start, end, label, texttemp in annotations:
            if len(text[start:end].replace('　', ' ').strip()) != 0:
                text = text[:start] + "<" + label + ">" + text[start:end] + "</" + label + ">" + text[end:]
        result=self.post_processing(text)
        annotations.reverse()
        return result.strip().replace('　', ' '), annotations
    
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