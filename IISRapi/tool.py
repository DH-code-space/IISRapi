import os
import torch
import re
import flair
import subprocess
import pickle
import io
import warnings
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from flair.models import SequenceTagger
from flair.data import Sentence
from data import struct
from typing import List, Tuple
from transformers import AutoTokenizer
from dataset import myDataset
from utils import create_mini_batch,get_recall_precision
from sklearn.metrics import f1_score
class IISRner:
    """
    This class provides methods for Named Entity Recognition (NER) using the NER model.

    Attributes:
        dev (int): Device ID for GPU usage (if available). Defaults to None (CPU).
        model_path (str): Path for NER model
    Methods:
        __init__(self, dev: int) -> None:
            Initializes the IISRner object with the optional device ID.

        get_path(self) -> str:
            Returns the path to the NER model files.If the model can't be found, 
            then this function will download it automatically.

        load_model(self) -> None:
            Loads the NER model onto the specified device (CPU or GPU).

        __call__(self, texts: Union[str, data.struct]) -> data.truct:
            Processes the input text for NER, handling both strings and TextStruct objects.

        ner(self, text: str) -> Tuple[str, List[Tuple[int, int, str, str]]]:
            Performs Named Entity Recognition on the input text.
            Returns the processed text and a list of where ner are used in the sentence.

        post_processing(self, word: str) -> str:
            Applies post-processing to a word after NER.
    """

    def __init__(self,dev) -> None:
        """
        Init IISRner with dev
        
        Get the model's path and load model and decide using GPU or CPU
        
        Args:
            dev(int):determine which GPU to use or simply use CPU
        """
        self.model_path=self.get_path()
        if(dev>=0 and torch.cuda.is_available()):
            flair.device = torch.device('cuda:' + str(dev))
            print(f"Running model with GPU No.{dev}")
        else:
            flair.device = torch.device('cpu')
            print("Running model with CPU")
       
        self.model = self.load_model()
        
    def get_path(self) -> str:
        """
        Retrieves the path to the model file, handling potential download.

        This method attempts to locate the model file (best-model-pun.pt) within the
        package directory of the `IISRner` module. If the module is not found or the
        model file is missing, it automatically downloads the required model package from a
        specified URL using pip.

        Returns:
            str: The absolute path to the model file.

        Raises:
            ModuleNotFoundError: If the model path not found.
        """
        try:
            import IISRner
            path=os.path.join(os.path.dirname(IISRner.__file__),"best-model-ner.pt")
        except ModuleNotFoundError:
            print("Model file not found. Downloading model...")
            model_url="https://github.com/DH-code-space/IISRapi/releases/tag/IISRmodel/IISRner-1.0-py3-none-any.whl"
            subprocess.call(["pip", "install", model_url])
            import IISRner
            path=os.path.join(os.path.dirname(IISRner.__file__),"best-model-ner.pt")
        return path

    def load_model(self) -> None:
        #load the model from get_path()
        return SequenceTagger.load(self.model_path)
        
    def __call__(self,texts:str) -> None:
        """
        Processes text input for Named Entity Recognition (NER), handling both strings and struct objects.

        Args:
            texts (Union[str, struct]): The input text to be processed.
                - If it's a string:
                    - The function self.ner() is applied to the text, returning the 
                      processed text (ret_txt) and named entity tags (ner_tag).
                    - A struct object is created with the original text (ori_txt), 
                      processed  text (ret_txt) and named entity tags (ner_tags).
                - If it's a struct object:
                    - The original text (ori_txt) from the struct object is used for NER  
                      processing with self.ner().
                    - The struct object is updated with the processed text (ret_txt) and named entity tags 
                      (ner_tags) using the `_replace` method.
                      
        Returns:
            struct: A class containing the original text(texts or texts.ori_txt) processed text (ret_txt) and named entity tags (ner_tags).
        """
        if isinstance(texts,str):
            ret_txt,ner_tags=self.ner(texts)
            result=struct(ori_txt=texts, ret_txt=ret_txt,ner_tags=ner_tags)
            return result
        elif isinstance(texts,struct):
            ret_txt,ner_tags=self.ner(texts.ori_txt)
            result=texts._replace(ori_txt=texts.ori_txt,ret_txt=ret_txt,ner_tags=ner_tags)
            return result
     
    def ner(self,text:str) -> Tuple[str, List[Tuple[int, int, str, str]]]:
        """
        Performs Named Entity Recognition (NER) on the input text, 
        inserting markup tags for entities.

        Args:
            text (str): The input text string.

        Returns:
            Tuple[str, List[Tuple[int, int, str, str]]]:
                - str: The processed text with inserted NER entity tags (e.g.,<LOC>肅藩</LOC>).
                - List[Tuple[int, int, str, str]]: A list of tuples containing NER information:
                    - int: Starting index of the entity in the original text.
                    - int: Ending index (exclusive) of the entity in the original text.
                    - str: NER label (e.g., "PER", "LOC").
                    - str: The original entity text substring.
        """
        pos=[]
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
        pos=temp
        temp.sort(key=lambda a: a[0], reverse=True)
        for start, end, label, texttemp in temp:
            if len(text[start:end].replace('　', ' ').strip()) != 0:
                text = text[:start] + "<" + label + ">" + text[start:end] + "</" + label + ">" + text[end:]
        result=self.post_processing(text)
        pos.reverse()
        return result.strip().replace('　', ' '),pos
    
    def post_processing(self,word:str)->str:
        """
        Applies post-processing rules to the input text after Named Entity Recognition (NER).

        This method addresses specific cases where overlapping or misplaced NER tags need
        correction

        Args:
            word (str): The input text string (potentially containing multiple lines).

        Returns:
            str: The post-processed text with corrected NER tags.
        """
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
        
class IISRpunctuation:
    """
    This class provides methods for puncuation using the puncuation model.

    Attributes:
        dev (int): Device ID for GPU usage (if available). Defaults to None (CPU).
        model_path: Path for puncuation model
    Methods:
        __init__(self, dev: int) -> None:
            Initializes the IISRner object with the optional device ID.

        get_path(self) -> str:
            Returns the path to the NER model files.If the model can't be found, 
            then this function will download it automatically.

        load_model(self) -> None:
            Loads the NER model onto the specified device (CPU or GPU).

        __call__(self, texts: Union[str, data.struct]) -> data.truct:
            Processes the input text for NER, handling both strings and TextStruct objects.

        tokenize(self,sentences) -> Tuple[str, List[Tuple[str, int]]]:
            Adds puncuations on the input text.
            Returns the processed text and a list of where puncuations are put.

         
    """
    def __init__(self,dev) -> None:
        """Init IISRpunctuation with dev
        
        Get the model's path and load model and decide using GPU or CPU
        
        Args:
            dev:determine which GPU to use or simply use CPU
        """
        self.model_path=self.get_path()
        if(dev>=0 and torch.cuda.is_available()):
            flair.device = torch.device('cuda:' + str(dev))
            print(f"Running model with GPU No.{dev}")
        else:
            flair.device = torch.device('cpu')
            print("Running model with CPU")
            
        self.model = self.load_model()
        
    def get_path(self) -> str:
        """
        Retrieves the path to the model file, handling potential download.

        This method attempts to locate the model file (best-model-pun.pt) within the
        package directory of the `IISRpunctuation` module. If the module is not found or the
        model file is missing, it automatically downloads the required model package from a
        specified URL using pip.

        Returns:
            str: The absolute path to the model file.

        Raises:
            ModuleNotFoundError: If the model path not found.
        """
        try:
            import IISRpunctuation
            path=os.path.join(os.path.dirname(IISRpunctuation.__file__),"best-model-pun.pt")
        except ModuleNotFoundError:
            print("Model file not found. Downloading model...")
            model_url="https://github.com/DH-code-space/IISRapi/releases/tag/IISRmodel/IISRpunctuation-1.0-py3-none-any.whl"
            subprocess.call(["pip", "install", model_url])
            import IISRpunctuation
            path=os.path.join(os.path.dirname(IISRpunctuation.__file__),"best-model-pun.pt")
        return path
    
    def load_model(self) -> None:
        #load the model from get_path()
        return SequenceTagger.load(self.model_path)
        
    def __call__(self,text) -> None:
        """
        Processes text input, handling both strings and TextStruct objects.
        
        Args:
            text(Union[str, struct]) : The input text to be processed.
                - If it's a string:
                    - Single-character strings are appended with a period (。) and a     
                      punctuation tag of ('。', 0).
                    - Multi-character strings are tokenized (split into meaningful units)
                      using the self.tokenize(text) method,
                    and the resulting processed text and punctuation tags are returned.
                - If it's a struct object:
                    - Single-character original text (ori_txt) is handled similarly to  
                      single-character strings.
                    - Multi-character original text is tokenized, and the struct object 
                      is updated with the processed text and punctuation tags.
        Returns:
            struct: A class containing the original text(text or text.ori_txt), processed text (ret_txt) and punctuation tags (pun_tags).
        """
        if isinstance(text, str):
            if(len(text)==1):
                ret_txt=text+'。'
                punct=[('。', 0)]
                result=struct(ori_txt=text, ret_txt=ret_txt,pun_tags=punct)
            else:
                ret_txt,punct=self.tokenize(text.split('\n'))
                result=struct(ori_txt=text, ret_txt=ret_txt,pun_tags=punct)
            return result
        
        elif isinstance(text,struct):
            
            if(len(text.ori_txt)==1):
                ret_txt=text.ori_txt+'。'
                punct=[('。', 0)]
                result=text._replace(ori_txt=text.ori_txt,ret_txt=ret_txt,pun_tags=punct)
                
            else:
                ret_txt,punct=self.tokenize(text.ori_txt.split('\n'))
                result=text._replace(ori_txt=text.ori_txt,ret_txt=ret_txt,pun_tags=punct)
            return result

    def tokenize(self,sentences : str) -> Tuple[str, List[Tuple[str, int]]]:
        """
        Inserts punctuation marks into input sentences based on character window size.

        Args:
            sentences (str): The input string containing multiple sentences.
    
        Returns:
            Tuple[str, List[Tuple[str, int]]]:
                - str: The tokenized sentence string with inserted punctuation marks.
                - List[Tuple[str, int]]: A list of inserted punctuation marks and their places in the sentence.
        """
        pos=[]
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
                            pos.append((token.get_label("ner").value,curr_pos))
                            if not end_flag:
                                curr_seg = curr_pos + 1
                    curr_pos += 1
                if end_flag and curr_seg != len(paragraph):
                    curr_seg = len(paragraph)
                    with_punctuation.append('\u3002')
                    pos.append(('\u3002',curr_pos))
                while curr_pos > curr_seg:
                    with_punctuation.pop()
                    curr_pos -= 1
            tokenized_sentences.append(''.join(with_punctuation))
            tokenized_string=''.join(tokenized_sentences)
            return tokenized_string,pos
        
class eamac:
    """
    This class provides methods for paraphrasing using the model.
    
    Methods:
        __init__(self, dev: int) -> None:
            Initializes the eamac object with the optional device ID.

        get_path(self) -> str:
            Returns the path to the NER model files.If the model can't be found, 
            then this function will download it automatically.

        load_model(self) -> None:
            Loads the model onto the specified device (CPU or GPU).

        __call__(self, texts: Union[str, data.struct]) -> data.truct:
            Processes the input text for NER, handling both strings and TextStruct objects.
            
        predict_NotSliced(self, model, dataloader, device) -> list:
            Return a list of prediction result.


    """
    def __init__(self, gpu="cpu", batch_size=4, pretrained_model_name="hsc748NLP/GujiBERT_fan") -> None:      
        """
        Initialize the eamac class with the specified GPU, batch size, and pretrained model name.
        
        Args:
            gpu (str): The device to use, either 'cpu' or 'cuda'.
            batch_size (int): The batch size for data loading.
            pretrained_model_name (str): The name of the pretrained model to use for tokenization.
        """
        warnings.filterwarnings("ignore")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.batch=batch_size
        self.model=self.load_model()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print("device:", self.device)
    
    def get_path(self) -> str:
        """
        Retrieves the path to the model file, handling potential download.

        This method attempts to locate the model file (model.pkl) within the
        package directory of the `eamacmodel` module. If the module is not found or the
        model file is missing, it automatically downloads the required model package from a
        specified URL using pip.

        Returns:
            str: The absolute path to the model file.

        Raises:
            ModuleNotFoundError: If the model path not found.
        """
        try:
            import eamacmodel
            path=os.path.join(os.path.dirname(eamacmodel.__file__),"model.pkl")
        except ModuleNotFoundError:
            print("Model file not found. Downloading model...")
            model_url="https://github.com/DH-code-space/IISRapi/releases/tag/IISRmodel/IISRmodel/eamacmodel-1.0-py3-none-any.whl"
            subprocess.call(["pip", "install", model_url])
            import eamacmodel
            path=os.path.join(os.path.dirname(eamacmodel.__file__),"model.pkl")
        return path
    
    def load_model(self) -> None:
        """
        Load the model from the specified path.
        
        Returns:
            The loaded model.
        """
        with open(self.get_path(), "rb") as file:
            if torch.cuda.is_available()==False:
                return CPU_Unpickler(file).load()
            else:
                return pickle.load(file)
    
    def split_sentence(self,sentence, max_length=254) -> list:
        """
        Split a sentence into smaller parts with a specified maximum length.
        
        Args:
            sentence (str): The sentence to split.
            max_length (int): The maximum length of each split part.
        
        Returns:
            list: A list of split sentences.
        """
        sentences = []
        while len(sentence) > max_length:
            split_point = sentence.rfind(' ', 0, max_length)
            if split_point == -1:
                split_point = max_length
            sentences.append(sentence[:split_point].strip())
            sentence = sentence[split_point:].strip()
        sentences.append(sentence)
        return sentences    
    
    def to_json(self,s1,s2) -> list:
        """
        Convert two sentences into a list of JSON strings with split parts.
        
        Args:
            s1 (str): The first sentence.
            s2 (str): The second sentence.
        
        Returns:
            list: A list of JSON strings.
        """
        paired_list=[]
        group_num=1
        paired_list.append({"label": 0, "s1": s1, "s2": s2, "id": "1-"+str(group_num)})
        group_num = group_num + 1
        if len(paired_list)==1:
            paired_list[0]["id"]="1"
        json_list = [json.dumps(item, ensure_ascii=False) for item in paired_list]
        return json_list
    
    def __call__(self,s1,s2) -> bool:
        """
        Call method to tokenize sentences and make predictions.
        
        Args:
            s1 (str): The first sentence.
            s2 (str): The second sentence.
        
        Returns:
            bool: The prediction results.
        """
        self.input=self.to_json(s1,s2)
        self.testset = myDataset(self.tokenizer,self.input)
        self.testloader = DataLoader(self.testset, shuffle=False, batch_size=self.batch, collate_fn=create_mini_batch)
        return self.predict_NotSliced(model=self.model, dataloader=self.testloader, device=self.device)
   
    def predict_NotSliced(self, model, dataloader, device) -> list:
        """
        Predicts labels for the test set without slicing.

        Args:
            model (nn.Module): The model to use for predictions.
            dataloader (DataLoader): DataLoader for the test data.
            device (torch.device): Device to run the predictions on.

        Returns:
            bool: If 2 articles are paraphrased.
        """
        model.eval()
        pred_list = []
        groundtruth = []
        bar = tqdm(enumerate(dataloader), total=len(dataloader))

        with torch.no_grad():
            for step, data in bar:
                data = [t.to(device) for t in data if t is not None]
                ids, seg, mask = data[:3]
                outputs = model(ids, seg, mask)

                p = torch.nn.functional.softmax(outputs, dim=1)
                posibility, pred = torch.max(p, 1)
                pred = pred.tolist()
                posibility = posibility.tolist()
                pred_list += pred
        #with open('./test.jsonl', "r", encoding="utf-8") as f1:
        #    json_list = list(f1)
        #for json_str in json_list:
        #    result = json.loads(json_str)
        #    groundtruth.append(result["label"])
        #f1 = f1_score(groundtruth, pred_list)
        #recall, precision = get_recall_precision(groundtruth, pred_list)
        #print(f1)
        #print(recall)
        #print(precision)
        return pred
    def predict_Sliced(self, model, dataloader, device, dataset):
        """
        Makes predictions on a dataset using the provided model.

        Sets the model to evaluation mode and processes data from the dataloader 
        to generate predictions.

        Args:
            model: The model to be used for prediction.
            dataloader: DataLoader object that provides batches of data.
            device: Device to run the model on (e.g., 'cpu' or 'cuda').
            dataset: Dataset object containing the id_list and label_list.
            args: Arguments containing paths for input and output folders and filenames.

        Returns:
            None. Outputs predictions to a file and prints evaluation metrics.
        """
        model.eval()
        pred_list = []
        bar = tqdm(enumerate(dataloader), total=len(dataloader))

        with torch.no_grad():
            for step, data in bar:
                data = [t.to(device) for t in data if t is not None]
                ids, seg, mask = data[:3]
                outputs = model(ids, seg, mask)

                p = torch.nn.functional.softmax(outputs, dim=1)
                _, pred = torch.max(p, 1)
                pred = pred.tolist()
                pred_list += pred

        id_list = dataset.id_list
        label_list = dataset.label_list
        groundtruth = []
        model_pred = []
        tempans = 0
        boolcheck = False
        for i in range(len(id_list)):
            if "-" not in id_list[i]:
                groundtruth.append(label_list[i])
            else:
                idx, seg = id_list[i].split("-")
                if seg == "1":
                    groundtruth.append(label_list[i])
        count = 1
        for i in range(len(id_list)):
            if "-" not in id_list[i]:
                if boolcheck == True:
                    if tempans == 1:
                        model_pred.append(1)
                    else:
                        model_pred.append(0)
                    count += 1
                    tempans = 0
                    boolcheck = False
                model_pred.append(pred_list[i])
                count += 1
            else:
                boolcheck = True
                idx, seg = id_list[i].split("-")
                if idx != str(count):
                    if tempans == 1:
                        model_pred.append(1)
                    else:
                        model_pred.append(0)
                    tempans = 0
                    count += 1
                    if pred_list[i] == 1:
                        tempans = 1
                if pred_list[i] == 1:
                    tempans = 1
            if i == len(id_list) - 1 and len(model_pred) == len(groundtruth) - 1:
                model_pred.append(tempans)
        result_list=[]
        index=0
        for json_str in self.input:
            result = json.loads(json_str)
            result["label"] = pred_list[index]
            index=index+1
            result_list.append(result)
            
        outlist = {1: {"s1": set(), "s2": set()}, 0: {"s1": set(), "s2": set()}}

        for entry in result_list:
            label = entry["label"]
            outlist[label]["s1"].add(entry["s1"])
            outlist[label]["s2"].add(entry["s2"])

        final_result = {
            True: {
                "s1": " ".join(outlist[1]["s1"]),
                "s2": " ".join(outlist[1]["s2"])
            },
            False: {
                "s1": " ".join(outlist[0]["s1"]),
                "s2": " ".join(outlist[0]["s2"])
            }
        }
        return final_result,len(groundtruth),len(model_pred),count
    
class CPU_Unpickler(pickle.Unpickler):
    """
    A custom unpickler that ensures tensors are loaded onto the CPU.

    Inherits from `pickle.Unpickler` and overrides the `find_class` method to
    replace the `torch.storage._load_from_bytes` function with a version that
    loads tensors onto the CPU using `torch.load(io.BytesIO(b), map_location='cpu')`.
    """
    def find_class(self, module, name):
        """
        Overrides the default `find_class` method to handle torch.storage._load_from_bytes.

        Args:
            module: The module name of the class being sought.
            name: The name of the class being sought.

        Returns:
            A callable that loads tensors onto the CPU, or None if not found.
        """
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)