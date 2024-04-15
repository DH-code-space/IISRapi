# Introduction
This package helps you adding punctuation and named entity recogniiton(ner) to Ming-shilu, it also shows the position where ner or punctuation is used.
# Installation
```
pip install IISRapi
```
# Usage
```python
from IISRapi import IISRner, IISRpunctuation
```
### To use GPU:
install pytorch with cuda using pip:  
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
using conda:  
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
You can check the No. of the GPU you want to use by going to task manager
```python
pun=IISRpunctuation("best-model-pun.pt",your_GPU_num)
ner=IISRner("best-model-ner.pt",your_GPU_num)
```
### To use CPU:
Change the second argument to -1 when you initialize the class

### Use the function:
```python
pun(your_str)
ner(your_str)
```

### print result:

#### method 1:
Save the result in Data
```python
from IISRapi import Data
result=Data(ori_txt=test_str,ret_txt=pun_re(your_str),ner_tags=ner.pos,punct=pun.pos)
for element in result:
   print(element)
```
#### method 2:
print the result you want like the example shown below
```python
print(ner(your_str))
print(pun.pos)
```
# Notice
You can combine 2 functions and print the result. But make sure you do punctuation first then ner, otherwise the result will be wrong.
