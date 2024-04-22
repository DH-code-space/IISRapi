# Introduction
This package helps you adding punctuation and named entity recognition(ner) to Ming-Shilu, it also shows the position where ner or punctuation is used.
# Installation
```
pip install IISRapi
```
# Usage
Create a folder, put both models and your code in this folder
### Import:
```python
from IISRapi.model import ner, pun, data
```
### To use GPU:
You can check the No. of the GPU you want to use by going to task manager
```python
ner_res=ner.IISRner(dev=your_GPU_num)
pun_res=pun.IISRpunctuation(dev=your_GPU_num)
```
### To use CPU:
Change your_GPU_num to -1

### Getting result:
#### method 1:
```python
print(ner_res(your_str))
print(pun_res(your_str))
```
#### method 2:
```python
a=Data(ori_txt=your_str,ret_txt="")
print(ner_res(a))
print(pun_res(a))
```
Both will print out result in the format Data(Input_string, Result_string, Ner_position, Punctuation_position)
# Notice
If you use both functions at the same time, either ner_res(pun_res(your_str)) or pun_res(ner_res(your_str)), the result will only show the result of the function which is done later.
