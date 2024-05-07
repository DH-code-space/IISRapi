# Introduction
This package helps you add punctuation and named entity recognition(ner) to Ming-Shilu, it also shows the position where ner or punctuation is used.
# Requirement
```
1. python>=3.8
2. pip>=20.0.2
3. conda>=3.9.0(optional)
```
# Installation
```
pip install IISRapi
```
# Usage
### Import:
```python
from IISRapi.model import tool, data
```
### To use GPU:
You can check the No. of the GPU you want to use by going to task manager
```python
ner_result=tool.IISRner(dev=your_GPU_num)
pun_result=tool.IISRpunctuation(dev=your_GPU_num)
```
### To use CPU:
Change your_GPU_num to -1

### Getting result:
#### method 1:
```python
print(ner_result(your_str))
print(pun_result(your_str))
```
#### method 2:
```python
a=data.struct(your_str)
print(ner_result(a))
print(pun_result(a))
```
Both will print out result in the format struct(Input_string, Result_string, Ner_position, Punctuation_position)

# example
```python
# -*- coding: utf-8 -*-
from IISRapi import tool,data

pun_result=tool.IISRpunctuation(dev=1)
ner_result=tool.IISRner(dev=1)

test_str=data.struct(ori_txt="天啟元年閏二月癸酉朔免肅藩貢馬先是萬歷十四年奉旨令肅府歲進馬百匹 上以額外煩費免之")

for element in pun_result(test_str):
    print(element)
    print('\n')

for element in ner_result(test_str):
    print(element)
    print('\n')
```

### result:
```
天啟元年閏二月癸酉朔免肅藩貢馬先是萬歷十四年奉旨令肅府歲進馬百匹 上以額外煩費免之


天啟元年閏二月癸酉朔，免肅藩貢馬。先是，萬歷十四年奉旨令肅府歲進馬百匹，上以額外煩費，免之。


None


[('，', 9), ('。', 14), ('，', 16), ('，', 31), ('，', 37), ('。', 40)]


天啟元年閏二月癸酉朔免肅藩貢馬先是萬歷十四年奉旨令肅府歲進馬百匹 上以額外煩費免之


天啟元年閏二月癸酉朔免<LOC>肅藩</LOC>貢馬先是萬歷十四年奉旨令<LOC>肅府</LOC>歲進馬百匹 上以額外煩費免之


[(11, 13, 'LOC', '肅藩'), (25, 27, 'LOC', '肅府')]


None
```
# Notice
1. If you want to use GPU to run modules, uninstall torch and visit https://pytorch.org/get-started/locally/  
2. If you use both functions at the same time, either ner_res(pun_res(your_str)) or pun_res(ner_res(your_str)), the result will only show the result of the function which is done later.
