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

test_str=data.struct(ori_txt="給宣府等處屯牛時調操軍二千八百人於宣府赤城雲州雕鶚屯田俱言無牛耕種乃以河間保定諸府衛官牛五千餘只給之")

for element in pun_result(test_str):
    print(element)
    print('\n')

for element in ner_result(test_str):
    print(element)
    print('\n')
```

### result:
```
給宣府等處屯牛時調操軍二千八百人於宣府赤城雲州雕鶚屯田俱言無牛耕種乃以河間保定諸府衛官牛五千餘只給之


給宣府等處屯牛。時調操軍二千八百人於宣府、赤城、雲州、雕鶚屯田，俱言無牛耕種，乃以河間、保定諸府衛官牛五千餘只給之。


None


[('。', 6), ('、', 18), ('、', 20), ('、', 22), ('，', 26), ('，', 32), ('、', 36), ('。', 50)]


給宣府等處屯牛時調操軍二千八百人於宣府赤城雲州雕鶚屯田俱言無牛耕種乃以河間保定諸府衛官牛五千餘只給之


給<LOC>宣府</LOC>等處屯牛時調操軍二千八百人於<LOC>宣府</LOC><LOC>赤城</LOC><LOC>雲州</LOC><LOC>雕鶚</LOC>屯田俱言無牛耕種乃以<WEI>河間</WEI><WEI>保定諸府衛</WEI>官牛五千餘只給之


[(1, 3, 'LOC', '宣府'), (17, 19, 'LOC', '宣府'), (19, 21, 'LOC', '赤城'), (21, 23, 'LOC', '雲州'), (23, 25, 'LOC', '雕鶚'), (35, 37, 'WEI', '河間'), (37, 42, 'WEI', '保定諸府衛')]


None
```
# Notice
1. If you want to use GPU to run modules, uninstall torch and visit https://pytorch.org/get-started/locally/  
2. If you use both functions at the same time, either ner_res(pun_res(your_str)) or pun_res(ner_res(your_str)), the result will only show the result of the function which is done later.
