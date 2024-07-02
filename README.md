# Introduction
This api has 2 features:  
1. Helps you add punctuation and named entity recognition(NER) to Ming-Shilu, it also shows the position where ner or punctuation is used.
2. Compare the articles in Qing-Shilu and Manchu-Old-Archives to see if they describe the same event.
# Requirement
python>=3.9.19
# Installation
```
pip install IISRapi
```
# Usage
### Punctuation and NER
#### Import:
```python
from IISRapi import tool, data
```
#### To use GPU:
You can check the No. of the GPU you want to use by going to task manager
```python
ner_result=tool.IISRner(dev=your_GPU_num)
pun_result=tool.IISRpunctuation(dev=your_GPU_num)
```
#### To use CPU:
Change your_GPU_num to -1

#### Result:
##### method 1:
```python
print(ner_result(your_str))
print(pun_result(your_str))
```
##### method 2:
```python
a=data.struct(your_str)
print(ner_result(a))
print(pun_result(a))
```
Both will print out result in the format struct(Input_string, Result_string, Ner_position, Punctuation_position)
### Paraphrasing
#### Import:
```python
from IISRapi import tool
```
#### To use GPU:
```python
test=tool.eamac(gpu="cuda: your_gpu_num")
result=test(s1=qsl,s2=moa)
```
#### To use CPU:
torch.device is set to cpu by default
#### Result:
```python
print(result)
```
If both articles describe the same event, print out True, otherwise print out False.
# example
### Punctuation and NER
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

### Result:
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
### Paraphrasing
```python
# -*- coding: utf-8 -*-
from IISRapi import tool
qsl="○辛亥。馬哈撒嘛諦汗使臣衛徵喇嘛等謁上陳所貢馬匹、野騾鵰翎弓等物。跪獻其主奏疏大學士希福受之。跪讀於御前。其疏曰。馬哈撒嘛諦塞臣汗謹奏威服諸國皇帝、凡歸順者、與為一體。遣使往來。我等原謂典籍之所首尚。但奉有不合與明國私貿馬匹之諭、我等正欲禁止因見喀爾喀部落七固山、及厄魯特四部落皆往交易。我等效而行之耳。讀畢。使臣行三跪九叩頭禮賜之大宴"
moa="初九日，前往蒙古喀爾喀部馬哈撒嘛諦汗處議和之衛寨桑等，攜馬哈撒嘛諦汗來朝議和進貢則畜使臣衛徵喇嘛、畢車齊吳巴希、哲赫渾津、畢車齊班第、德得依冰圖、烏珠穆沁之納木渾津等六人及商人一百五十六人還。十一日，馬哈撒嘛諦汗使臣衛徵喇嘛等朝見聖汗，陳所貢財物牲畜，衛徵喇嘛捧其汗奏疏率眾跪。蒙古大學士希福受之，跪讀於聖汗前。其疏曰：\"馬哈撒嘛諦色臣汗謹奏威服一切之天聰汗。共持和睦之道，相互遣使往來，乃謂典籍之所首尚。然奉有與明國貿易易不合賣馬之諭，我等正欲禁止貿易，因見喀爾喀部七旗及厄魯特四部落俱往交易，幫我等亦往交七旗及厄魯特四部落俱往交易，幫我等亦往交易。為首使臣以衛徵喇嘛在內共六人。\"讀畢，衛徵喇嘛等行三跪九叩頭禮。大筵宴之。馬哈撒嘛諦汗貢馬三十、野驢一、兒鵰翎四、弓二。為首前來議和之衛徵喇嘛貢馬三、其跟役十人。古穆西班第貢馬一，畢車齊吳巴希貢其跟役二人。畢車齊吳巴希轉獻班迪大喇嘛所貢馬一，哲赫渾津貢其跟役三人，畢車齊班第貢其跟役三人，德得依貢其跟役三人。浩齊特巴琫土謝圖貢馬三，前來議和之納穆寨侍衛巴克什貢其跟役三人。烏珠穆沁之多爾吉車臣濟濃貢馬四，前來議和之納木渾津達爾漢班第貢其跟役四人。碩雷之子色稜諾木奇爾漢班第貢其跟役四人。碩雷之子色稜諾木奇、阿當阿貢馬二、跟役四人。蘇尼特戴青黃臺吉貢馬二、跟役四人。戴青黃臺吉、額爾德尼鄂木布喇嘛貢馬一、跟役二人。衛徵巴圖魯臺吉貢馬跟役二人。囊蘇喇嘛貢馬一、跟役二人。古穆臺吉貢馬一、跟役二人。烏珠穆沁之奇塔特皁鵰翎一、跟役二人。烏珠穆沁之奇塔特哈坦巴圖魯貢馬二、跟役五人。恩克依代巴圖魯貢馬三、跟役二人。奇塔特臺吉貢馬二、跟役二人。浩齊特額爾德　諾木齊奇巴海貢馬一、跟役二人。額爾德尼諾木齊貢馬一、活皁雕一、跟役二人。碩洛依額爾克齋桑貢馬二。烏珠穆沁之塞冷額爾德尼貢馬一、桑貢馬二。烏珠穆沁之塞冷額爾德尼貢馬一、跟役四人。齊巴幹齊喇嘛貢馬二、跟役三人。杜斯噶爾濟濃下達爾漢諾彥貢馬二、跟役三人。碩洛依額爾哲伊圖嘎巴楚喇嘛貢馬一。布雅胡達爾漢諾顏下西達布哈岱貢馬二。班迪大喇嘛貢馬二。濟濃綽爾濟貢馬二。諾木漢喇嘛貢馬三。衛徵班第貢馬二。鬆艾蓋嘎布楚喇嘛貢馬三。達賴綽爾吉喇嘛貢馬二。浩齊特之巴琫土謝圖下託哩喇嘛之薩滿達班第貢馬二。古希貢馬一。額吉根諾彥合貢馬四、貂皮皮端罩一。蘇米爾侍衛臺吉貢馬二。袞楚克貢馬二。恩德恩侍衛臺吉貢馬一。烏珠穆沁之色楞額爾德尼額木齊喇嘛貢馬一。色楞額爾之色楞額爾德尼額木齊喇嘛貢馬一。色楞額爾德尼滿朱習禮喇嘛貢馬二。衛徵班第貢馬二。杜斯噶爾貢馬二。"
test=tool.eamac()
result=test(s1=qsl,s2=moa)
print(result)
```
### Result:
```
True
```
# Notice
1. If you want to use GPU to run modules, visit https://pytorch.org/get-started/locally/ 
2. If you use both functions at the same time, either ner_res(pun_res(your_str)) or pun_res(ner_res(your_str)), the result will only show the result of the function which is done later.
3. Description for models:  
   https://github.com/DH-code-space/EAMAC_paraphrasing (eamac)  
   https://github.com/DH-code-space/Named-entity-Recognition-for-Ming-Shilu (NER)  
   https://github.com/DH-code-space/Automatic-Punctuation-for-Ming-Shilu (punctuation)
