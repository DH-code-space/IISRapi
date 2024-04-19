from typing import List,Tuple

def combine_tags(original_txt: str, ner_tags: List[Tuple[str, int, int]]=None, punt_tags: List[Tuple[str, int]]=None):
    #ne_tags存NE的標記，例如[('OFF', 1, 3), ......] 意思是ori_txt[1]到ori_txt[2]這兩個字屬於OFF
    #punct存句讀的標記，例如[('，', 2), ....] 意思是ori_txt[2]那個字後面要接逗號
    #ori_txt存使用者最一開始輸入的文字，不論哪個模型的輸出都不會對他做變更
    #ret_txt是把兩者標記套用在ori_txt之後的結果，所以上面例子會變成:
    #"  ori_txt[0]<OFF>ori_txt[1]ori_txt[2]</OFF>，.....   "  這也是應該回傳的值
    #注意: ner_tags或punt_tags可能會是None
    raise NotImplementedError('your turn to write this code.')