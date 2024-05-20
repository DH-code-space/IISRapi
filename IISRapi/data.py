from typing import NamedTuple,List,Tuple

class struct(NamedTuple):
    """ 
    This is a class that stores the result of NER or puncuation

    Attributes:
        ori_txt (str): original sentence.
        ret_txt (str): the result of NER or puncuation.Set as None by default.
        ner_tags (List[Tuple[str, int, int]]):
        - List[Tuple[int, int, str, str]]: A list of tuples containing NER information:
                    - int: Starting index of the entity in the original text.
                    - int: Ending index (exclusive) of the entity in the original text.
                    - str: NER label (e.g., "PER", "LOC").
                    - str: The original entity text substring.
        pun_tags (List[Tuple[str, int]]):
        Tuple[str, List[Tuple[str, int]]]:
                - str: The tokenized sentence string with inserted punctuation marks.
                - List[Tuple[str, int]]: A list of inserted punctuation marks 
                  and their places in the sentence.
    """
    ori_txt: str
    ret_txt: str=None
    ner_tags: List[Tuple[str, int, int]] = None
    pun_tags: List[Tuple[str, int]] = None

