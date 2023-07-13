import torch
import numpy as np
from torchcrf import CRF
from typing import List,Tuple,Dict

cn_tags=['O', 
         'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME', 
         'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT',
         'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU', 
         'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE',
         'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG', 
         'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE',
         'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO', 
         'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC']
cn_tag2int={'O': 0, 
            'B-NAME': 1, 'M-NAME': 2, 'E-NAME': 3, 'S-NAME': 4, 
            'B-CONT': 5, 'M-CONT': 6, 'E-CONT': 7, 'S-CONT': 8,
            'B-EDU': 9, 'M-EDU': 10, 'E-EDU': 11, 'S-EDU': 12, 
            'B-TITLE': 13, 'M-TITLE': 14, 'E-TITLE': 15, 'S-TITLE': 16, 
            'B-ORG': 17, 'M-ORG': 18, 'E-ORG': 19, 'S-ORG': 20,
            'B-RACE': 21, 'M-RACE': 22, 'E-RACE': 23, 'S-RACE': 24, 
            'B-PRO': 25, 'M-PRO': 26, 'E-PRO': 27, 'S-PRO': 28, 
            'B-LOC': 29, 'M-LOC': 30, 'E-LOC': 31, 'S-LOC': 32}
en_tags=["O",
         "B-PER","I-PER",
         "B-ORG","I-ORG",
         "B-LOC","I-LOC",
         "B-MISC","I-MISC"]
en_tag2int={'O': 0,
            'B-PER': 1, 'I-PER': 2,
            'B-ORG': 3, 'I-ORG': 4, 
            'B-LOC': 5, 'I-LOC': 6, 
            'B-MISC': 7, 'I-MISC': 8}
def tag2int(tag:str,lang:str="en"):
    if lang.lower()=="en":
        return en_tag2int[tag]
    else:
        return cn_tag2int[tag]
    
    
cn_train="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/Chinese/train.txt"
cn_validation="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/Chinese/validation.txt"
en_train="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/English/train.txt"
en_validation="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/English/validation.txt"
DEVICE=torch.device('mps')

# global variables
word2int={}
max_length=115

def load_file(filename:str,lang="en")->List[List[Tuple[str,int]]]:
        with open(filename, mode='r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        sentence_list=[[]]
        for i in range(len(lines)):
            if len(lines[i])==0:
                sentence_list.append([])
            else:
                word,tag=lines[i].split()
                sentence_list[-1].append((word.lower(),tag2int(tag,lang)))
        return sentence_list
def data_to_tensor(sentence_list:List[List[Tuple[str,int]]],lang='en')->List[Tuple[torch.Tensor,torch.Tensor]]:
    tensor_list=[]
    for sentence in sentence_list:
        word_list=[word for word,_ in sentence]
        tag_list=[tag for _,tag in sentence]
        length=len(sentence)
        assert(length<max_length)

        data = word_list + ['padding'] * (max_length - length)
        tags = tag_list + [0] * (max_length - length)
        word_tensor=torch.tensor([get_word_idx(w) for w in data])
        tag_tensor=torch.tensor(tags)
        mask_tensor = torch.tensor([1]*length + [0]*(max_length - length)).bool()
        tensor_list.append((word_tensor,tag_tensor,mask_tensor))
        
    return tensor_list
        

def get_word_idx(word:str)->int:
    if word in word2int:
        return word2int[word]   
    else:
        return word2int['unknown']

def create_word_index(word_list:List[str])->Dict[str,int]:
    word2int={}
    for word in word_list:
        if not word in word2int:
            word2int[word]=len(word2int)
    word2int['padding']=len(word2int)
    word2int["unknown"]=len(word2int)
    return word2int

def neg_log_likelihood(self, sentence_tensor=None, label_tensor=None, mask_tensor=None):
    return -self.crf(emissions=sentence_tensor, tags=label_tensor, mask=mask_tensor)

train_set=load_file(en_train)
word_list=[word for sentence in train_set for word,_ in sentence]
word2int=create_word_index(word_list)
tensor_list=data_to_tensor(train_set)

model=CRF(len(en_tags),True)
# train
model=model.train()
for tensor_tuple in tensor_list:
    word_tensor,tag_tensor,mask_tensor=tensor_tuple
    out=model.decode(word_tensor,mask_tensor)
    loss=neg_log_likelihood(model,word_tensor,tag_tensor,mask_tensor)
    print(out,loss)