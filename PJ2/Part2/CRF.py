import numpy as np
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
    
word2int={}


cn_train="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/Chinese/train.txt"
cn_validation="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/Chinese/validation.txt"
en_train="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/English/train.txt"
en_validation="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/English/validation.txt"
CRF_template_mask=[
                #    [1,0,0,0,0],
                   [0,1,0,0,0],
                   [0,0,1,0,0],
                   [0,0,0,1,0],
                #    [0,0,0,0,1],
                #    [1,1,0,0,0],
                   [0,1,1,0,0],
                #    [0,1,0,1,0],
                   [0,0,1,1,0],
                #    [0,0,0,1,1],
                   ]
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
def create_word2int(sentence_list:List[List[Tuple[str,int]]])->Dict[str,int]:
    word2int={}
    for sentence in sentence_list:
        for word,_ in sentence:
            if word not in word2int:
                word2int[word]=len(word2int)
    return word2int
def get_word2int(word:str)->int:
    if word not in word2int:
        word2int[word]=len(word2int)
    return word2int[word]

class CRF():
    def __init__(self,lang="en") -> None:
        if lang=="en":
            self.num_tags=len(en_tags)
        else:
            self.num_tags=len(cn_tags)
        self.unigram_map=[{} for _ in range(self.num_tags)]
        self.bigram_map=[[{} for _ in range(self.num_tags)] for _ in range(self.num_tags)]
    def get_score_from_feature(self,feature:Tuple[int],cur_tag:int,pre_tag:int)->int:
        score=0
        for mask in CRF_template_mask:
            key=[None if mask[i]==0 else feature[i] for i in range(5)]
            key=tuple(key)
            if key in self.unigram_map[cur_tag]:
                score+=self.unigram_map[cur_tag][key]
            elif not len(self.unigram_map[cur_tag])==0:
                score+=max(self.unigram_map[cur_tag].values())

            if pre_tag and key in self.bigram_map[pre_tag][cur_tag]:
                score+=self.bigram_map[pre_tag][cur_tag][key]
            elif pre_tag and not len(self.bigram_map[pre_tag][cur_tag])==0:
                score+=max(self.bigram_map[pre_tag][cur_tag].values())
        return score
    def update_unigram_feature_from_context(self,context:Tuple[int],cur_tag:int,value:int=0)->None:
        for mask in CRF_template_mask:
            key=tuple([None if mask[i]==0 else context[i] for i in range(5)])
            if key == (None,None,None,None,None):
                continue
            if key not in self.unigram_map[cur_tag]:
                self.unigram_map[cur_tag][key]=0
            self.unigram_map[cur_tag][key]+=value
    def update_bigram_feature_from_context(self,context:Tuple[int],cur_tag:int,pre_tag:int,value:int=0)->None:
        for mask in CRF_template_mask:
            key=tuple([None if mask[i]==0 else context[i] for i in range(5)])
            if key not in self.bigram_map[pre_tag][cur_tag]:
                self.bigram_map[pre_tag][cur_tag][key]=0
            self.bigram_map[pre_tag][cur_tag][key]+=value
    def train_sentence(self,sentence:List[Tuple[str,int]])->None:
        length=len(sentence)
        word_list=[get_word2int(word) for word,_ in sentence]
        tag_list=[tag for _,tag in sentence]
        pred_tag_list=self.predict(word_list)
        for i in range(length):
            if not tag_list[i]==pred_tag_list[i]:
                context=word_list[max(0,i-2):i+3]
                context=tuple([None]*max(0,2-i)+context+[None]*max(0,i+3-length))
                self.update_unigram_feature_from_context(context,tag_list[i],1)
                self.update_unigram_feature_from_context(context,pred_tag_list[i],-1)
                if not i==0 and tag_list[i-1]==pred_tag_list[i-1]:
                    self.update_bigram_feature_from_context(context,tag_list[i],tag_list[i-1],1)
                    self.update_bigram_feature_from_context(context,pred_tag_list[i],pred_tag_list[i-1],-1)
                if not i==length-1 and tag_list[i+1]==pred_tag_list[i+1]:
                        context=word_list[max(0,i-1):i+4]
                        context=tuple([None]*max(0,1-i)+context+[None]*max(0,i+4-length))
                        self.update_bigram_feature_from_context(context,tag_list[i+1],tag_list[i],1)
                        self.update_bigram_feature_from_context(context,pred_tag_list[i+1],pred_tag_list[i],-1)
                    
    def train(self,train_set:List[List[Tuple[str,int]]])->None:
        for i in range(len(train_set)):
            self.train_sentence(train_set[i])
            if i%1000==0:
                print("training: {:2}%".format(i/len(train_set)*100))
            
    def predict(self,word_list:List[int])->List[int]:
        length=len(word_list)
        score_matrix=np.zeros((length,self.num_tags),dtype=int)
        last_tag_matrix=np.zeros((length,self.num_tags),dtype=int)
        for tag_id in range(self.num_tags):
            i=0
            context=word_list[max(0,i-2):i+3]
            context=tuple([None]*max(0,2-i)+context+[None]*max(0,i+3-length))
            score_matrix[0]=self.get_score_from_feature(context,tag_id,None)
        for i in range(1,length):
            context=word_list[max(0,i-2):i+3]
            context=tuple([None]*max(0,2-i)+context+[None]*max(0,i+3-length))
            for tag_id in range(self.num_tags):
                new_score=[self.get_score_from_feature(context,pre_tag_id,tag_id) for pre_tag_id in range(self.num_tags)]
                temp=new_score+score_matrix[i-1]
                last_tag_matrix[i][tag_id]=np.argmax(temp)
                score_matrix[i][tag_id]=np.max(temp)
        path=np.zeros(length,dtype=int)
        path[-1]=np.argmax(score_matrix[-1])
        for i in range(length-1,0,-1):
            path[i-1]=last_tag_matrix[i][path[i]]
        return list(path)
    
    # def log_normalize(self)->None:
    #     for tag_id in range(self.num_tags):
            

    

    
    
            
if __name__=="__main__":
    train_set=load_file(en_train,lang="en")
    word2int=create_word2int(train_set)
    model=CRF("en")
    for epoch in range(10):
        model.train(train_set)
        error_count=0
        word_count=0
        for sentence in train_set:
            word_list=[get_word2int(word) for word,_ in sentence]
            tag_list=[tag for _,tag in sentence]
            pred_tag_list=model.predict(word_list)
            for i in range(len(sentence)):
                if not tag_list[i]==pred_tag_list[i]:
                    error_count+=1
                word_count+=1
        print("Accuracy: {:.2%}".format(1-error_count/word_count))
    test_set=load_file(en_validation,lang="en")
    pred_tags=model.predict([get_word2int(word) for word,_ in test_set[0]])
    print(test_set[0])
    print(pred_tags)
    

    
    