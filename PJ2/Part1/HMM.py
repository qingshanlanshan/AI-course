# %% [markdown]
# # Import

# %%
import random
import numpy as np
import os


# %% [markdown]
# # Helper functions

# %%
def log_normalize(array:np.ndarray)->np.ndarray:
    if np.min(array)<1:
        array=array+1
    return np.log(array/np.sum(array))

def normalize(array:np.ndarray)->np.ndarray:
    try:
        return array/np.sum(array)
    except:
        return array


# %% [markdown]
# # HMM

# %%
class HMM:
    def __init__(self) -> None:
        pass
        # if language=="cn":
        #     tags=cn_tags
        #     self.emit_range=65535
        #     self.read_file(cn_train)
        # else:
        #     tags=en_tags
        #     self.read_file(en_train)
        #     word2int_set=set()
        #     for i in range(len(self.lines)):
        #         if len(self.lines[i])==1:
        #             continue
        #         word,_=self.lines[i].split()
        #         word2int_set.update({word.lower()})
        #     word2int_set=list(word2int_set)
        #     self.en2int={}
        #     for i in range(len(word2int_set)):
        #         self.en2int[word2int_set[i]]=i
        #     self.emit_range=len(word2int_set)
        
            
    def set_tags(self,tags:list)->None:
        self.tags={}
        self.reverse_tags={}
        for i in range(len(tags)):
            self.tags[tags[i]]=i
            self.reverse_tags[i]=tags[i]
        
    def read_file(self,filename:str)->None:
        with open(filename, mode='r', encoding='utf-8') as f:
            self.lines = f.readlines()
    
    def word2int(self,word:str)->int:
        # if self.lang=="cn":
        #     return ord(word)
        try:
            return self.en2int_dict[word.lower()]
        except KeyError:
            return -1
    
    def train(self):
        word2int_set=set()
        for i in range(len(self.lines)):
            if not len(self.lines[i].strip()):
                continue
            word,_=self.lines[i].split()
            word2int_set.update({word.lower()})
        word2int_set=list(word2int_set)
        self.en2int_dict={}
        for i in range(len(word2int_set)):
            self.en2int_dict[word2int_set[i]]=i
        self.emit_range=len(word2int_set)
        
        # 初始状态
        self.init_states = np.zeros(len(self.tags))
        # 状态转移概率
        self.states_trans_prob = np.zeros((len(self.tags), len(self.tags)))
        # 输出概率
        self.emit_prob = np.zeros((len(self.tags), self.emit_range))
        
        
        lines=self.lines
        last_tag=None
        for i in range(len(lines)):
            # if end of a sentence
            if len(lines[i].strip())==0:
                last_tag=None
                continue
            # read char and tag
            char, tag = lines[i].split()

            self.emit_prob[self.tags[tag]][self.word2int(char)] += 1
            # if start of a new sentence
            if last_tag==None:
                self.init_states[self.tags[tag]] += 1
                last_tag=tag
                continue
            self.states_trans_prob[self.tags[last_tag]][self.tags[tag]] += 1
            last_tag=tag
            
        # normalize
        self.init_states=normalize(self.init_states)
        self.states_trans_prob=normalize(self.states_trans_prob)
        self.emit_prob=normalize(self.emit_prob)
        
    def get_emit_prob_safe(self,word:str):
        try:
            return self.emit_prob[:,self.word2int(word)]
        except:
            temp=np.mean(self.emit_prob,axis=1)
            return self.emit_prob[:,np.argmax(temp)]
    def viterbi(self,chars:list)->list:
        path=np.zeros(len(chars),dtype=int)
        prob=np.zeros((len(chars),len(self.tags)))
        last_state=np.zeros((len(chars),len(self.tags)))
        # init
        prob[0]=self.init_states*(self.get_emit_prob_safe(chars[0]).T)
        for i in range(1,len(chars)):
            temp=prob[i-1].T.reshape(-1,1)*self.states_trans_prob
            last_state[i]=np.argmax(temp,axis=0)
            prob[i]=np.max(temp,axis=0)*self.get_emit_prob_safe(chars[i])
        path[-1]=np.argmax(prob[-1])
        for i in range(len(chars)-2,-1,-1):
            path[i]=last_state[i+1][path[i+1]]
        return path
            
    def test(self,write:bool=False):
        if write:
            f = open("cn_HMM_my_result.txt", mode='w', encoding='utf-8')
        char_count=0;
        error_count=0
        lines=self.lines
        chars=[]
        tags=[]
        for i in lines:
            if not len(i.strip()):
                # end of a sentence
                result=self.viterbi(chars)
                for i in range(len(chars)):
                    if write:
                        f.write(chars[i]+" "+self.reverse_tags[result[i]]+"\n")
                    if result[i]!=self.tags[tags[i]]:
                        error_count+=1
                char_count+=len(chars)
                chars=[]
                tags=[]
                if write:
                    f.write("\n")
                continue
            char,tag=i.split()
            chars.append(char)
            tags.append(tag)
        try:
            result=self.viterbi(chars)
            for i in range(len(chars)):
                if write:
                    f.write(chars[i]+" "+self.reverse_tags[result[i]]+"\n")
                if result[i]!=self.tags[tags[i]]:
                    error_count+=1
            char_count+=len(chars)
        except:
            pass
        if write:
            f.close()
        print("Accuracy: {}%".format(100-error_count/char_count*100))
            
        
        



# %% [markdown]
# # Pretrain

# %%
cn_tags=['O', 
         'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME', 
         'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT',
         'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU', 
         'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE',
         'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG', 
         'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE',
         'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO', 
         'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC']
en_tags=["O",
         "B-PER","I-PER",
         "B-ORG","I-ORG",
         "B-LOC","I-LOC",
         "B-MISC","I-MISC"]

cn_train="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/Chinese/train.txt"
cn_validation="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/Chinese/validation.txt"
en_train="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/English/train.txt"
en_validation="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/English/validation.txt"

# %% [markdown]
# # Train
if __name__=="__main__":
    # %%
    model=HMM()

    model.set_tags(cn_tags)
    # model.set_tags(en_tags)

    model.read_file(cn_train)
    # model.read_file(en_train)

    model.train()

    # %% [markdown]
    # # Test

    # %%
    print("Training Set ",end='')
    model.test()
    # model.read_file(cn_validation)
    # model.read_file(en_validation)
    model.read_file("chinese_test.txt")
    # model.read_file("english_test.txt")
    print("Test Set ",end='')
    model.test(write=True)


