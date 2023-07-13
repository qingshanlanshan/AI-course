from typing import List,Tuple

import sklearn_crfsuite

def load_file(filename:str)->List[List[Tuple[str,int]]]:
    with open(filename, mode='r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    sentence_list=[[]]
    for i in range(len(lines)):
        if len(lines[i])==0:
            sentence_list.append([])
        else:
            word,tag=lines[i].split()
            sentence_list[-1].append((word,tag))
    return sentence_list

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        # 'word[-3:]': word[-3:],
        # 'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper()
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

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

en_train="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/English/train.txt"
en_validation="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/English/validation.txt"
cn_train="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/Chinese/train.txt"
cn_validation="/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/NER/Chinese/validation.txt"

train_set=load_file(en_train)
X_train = [sent2features(s) for s in train_set]
y_train = [sent2labels(s) for s in train_set]

# test_set=load_file(cn_validation)
# test_set=load_file("chinese_test.txt")
test_set=load_file("english_test.txt")
X_test = [sent2features(s) for s in test_set]
Y_test = [sent2labels(s) for s in test_set]

print(X_train[0][0])

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True)
    
crf.fit(X_train, y_train)

Y_pred = crf.predict(X_test)

# f=open("/Users/jiaruiye/Desktop/FDU/专业课程/必修课程/人工智能/Projects/PJ2/CRF_my_result.txt","w",encoding="utf-8")
with open ("en_CRF_my_result.txt","w",encoding="utf-8") as f:
    for sentence_id in range(len(test_set)):
        for word_id in range(len(test_set[sentence_id])):
            f.write(test_set[sentence_id][word_id][0]+" "+Y_pred[sentence_id][word_id]+"\n")
        if not sentence_id==len(test_set)-1:
            f.write("\n")