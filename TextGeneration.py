def read_file(path):
    with open(path) as f:
        str = f.read()
    return str

import spacy 

nlp = spacy.load('en',disable = ['parser','tagger','ner'])
nlp.max_length = 1198623

def separate_punc(text):
    return [token.text.lower() for token in nlp(text) if token.text not in ' \n,\n\n \'...}|{[]!.?\/-_@^%$#@!@"']

d = read_file('TextGeneration.txt')
tokens = separate_punc(d)

train_len = 26

text_seq = []

for i in range(train_len,len(tokens)):
    seq  = tokens[i-train_len:i]
    text_seq.append(seq)
    
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_seq)

sequences = tokenizer.texts_to_sequences(text_seq)

import numpy as np 

sequences = np.array(sequences)

from keras.utils import to_categorical

X = sequences[:,:-1]
y = sequences[:,-1]
y = to_categorical(y,num_classes = len(tokenizer.word_counts) + 1)
seq_len = X.shape[1]

from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding

def create_model(vocab_size,seq_len):
    model = Sequential()
    model.add(Embedding(vocab_size,seq_len,input_length = seq_len))
    model.add(LSTM(seq_len*2,return_sequences = True))
    model.add(LSTM(seq_len*2))
    model.add(Dense(seq_len*2,activation = 'relu'))
    model.add(Dense(vocab_size,activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    return model
'''
model = create_model(len(tokenizer.word_counts)+1,seq_len)

model.fit(X,y,batch_size=128,epochs = 300 ,verbose = 2)

model.save('model.h5')
'''
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
model = load_model('model.h5')
def generate(model,tokenizer,seq_len,input_text):
    
    output=input_text    
    
    for i in range(0,20):
        encoded = tokenizer.texts_to_sequences([output])[0]
        pad_encoded = pad_sequences([encoded],maxlen = seq_len)
        pred_word_ind = model.predict_classes(pad_encoded,verbose = 0)[0]
        pred_word = tokenizer.index_word[pred_word_ind]
        output +=" "+pred_word       
    
    return output    

print(generate(model,tokenizer,seq_len,'I would play football if I had enough time'))




