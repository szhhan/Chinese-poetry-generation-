#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:04:43 2020

@author: sizhenhan
"""

import keras
import numpy as np
from tensorflow.python.keras.callbacks import LambdaCallback, ModelCheckpoint
from tensorflow.python.keras.models import Input, Model, load_model
from tensorflow.python.keras.layers import LSTM, Dropout, Dense,Embedding,SpatialDropout1D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils.data_utils import Sequence
from pre import preprocess, clean 
import os

class generate_poetry(object):
    def __init__(self,length,number_to_train,epoch,keep_training=False):
        self.len = length 
        self.poems_with_5_words, self.poems_with_7_words, self.num_to_word, self.word_to_num_get, self.words = preprocess('poetry.txt',number_to_train)
        self.poems5 = self.poems_with_5_words.split('|')
        self.poems7 = self.poems_with_7_words.split('|')
        self.poems5 = clean(self.poems5,5)
        self.poems7 = clean(self.poems7,7)
        self.poems_with_5_words = "|".join(self.poems5)
        self.poems_with_7_words = "|".join(self.poems7)
        self.epoch = epoch 
        
        if self.len == 5: 
            if os.path.exists('poetry_models.h5'):
                self.model = self.build_model()
                self.model.load_weights('poetry_models.h5')

            else:
                self.train()
        if self.len == 7:
            if os.path.exists('poetry_models7.h5'):
                self.model = self.build_model_7()
                self.model.load_weights('poetry_models7.h5')
            else:
                self.train()
        if keep_training:
            self.train()
    
    def build_model(self):
        input_ = Input(shape=(6,))
        emb = Embedding(len(self.words)+1, 300, embeddings_initializer = 'uniform')(input_)
        emb = SpatialDropout1D(0.01)(emb)
        lstm_1 = LSTM(128, return_sequences=True)(emb)
        lstm_1 = Dropout(0.01)(lstm_1)
        lstm_2 = LSTM(128)(lstm_1)
        lstm_2 = Dropout(0.01)(lstm_2)
        out = Dense(len(self.words), activation='softmax')(lstm_2)
        model = Model(input_, out)
        opt = Adam(lr=0.002)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model 
    
    def build_model_7(self):
        input_ = Input(shape=(8,))
        emb = Embedding(len(self.words)+1, 300, embeddings_initializer = 'uniform')(input_)
        emb = SpatialDropout1D(0.01)(emb)
        lstm_1 = LSTM(128, return_sequences=True)(emb)
        lstm_1 = Dropout(0.01)(lstm_1)
        lstm_2 = LSTM(128)(lstm_1)
        lstm_2 = Dropout(0.01)(lstm_2)
        out = Dense(len(self.words), activation='softmax')(lstm_2)
        model = Model(input_, out)
        opt = Adam(lr=0.002)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model 
    
    def random_pick(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds_exp = np.power(preds,1./temperature)
        preds = preds_exp / np.sum(preds_exp)
        out = np.random.choice(range(len(preds)),1,p=preds)
        return int(out.squeeze())
    
    def predict_single_char(self,sentence,temperature =1):
        if self.len == 5 and len(sentence) != 6:
            print('Error! Input length should be 6! ')
            return
        if self.len == 7 and len(sentence) != 8:
            print('Error! Input length should be 8! ')
            return
        xs = np.zeros((1, self.len+1))
        for i in range(len(sentence)):
            xs[0, i] =  self.word_to_num_get(sentence[i])
        
        preds = self.model.predict(xs)[0]
    
        chosen_ind = self.random_pick(preds,temperature)
    
        generated = self.num_to_word[chosen_ind]
        
        return generated
    
    def predict_whole_poem(self,text,length,temperature =1):
        if self.len == 5 and len(text)!=6:
            print('Error! length should be equal 6')
            return
        if self.len == 7 and len(text)!=8:
            print('Error! length should be equal 8')
            return
        cur = text
        out = ''
        for i in range(length):
            generated = self.predict_single_char(text,temperature)
            out += generated
            text = text[1:] + generated
        cur += out
        return cur
    
    def predict_with_first_sentence(self,temperature=1):
        if self.len == 5:
            ind = np.random.randint(0, len(self.poems5))
            first_line = self.poems5[ind][:6]
            out = self.predict_whole_poem(first_line,18,temperature)
        else:
            ind = np.random.randint(0, len(self.poems7))
            first_line = self.poems7[ind][:8]
            out = self.predict_whole_poem(first_line,24,temperature)
        return out
    
    def generate_poem(self):
        k = 0
        if self.len == 5:
            while True:
                x = self.poems_with_5_words[k: k + 6]
                y = self.poems_with_5_words[k + 6]

                if '|' in x or '|' in y:
                    k += 1
                    continue

                xs = np.zeros((1, 6))
                ys = np.zeros((1, len(self.words)),dtype=np.bool)
                ys[0, self.word_to_num_get(y)] = 1.0
                for i in range(len(x)):
                    xs[0, i] =  self.word_to_num_get(x[i])

                yield xs, ys
        
                k += 1
                if k + 6 >= len(self.poems5):
                    k = 0
        else:
            while True:
                x = self.poems_with_7_words[k: k + 8]
                y = self.poems_with_7_words[k + 8]

                if '|' in x or '|' in y:
                    k += 1
                    continue

                xs = np.zeros((1, 8))
                ys = np.zeros((1, len(self.words)),dtype=np.bool)
                ys[0, self.word_to_num_get(y)] = 1.0
                for i in range(len(x)):
                    xs[0, i] =  self.word_to_num_get(x[i])

                yield xs, ys
        
                k += 1
                if k + 8 >= len(self.poems7):
                    k = 0
    
    def get_sample(self,epoch,logs):
        if epoch % 200 != 0:
            return
        
        with open('out.txt', 'a', encoding='utf-8') as f:
            f.write('==================epoch {}=====================\n'.format(epoch))
                
        for temp in [0.5, 1.0, 1.5]:
            print("temperature value {}:".format(temp))
            out = self.predict_with_first_sentence(temp)
            print(out)
            
            with open('out.txt', 'a',encoding='utf-8') as f:
                f.write(out+'\n')
    
    def train(self):
        if self.len == 5:
            self.model = self.build_model()
            if os.path.exists('poetry_models.h5'):
                self.model.load_weights('poetry_models.h5')
            self.model.fit_generator(
            generator=self.generate_poem(),
            verbose=True,
            steps_per_epoch=128,
            epochs=self.epoch,
            callbacks=[
                ModelCheckpoint('poetry_models.h5', save_weights_only=True),
                LambdaCallback(on_epoch_end=self.get_sample)
            ]
        )
        else:
            self.model = self.build_model_7()
            if os.path.exists('poetry_models7.h5'):
                self.model.load_weights('poetry_models7.h5')
            self.model.fit_generator(
            generator=self.generate_poem(),
            verbose=True,
            steps_per_epoch=128,
            epochs=self.epoch,
            callbacks=[
                ModelCheckpoint('poetry_models7.h5', save_weights_only=True),
                LambdaCallback(on_epoch_end=self.get_sample)
            ]
        )
    
    def hide_poem(self,text,temperature = 1):
        if len(text)!=4:
            print('error! It must be 4 letters!')
            return
        if self.len == 5:
            ind = np.random.randint(0, len(self.poems5))
            cur = self.poems5[ind][-5:] + text[0]
            generated = text[0]
        
            for i in range(5):
                next_ = self.predict_single_char(cur,temperature)   
                cur = cur[1:] + next_
                generated += next_
        
            for i in range(3):
                generated += text[i+1]
                cur = cur[1:] + text[i+1]
                for i in range(5):
                    next_ = self.predict_single_char(cur,temperature)           
                    cur = cur[1:] + next_
                    generated+= next_
        else:
            ind = np.random.randint(0, len(self.poems7))
            cur = self.poems7[ind][-7:] + text[0]
            generated = text[0]
        
            for i in range(7):
                next_ = self.predict_single_char(cur,temperature)   
                cur = cur[1:] + next_
                generated += next_
        
            for i in range(3):
                generated += text[i+1]
                cur = cur[1:] + text[i+1]
                for i in range(7):
                    next_ = self.predict_single_char(cur,temperature)           
                    cur = cur[1:] + next_
                    generated+= next_
        return generated
    
    def first_word_poem(self,inp,temperature =1):   
        if self.len == 5:
            ind = np.random.randint(0, len(self.poems5))
            cur = self.poems5[ind][-5:] + inp
            generated = inp + self.predict_whole_poem(cur,length=23,temperature=temperature)
            return generated[6:]
        else:
            ind = np.random.randint(0, len(self.poems7))
            cur = self.poems7[ind][-7:] + inp
            generated = inp + self.predict_whole_poem(cur,length=31,temperature=temperature)
            return generated[8:]
    
    def first_sentence_poem(self,text,temperature =1):
        if self.len == 5 and len(text)!=6:
            print('Error! Length should be 6!')
            return
        if self.len == 7 and len(text) != 8:
            print('Error! Length should be 8!')
            return
        if self.len == 5:
            cur = text[-6:]
            generated = cur + self.predict_whole_poem(cur,length=18,temperature=temperature)
            return generated[6:]
        else:
            cur = text[-8:]
            generated = cur + self.predict_whole_poem(cur,length=24,temperature=temperature)
            return generated[8:]