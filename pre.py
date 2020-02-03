#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:53:33 2020

@author: sizhenhan
"""

def preprocess(poets,number_to_train):
    p_list = []
    with open(poets, 'r',encoding='UTF-8') as f:
        for line in f:
            p_list.append(line)
    
    p_list = p_list[:number_to_train]
    poems_with_5_words = ""
    poems_with_7_words = ""
    for poem in p_list:
        x = poem.split(':')[1].strip() + "|"
        if len(x) >5 and x[5] == '，':
            poems_with_5_words += x
        elif len(x) >7 and x[7] == "，":
            poems_with_7_words += x
    all_poems = poems_with_5_words + poems_with_7_words
    
    chars = list(all_poems)
    word_count = {}
    for char in chars:
        word_count[char] = word_count.get(char,0) + 1
    low_frequency_words = []
    for word,freq in word_count.items():
        if freq <= 5:
            low_frequency_words.append(word)
    for word in low_frequency_words:
        del word_count[word]
    words = sorted(word_count.items(), key=lambda x: -x[1])
    
    word_list = sorted(word_count, key=word_count.get, reverse=True)
    word_list.append(" ")
    word_to_num = {}
    num_to_word = {}
    for i, w in enumerate(word_list):
        word_to_num[w] = i
        num_to_word[i] = w
        
    word_to_num_get = lambda x: word_to_num.get(x, len(words) - 1)
    
    return poems_with_5_words, poems_with_7_words, num_to_word, word_to_num_get, words


def clean(poems,length):
    poems_f = []
    if length == 5:
        for poem in poems:
            k = 5
            flag = True
            for i in range(len(poem)):
                if i != k and (poem[i] == '，' or poem[i] == '。'):
                    flag = False
                    break
                if i == k and (poem[i] != '，' and poem[i] != '。'):
                    flag = False
                    break
                if i == k :
                    k += 6
            if flag:
                poems_f.append(poem)
    else:
        for poem in poems:
            k = 7
            flag = True
            for i in range(len(poem)):
                if i != k and (poem[i] == '，' or poem[i] == '。'):
                    flag = False
                    break
                if i == k and (poem[i] != '，' and poem[i] != '。'):
                    flag = False
                    break
                if i == k :
                    k += 8
            if flag:
                poems_f.append(poem)
    return poems_f