# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:23:04 2019

@author: danish
"""

import pandas as pd
import numpy as np
import re
import regex
import time

files = open("texts/machado") # change this for the raw text file
# read lines and convert to lowercase
data = [i.lower() for i in files.readlines()]

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

unique_data =  list(uniq(sorted(data, reverse=True)))

sentence_count = {}
total_sentences = 0
for line in unique_data:
    if line not in sentence_count:
        sentence_count[line] = 1
    else:
        sentence_count[line] += 1
    total_sentences += 1
        
      
# Removing the lines that contains numeric data 
unique_data_str = []
for i in range(len(unique_data)):
    if type(unique_data[i]) is str:
        unique_data_str.append(unique_data[i])
    else:
        None

# Removing the lines which are to short or to long
short_data = []
for line in unique_data_str:
    if 5 <= len(line.split()) <= 30:
        short_data.append(line)
    else:
        None

# Counting the appearnce of each word in the corpus also calculates the number of unique words also
word2count = {}
total_words = 0
for text in short_data:
    for word in text.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
        total_words += 1
        
# creating a list that will only contain the words that appear more than 15 times
word15 = []
threshold = 15
for word, count in word2count.items():
    if count >= threshold:
        if len(word) > 1:
            word15.append(word)

print(f"Words in the vocubulary before thresholding: {total_words}")
            
# Removing the words from each string which appear less than 15 times
data_15 = []
for line in short_data:
    str1=''
    for word in line.split():
        if word in word15:
            str1 = " ".join((str1, word))
    data_15.append(str1)

#      
short_data_consize = []
for line in data_15:
    if 3 <= len(line.split()) <= 15:
        short_data_consize.append(line)
    else:
        None
        
clean_unique_data = []
for i in range(len(short_data_consize)):
    if short_data_consize[i] not in clean_unique_data:
        clean_unique_data.append(short_data_consize[i])
    else:
        None
        
# Total number of words in corpus after removing the words which appears less than 15 times and further cleaning
total_words_d15 = 0
for line in data_15:
    for word in line.split():
      total_words_d15 += 1 
print(f"Words in the vocubulary after thresholding: {total_words_d15}")
    
#defining a function to save data
def write_txt(name, data):
    file1 = open("{0}.txt".format(name),"w") 
    for line in data:
        file1.writelines(line) 
        file1.writelines('\n') 
    file1.close() #to change file access modes

# Reading text file
#fl = open("EU-AU-Description-19-9-2019.txt","r+")  
#clean_unique_data = fl.read().splitlines()



# Splitting the cleaned and preprocessd data into 4 equal parts      
clean_unique_data_qtr1 = clean_unique_data[0:int(len(clean_unique_data)*0.25)]      
clean_unique_data_qtr2 = clean_unique_data[int(len(clean_unique_data)*0.25):int(len(clean_unique_data)*0.5)]  
clean_unique_data_qtr3 = clean_unique_data[int(len(clean_unique_data)*0.5):int(len(clean_unique_data)*0.75)]  
clean_unique_data_qtr4 = clean_unique_data[int(len(clean_unique_data)*0.75):len(clean_unique_data)]        

# writing data to text files
write_txt(name = 'machado_preprocessed', data = clean_unique_data)
