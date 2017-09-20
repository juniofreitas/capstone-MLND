# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 09:41:06 2017
@author: u4uz
"""
import os
import re
from string import punctuation

class Cleaner():
   '''
    Realiza algumas limpezas no texto repassado
   '''
   
   def __init__(self):
      self.status = True
      self.LWS = set() # Lista de stop word ingles 
      
      
   def onlytext(self, text):
      # Clean the text
      text = text.lower()      
      text = re.sub(r"what's", "whats", text, flags=re.IGNORECASE)
      text = re.sub(r"\'s", " ", text)
      text = re.sub(r"\'ve", " have ", text)
      text = re.sub(r"can't", "cannot ", text)
      text = re.sub(r"n't", " not ", text)
      text = re.sub(r"I'm", "I am", text)
      text = re.sub(r" m ", " am ", text)
      text = re.sub(r"\'re", " are ", text)
      text = re.sub(r"\'d", " would ", text)
      text = re.sub(r"\'ll", " will ", text)
      text = re.sub(r"60k", " 60000 ", text)
      text = re.sub(r" e g ", " eg ", text)
      text = re.sub(r" b g ", " bg ", text)
      text = re.sub(r"\0s", "0", text)
      text = re.sub(r" 9 11 ", "911", text)
      text = re.sub(r"e-mail", "email", text)
      text = re.sub(r"\s{2,}", " ", text)
      text = re.sub(r"quikly", "quickly", text)
      text = re.sub(r" usa ", " America ", text)
      text = re.sub(r" USA ", " America ", text)
      text = re.sub(r" u s ", " America ", text)
      text = re.sub(r" uk ", " England ", text)
      text = re.sub(r" UK ", " England ", text)
      text = re.sub(r"india", "India", text)
      text = re.sub(r"switzerland", "Switzerland", text)
      text = re.sub(r"china", "China", text)
      text = re.sub(r"chinese", "Chinese", text) 
      text = re.sub(r"imrovement", "improvement", text)
      text = re.sub(r"intially", "initially", text)
      text = re.sub(r"quora", "Quora", text)
      text = re.sub(r" dms ", "direct messages ", text)  
      text = re.sub(r"demonitization", "demonetization", text) 
      text = re.sub(r"actived", "active", text)
      text = re.sub(r"kms", " kilometers ", text)
      text = re.sub(r"KMs", " kilometers ", text)
      text = re.sub(r" cs ", " computer science ", text) 
      text = re.sub(r" upvotes ", " up votes ", text)
      text = re.sub(r" iPhone ", " phone ", text)
      text = re.sub(r"\0rs ", " rs ", text) 
      text = re.sub(r"calender", "calendar", text)
      text = re.sub(r"ios", "operating system", text)
      text = re.sub(r"gps", "GPS", text)
      text = re.sub(r"gst", "GST", text)
      text = re.sub(r"programing", "programming", text)
      text = re.sub(r"bestfriend", "best friend", text)
      text = re.sub(r"dna", "DNA", text)
      text = re.sub(r"III", "3", text) 
      text = re.sub(r"the US", "America", text)
      text = re.sub(r"Astrology", "astrology", text)
      text = re.sub(r"Method", "method", text)
      text = re.sub(r"Find", "find", text) 
      text = re.sub(r"banglore", "Banglore", text)
      text = re.sub(r" J K ", " JK ", text)      
      text = re.sub(r"([0-9])[Kk] ",r"\1 000 ",text)
      text = re.sub(r"[^A-Za-z0-9]", " ", text)
      self.rempoint(text)
      return text


   def rempoint(self, text):      
      # Remove punctuation from text
      text = ''.join([c for c in text if c not in punctuation])
      return text


   def loadStopWords(self):
      if(len(self.LSW)==0):
         install_path = self.get_install_path()
         dataset_path = os.sep.join([install_path, 'estruturas'])         
         stop_words_file = os.sep.join([dataset_path, 'stop_words.txt'])
         with open(stop_words_file, "rb") as stopwords_file:
            for stop_words in stopwords_file:
               self.LSW.add(stop_words.rstrip())
         stopwords_file.close()
