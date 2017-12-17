# -*- coding: utf-8 -*-
"""
Created on Sat May 06 12:39:49 2017
@author: junio
"""
import os
import re
from estruturas import install_path
from estruturas.bagtok import BagTok

class Tokenizer():
   '''
    Classe que implementa tokenizacao de strings. Esta classe implementa dois 
    tipos de tokenizacao:
       1. Por token: separa o texto por palavras (tokens)
       2. Por n-gran: separa o texto por tokens de tamanho N
   '''
   
   REMSIMB = "[\'\"\(\)\[\]\<\>\/]"
   ESPSIMB = "[,\.\!\?\+\-\=\:\*\$\&\%\#\@]"
   
   def __init__(self, sw=True):
      self.lower = True # flag para transformar para minusculo o token
      self.simbol = False # flag para remover simbolos do token
      self.trunc = True # flag apra remover excesso de espacos no token
      self.stopword = sw # flag para remover stopword da tokenizacao
      self.delim = set([' ', '\t', '\n']) # delimitadores de token
      self.LSW = set() # Lista de stopword ingles 
      self.qval = 3 # tamnho no n-gram
      self.padding = False # flag para completar com sufixo e prefixo o n-gram
      self.prefix_pad = '#' # prefixo do ngram
      self.suffix_pad = '$' # sufixo do ngram
      self.dvalue = 1.0 # valor do peso padrao dos tokens do bag gerado
      if(self.stopword==2):
         self.loadStopWords(f='minimal-stop.txt')
      elif(self.stopword):
         self.loadStopWords()
   
   
   def get_install_path(self):
      path_list = install_path.split(os.sep)
      return os.sep.join(path_list[0:len(path_list)-1])   


   def loadStopWords(self,f='stop_words.txt'):
      if(len(self.LSW)==0):
         install_path = self.get_install_path()
         dataset_path = os.sep.join([install_path, 'estruturas'])         
         stop_words_file = os.sep.join([dataset_path, f])
         with open(stop_words_file, "rb") as stopwords_file:
            for stop_words in stopwords_file:
               self.LSW.add(stop_words.rstrip())
         stopwords_file.close()

   
   def getStopWord(self):
      return self.LSW

   
   def remSymbol(self, txt):
      txt = re.sub(self.REMSIMB,'',txt)
      txt = re.sub(self.ESPSIMB,'',txt)
      return txt      

   
   def setDvalue(self,dvalue):
      self.dvalue = dvalue

   
   def genBag(self, ltok):
      if len(ltok)==0:
         return BagTok(None)
      else:
         return BagTok(ltok,self.dvalue)
            
   # TOKENIZE PADRAO 9POR TOKEN)
   def tokenize(self, txt):
      if len(txt)==0:
         return None
            
      if(self.simbol):
         txt = self.remSymbol(txt)
      
      if(self.trunc):
         txt = re.sub(' +',' ',txt.strip())
         
      if(self.lower):
         txt = txt.lower()         
            
      if(self.stopword):
         ltoken = list(filter(lambda x: len(x)>1 and x not in self.LSW,txt.split()))
      else:
         ltoken = list(filter(lambda x: len(x)>1,txt.split()))   
      
      # Cria e retorna o bag of token 
      bag = self.genBag(ltoken)
      del ltoken[:]
      return bag 
   
   
   # TOKENIZE NGRAM 
   def tokenize_ngram(self, txt):
      if len(txt)==0:
         return None

      if(self.simbol):
         txt = self.remSymbol(txt)
      
      if(self.trunc):
         txt = re.sub(' +',' ',txt.strip())
         
      if(self.lower):
         txt = txt.lower()  
         
      ltoken = []

      if self.padding:
         txt = (self.prefix_pad * (self.qval - 1)) + txt + (self.suffix_pad * (self.qval - 1))   
      
      if len(txt) < self.qval:
         return self.genBag(ltoken)
      
      ltoken = [txt[i:i + self.qval] for i in xrange(len(txt) - (self.qval - 1))]      
      ltoken = list(filter(None, ltoken))

      # Cria e retorna o bag of token 
      return self.genBag(ltoken)
   
