# -*- coding: utf-8 -*-
"""
Created on Wed May 10 01:05:51 2017
@author: junio
"""

from estruturas.tokenizer import Tokenizer
from estruturas.statok import StaTok
from estruturas.infinityarray import InfinityArray
import math
from util.cleaner import Cleaner

class Colecao():
   '''
    Classe que implementa uma lista invertida e uma lista de bag of tokens.
    Uma lista invertida é um indice de texto, onde cada palavra aponta para
    o documento (registro) em que ela pertence, conforme esquematizado abaixo:
    Dataset:
       1. aa bb ccc
       2. bb ddd mm
       3. mm ccc aa
       4. bb ddd bb
    Lista invertida:
       aa  => [1, 3]
       bb  => [1, 2, 4]
       ccc => [1]
       ddd => [2, 4]
       mm  => [2, 3]
    [aa, bb, ccc, ddd, mm] é o vocabulário. 
    Os documentos de ids 1,2,3 e 4 são armazenados como bag of tokens.
   '''
   
   def __init__(self, ltxt=None, name='colecao', tks=None, stopw=True, ngram=False):      
      self.tokenizer = tks if tks else Tokenizer(sw=stopw)
      self.voc = {} # Lista invertida: vocabulario => lista
      self.totaltok = 0 # Total de tokens do dataset
      self.documentos =  InfinityArray(name+'.bin',name+'.idx') # lista de bagtok
      self.documentos.open() 
      self.ngram = ngram
      if(ltxt is None or self.documentos.ready()): 
         for i in xrange(self.documentos.size):
            bag = self.documentos.getx(i)
            for t in bag.listToken():
               st = self.voc.get(t)
               if not st: 
                  st = StaTok()
                  self.voc[t] = st
               st.addIdDoc(i,bag.getWeight(t))
               self.totaltok += 1
      else:
         i = 0     
         cls = Cleaner()
         for txt in ltxt:
            txt = cls.onlytext(str(txt))
            bag = self.tokenizer.tokenize_ngram(txt) if(self.ngram) else self.tokenizer.tokenize(txt)
            i += 1
            for t in bag.listToken():
               st = self.voc.get(t)
               if not st: 
                  st = StaTok()
                  self.voc[t] = st
               st.addIdDoc(i,bag.getWeight(t))
               self.totaltok += 1
            self.documentos.append(bag)
         self.documentos.end()

   
   def setDoc(self, i, txt):
      txt = str(txt)
      bag = self.tokenizer.tokenize_ngram(txt) if(self.ngram) else self.tokenizer.tokenize(txt)
      for t in bag.listToken():
         st = self.voc.get(t)
         if not st: 
            st = StaTok()
            self.voc[t] = st
         st.addIdDoc(i,bag.getWeight(t))
         self.totaltok += 1
      self.documentos.append(bag)
      

   def getDocumentFrequency(self, t):
      st = self.voc.get(t)
      if not st:
         return 0.0
      else:
         return st.getFreq()


   def BagFrequency(self,b):
      bf = {}
      for t in b.listToken():
         bf[t] = self.getDocumentFrequency(t)
      return bf

   
   def getDocumentFrequencyTot(self, t):
      st = self.voc.get(t)
      if not st:
         return 0.0
      else:
         return st.getTotalFreq()


   def BagFrequencyTot(self,b):
      bf = {}
      for t in b.listToken():
         bf[t] = self.getDocumentFrequencyTot(t)
      return bf         

      
   def getDocumentStatistic(self, t):
      return self.voc.get(t)

   
   def getFrequencyDoc(self, t, d):
      st = self.voc.get(t)
      if not st:
         return 0.0
      else:
         return st.getFreqDoc(d)

      
   def BagFrequencyDoc(self, b, d):
      bf = {}
      for t in b.listToken():
         bf[t] = self.getFrequencyDoc(t,d)
      return bf


   def getListDoc(self, t):
      st = self.voc.get(t)
      if not st:
         return 0.0
      else:
         return st.listIdDocs()      

   
   def size(self):
      if self.voc:
         return len(self.voc)
      else:
         return 0   

   
   def getTotalTok(self):
      return self.totaltok
   
   
   def getTotalDoc(self):
      if self.documentos:
         return (self.documentos.size)
      else:
         return 0.0   

   
   def getDocumentos(self):
      return self.documentos
   
   
   def getDoc(self,iddoc):
      if(iddoc<0 and iddoc>=self.getTotalDoc()):
         return None
      else:
         if(not self.documentos.ready()):
            self.documentos.start()
         return self.documentos.getx(iddoc)
   
   # Calcula o peso dos bags como TF e IDF conforme os tokens do dataset
   # aparecem na colecao. TF e IDF sao usados no calculo de similaridade    
   def calcTFIDF(self,iddoc):
      norm = 0.0      
      bag = self.getDoc(iddoc)      
      if(not bag):
         print "Erro!!!"
         return None
      if(bag.prep): return bag
      
      collectionSize = self.getTotalDoc()
      for t in bag.listToken():
         if(collectionSize>0):
            df = self.getDocumentFrequency(t)
            if(df==0):
               df=1.0
            w = math.log(bag.getWeight(t)+1) * math.log(collectionSize/df)
            bag.setWeight(t,w)
            norm += w*w
         else:
            bag.setWeight(t,1.0)
            norm += 1.0
      # Normaliza
      norm = math.sqrt(norm)
      for t in bag.listToken():
         bag.setWeight(t,bag.getWeight(t)/norm)
      bag.prep = True
      return bag
