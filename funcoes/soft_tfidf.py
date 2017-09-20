# -*- coding: utf-8 -*-
"""
Created on Sun May 21 21:27:49 2017
@author: junio
"""

from estruturas.tokenizer import Tokenizer
from funcoes.jarowinkler import JaroWinkler
import math

class softTFIDF():
   '''
    Computes the hybrid variance of TFIDF measure. 
    Metrica baseada em token.
    Descricao detalhada desta funcao no relatorio.    
   '''
   
   def __init__(self, tok=Tokenizer, lim=0.8, func=JaroWinkler()):
      self.DEF_LIMIAR = lim
      self.func = func
      self.tokenizer = tok
      self.name = "softTFIDF"


   def __str__(self):
      return '[{}]'.format(self.name)   
   
   # Calcula os pesos do bag of tokens
   def prepare(self, b):
      for t in b.listToken():
         b.setWeight(t,1.0/float(math.sqrt(b.size())))
      return b
   
   # obtem a maior similaridade entre tokens de dois bags
   def bestSim(self, s, bag):
      sim = 0.0
      lim = self.DEF_LIMIAR
      m = None
      for t in bag.listToken():
         sim = self.func.score(s,t)
         if(sim>=lim):
            m=t
            lim=sim
      return m,lim if(m) else None
            
   # calcula a similaridade a partir dos bag of tokens repassados
   def score(self, sb, tb):
      sim = 0.0
      
      for s in sb.listToken():
         if(tb.contains(s)): 
            sim += sb.getWeight(s)*tb.getWeight(s)
         else:
            it,isim = self.bestSim(s,tb)
            if(it):
               sim += sb.getWeight(s)*tb.getWeight(it)*isim
      return 1.0 if(sim>1.0) else sim
   
   # calcula a similaridade a partir das strings repassadas
   def scoreStr(self, s, t):
      f = self.tokenizer()
      sb = self.prepare(f.tokenize(s))
      st = self.prepare(f.tokenize(t))
      return self.score(sb,st)     