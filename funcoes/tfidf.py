# -*- coding: utf-8 -*-
"""
Created on Sun May 21 20:48:40 2017
@author: junio
"""

from estruturas.tokenizer import Tokenizer
import math

class TFIDF():
   '''
    TFIDF-based distance metric.   
    Metrica baseada em token.
    Descricao detalhada desta funcao no relatorio.       
   '''
   
   def __init__(self, tok=Tokenizer):
      self.tokenizer = tok
      self.name = "TFIDF"


   def __str__(self):
      return '[{}]'.format(self.name)   
   
   # Calcula os pesos do bag of tokens
   def prepare(self, b):
      for t in b.listToken():
         b.setWeight(t,1.0/float(math.sqrt(b.size())))
      return b
   
   # calcula a similaridade a partir dos bag of tokens repassados
   def score(self, sb, tb):
      sim = 0.0
      
      for t in sb.listToken():
         if(tb.contains(t)): 
            sim += sb.getWeight(t)*tb.getWeight(t)
      return sim
   
   # calcula a similaridade a partir das strings repassadas
   def scoreStr(self, s, t):
      f = self.tokenizer()
      sb = self.prepare(f.tokenize(s))
      st = self.prepare(f.tokenize(t))
      return self.score(sb,st)     