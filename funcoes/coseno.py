# -*- coding: utf-8 -*-
"""
Created on Sun May 21 20:20:59 2017
@author: junio
"""

from estruturas.tokenizer import Tokenizer
import math

class Coseno():
   '''
    Computes a variant of cosine measure known as Ochiai coefficient.   
    Metrica baseada em token.
    Descricao detalhada desta funcao no relatorio.
   '''
   
   def __init__(self, tok=Tokenizer):
      self.tokenizer = tok
      self.name = "Coseno"


   def __str__(self):
      return '[{}]'.format(self.name)   
   
   # calcula a similaridade a partir dos bag of tokens repassados
   def score(self, sb, tb):
      numCommon = 0.0
      if(sb.size()==0 or tb.size()==0): return 0.0
      
      for t in sb.listToken():
         if(tb.contains(t)): numCommon +=1.0
      return numCommon / (math.sqrt(float(sb.size())) * math.sqrt(float(tb.size())))
   
   # calcula a similaridade a partir das strings repassadas
   def scoreStr(self, s, t):
      f = self.tokenizer()
      sb = f.tokenize(s)
      st = f.tokenize(t)
      return self.score(sb,st)   