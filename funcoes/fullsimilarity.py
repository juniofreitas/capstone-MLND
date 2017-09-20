# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:23:21 2017
@author: junio
"""

from estruturas.tokenizer import Tokenizer
from funcoes.jarowinkler import JaroWinkler
import math

class FullSimilarity():
   '''
    Computes Full Similarity measure.
    For two sets X and Y, where Xi is the token in X e Yi is the token in Y,
    and n and m are number of tokens of X and Y, respectively
    the similarity score is:
       ` max sim(Xi,Yi)/n+m `
    Metrica baseada em token.
    Descricao detalhada desta funcao no relatorio.
   '''
      
   def __init__(self, tok=Tokenizer, lim=0.85, func=JaroWinkler()):
      self.DEF_LIMIAR = lim
      self.tokenizer = tok
      self.func = func
      self.name = "Full Similarity"


   def __str__(self):
      return '[{}]'.format(self.name)

   # Calcula os pesos do bag of tokens
   def prepare(self, b):
      for t in b.listToken():
         b.setWeight(t,1.0/float(math.sqrt(b.size())))
      return b   
  
   # calcula a similaridade a partir dos bag of tokens repassados
   def score(self, sb, tb):
      if(sb.size()==0 or tb.size()==0): return 0.0
      
      tsim = 0
      for x in sb.listToken():
         msim = 0
         tot = 0
         for y in tb.listToken():
            sim = self.func.score(x,y)
            if(sim>=self.DEF_LIMIAR): 
               msim += sim
               tot += 1
         if(tot>0):      
            tsim += msim/float(tot)
         else:
            tsim += 0
         
      return tsim / float(sb.size())
   
   # calcula a similaridade a partir das strings repassadas
   def scoreStr(self, s, t):
      f = self.tokenizer()
      sb = f.tokenize(s)
      st = f.tokenize(t)
      return self.score(sb,st)
   