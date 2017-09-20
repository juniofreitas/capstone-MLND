# -*- coding: utf-8 -*-
"""
Created on Sat May 20 19:07:46 2017
@author: junio
"""

from estruturas.tokenizer import Tokenizer
import math

class FelligiSunter():
   '''
    Computes Jaccard measure. <== corrigir 
    For two sets X and Y, the Jaccard similarity score is:
       `jaccard(X, Y) = \\frac{|X \\cap Y|}{|X \\cup Y|}`
    Note:
       In the case where both X and Y are empty sets, we define their Jaccard score to be 1.    
    Metrica baseada em token.
    Descricao detalhada desta funcao no relatorio.
   '''
      
   def __init__(self, tok=Tokenizer, mmf = 0.5):
      self.tokenizer = tok
      self.mismatchFactor = mmf
      self.name = "Felligi Sunter"


   def __str__(self):
      return '[{}]'.format(self.name)
   
   # Calcula os pesos do bag of tokens
   def prepare(self, b):
      for t in b.listToken():
         b.setWeight(t,1)
      return b
   
   # calcula a similaridade a partir dos bag of tokens repassados
   def score(self, sb, tb):
      sim = 0.0
      for t in sb.listToken():
         if(tb.contains(t)):
            p = tb.getWeight(t)
            sim += p
            #p = math.exp(-p); 
            #sim -= math.log( 1.0 - math.exp(sb.size() * tb.size() * math.log(1.0 - p*p)))
         else:
            sim -= sb.getWeight(t)*self.mismatchFactor
      return sim

   # calcula a similaridade a partir das strings repassadas
   def scoreStr(self, s, t):
      f = self.tokenizer()
      sb = self.prepare(f.tokenize(s))
      st = self.prepare(f.tokenize(t))
      return self.score(sb,st)
   
   
   