# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:38:44 2017
@author: junio
"""

from estruturas.tokenizer import Tokenizer

class Jaccard():
   '''
    Computes Jaccard measure.
    For two sets X and Y, the Jaccard similarity score is:
       `jaccard(X, Y) = \\frac{|X \\cap Y|}{|X \\cup Y|}`
    Note:
       In the case where both X and Y are empty sets, we define their Jaccard score to be 1.    
    Metrica baseada em token.
    Descricao detalhada desta funcao no relatorio.        
   '''
      
   def __init__(self, tok=Tokenizer):
      self.tokenizer = tok
      self.name = "Jaccard"


   def __str__(self):
      return '[{}]'.format(self.name)
   
   # calcula a similaridade a partir dos bag of tokens repassados
   def score(self, sb, tb):
      numCommon = 0.0
      if(sb.size()==0 or tb.size()==0): return 0.0
      
      for t in sb.listToken():
         if(tb.contains(t)): numCommon +=1
      return numCommon / float(sb.size() + tb.size() - numCommon)
   
   # calcula a similaridade a partir das strings repassadas   
   def scoreStr(self, s, t):
      f = self.tokenizer()
      sb = f.tokenize(s)
      st = f.tokenize(t)
      return self.score(sb,st)
   