# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:25:37 2017
@author: junio
"""

from estruturas.tokenizer import Tokenizer

class BagSim():
   '''
    Computes diff either bag weights.
    Metrica baseada em token.
    Descricao detalhada desta funcao no relatorio.
   '''
      
   def __init__(self, tok=Tokenizer):
      self.tokenizer = tok
      self.name = "BagSim"


   def __str__(self):
      return '[{}]'.format(self.name)
   
   # calcula a similaridade a partir dos bag of tokens repassados
   def score(self, sb, tb):
      ws = sb.getTotalWeight()
      wt = tb.getTotalWeight()
      if(ws == wt): return 1.0
      
      htok = {}
      tsim = 0.0      
      for x in sb.listToken():
         ssim = 0.0
         mindif = 0.1
         ytok = None
         for y in tb.listToken():
            if(x == y):               
               ssim = sb.getWeight(x)*tb.getWeight(y)
               ytok = y
               break
            elif abs(sb.getWeight(x)-tb.getWeight(y))<=mindif and not htok.has_key(y):
               ssim = sb.getWeight(x)*tb.getWeight(y)
               mindif = abs(sb.getWeight(x)-tb.getWeight(y))
               ytok = y
         if(ytok is not None): htok[ytok] = 1
         tsim += ssim
         if (tsim>1): tsim=1.0
      return tsim 
   
   # calcula a similaridade a partir das strings repassadas
   def scoreStr(self, s, t):
      f = self.tokenizer()
      sb = f.tokenize(s)
      st = f.tokenize(t)
      return self.score(sb,st)
   