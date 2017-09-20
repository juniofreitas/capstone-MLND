# -*- coding: utf-8 -*-
"""
Created on Sat May 20 10:36:36 2017

@author: junio
"""
from funcoes.jaro import Jaro

class JaroWinkler():
   '''
    Jaro distance metric. From 'An Application of the Fellegi-Sunter
    Model of Record Linkage to the 1990 U.S. Decennial Census' by
    William E. Winkler and Yves Thibaudeau.
    Metrica baseada em caractere.
    Descricao detalhada desta funcao no relatorio.    
   '''
   
   def __init__(self):
      self.name = 'Jaro Winkler'
      self.PREFIX = 4

   
   def __str__(self):
      return '[{}]'.format(self.name)
   
   
   def commonPrefixLength(self, maxLength, common1, common2):
      lcommon1 = len(common1)
      lcommon2 = len(common2)
      n = min(maxLength, min(lcommon1, lcommon2))
      i = 0
      while i<n:
         if(common1[i]!=common2[i]): return i
         i += 1
      return n
   
   # calculo da similaridade pelo ajuste do winkler
   def winklerScore(self, s, t):
      return self.commonPrefixLength(self.PREFIX,s,t)
   
   # calculo da similaridade 
   def score(self, s, t):
      jaro = Jaro()
      dist = jaro.score(s,t)
      if(dist<0 or dist>1):
         print('Valor da similaridade nao está entre 0 e 1')
         raise 
      prefLength = self.winklerScore(s,t)
      dist = dist + prefLength*0.1 * (1 - dist)
      return dist
      
      