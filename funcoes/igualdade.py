# -*- coding: utf-8 -*-
"""
Created on Sat May 20 11:19:18 2017
@author: junio
"""

class Igualdade():
   '''
    Verifica se uma string é igual a outra
    NÃO UTILIZADA NESTE PROJETO!!!
   '''
   
   def __init__(self):
      self.name = 'Igualdade'

   
   def __str__(self):
      return '[{}]'.format(self.name)   
   
   
   def score(self, s, t):
      if (s == '') or (t == '') or s!=t:
         return 0.0
      else:
         return 1.0
      
