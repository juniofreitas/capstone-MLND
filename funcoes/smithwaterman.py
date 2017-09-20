# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:08:35 2017
@author: junio
"""

class SmithWaterman():
   '''
    The Smith-Waterman algorithm performs local sequence alignment; 
    that is, for determining similar regions
    between two strings. Instead of looking at the total sequence, 
    the Smith–Waterman algorithm compares segments of
    all possible lengths and optimizes the similarity measure. 
    See the string matching chapter in the DI book (Principles of Data Integration).
    Metrica baseada em caractere.
    Descricao detalhada desta funcao no relatorio.    
   '''

   def __init__(self):
      self.name = 'Smith-Waterman'
      self.gap_cost = 1
      self.match = self.charMatchScore21
   
   
   def __str__(self):
      return '[{}]'.format(self.name)
   
   # peso para match 0
   def charMatchScore01(self, c, d):
      return 0 if c==d else -1;
   
   # peso para match 2
   def charMatchScore21(self, c, d):
      return 2 if c==d else -1;   
   
   # peso para match 0 ou 1
   def charMatchScoreIdent(self, c, d):
      return int(c == d)   
   
   # calculo da similaridade
   def score(self,s, t):
      if (s == '') or (t == ''): 
         return 0.0
      elif (s == t):
         return 1.0
      
      n = len(s)
      m = len(t)       
      min_value = min(n,m)
      max_cost = self.match('a','a')
      d = [x[:] for x in [[0.0] * int(m+1)] * int(n+1)]
      max_value = 0
      # Smith Waterman DP calculations
      i=1
      while i<=n:
         j = 1
         while j <= m:
            match = d[i - 1][ j - 1] + self.match(s[i - 1],t[j - 1])
            delete = d[i - 1][j] - self.gap_cost
            insert = d[i][j - 1] - self.gap_cost
            d[i][j] = max(0, match, delete, insert)
            max_value = max(max_value, d[i][j])
            j +=1
         i +=1
      # Normaliza
      if(max_cost > -self.gap_cost):
         min_value *= max_cost
      else:
         min_value *= -self.gap_cost
      if(min_value == 0):
         ret = 1.0
      else:
         ret = (max_value / float(min_value));      
      return ret
      



