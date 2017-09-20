# -*- coding: utf-8 -*-
"""
Created on Sat May 20 11:44:22 2017

@author: junio
"""

class Levenstein():
   '''
    The LEvenstein (edit distance) is the minimal number of insertions, 
    deletions and substitutions needed to make two strings equal.
    Metrica baseada em caractere.
    Descricao detalhada desta funcao no relatorio.    
   '''
   
   def __init__(self):
      self.name = 'Levenstein'

   
   def __str__(self):
      return '[{}]'.format(self.name)
   
   # calculo do peso
   def custo_edicao(self, s, sIndex, t, tIndex):
      return 0.0 if(s[sIndex] == t[tIndex]) else 1.0
   
   # processo de distancia de edicao
   def levenstein(self, s, t):
      n = len(s) # length of s
      m = len(t) # length of t
      cost=0.0 # cost
      d = [x[:] for x in [[0] * int(n+1)] * int(m+1)] # d[MAX_STRING_LENGTH][MAX_STRING_LENGTH]; # matrix  
      #-- Step 1
      if (n == 0):
         return m
      if (m == 0):
         return n
      #-- Step 2
      i = 0
      while (i <= n):      
         d[0][i] = i
         i += 1
      j = 0
      while (j <= m):      
         d[j][0] = j
         j += 1
      #//-- Step 3
      i = 1
      while (i <= n):
         #-- Step 4
         j = 1          
         while (j <= m):
            #-- Step 5
            cost = self.custo_edicao(s, i - 1, t, j - 1)
            #-- Step 6
            d[j][i] = min(d[j][i - 1] + 1, d[j - 1][i] + 1)
            d[j][i] = min(d[j][i], d[j - 1][i - 1] + cost)
            j += 1
         i += 1
      #-- Step 7
      return d[m][n]
   
   # calculo da similaridade
   def score(self, s, t):
      #--
      levensteinDistance = self.levenstein(s, t)
      #-- get the max possible levenstein distance score for string
      maxLen = len(s)
      if (maxLen < len(t)): maxLen = len(t)
      #-- check for 0 maxLen
      if (maxLen == 0):
         levensteinDistance = 1.0 # as both strings identically zero length
      else:
         levensteinDistance = (1.0 - (levensteinDistance / maxLen)) #return actual / possible levenstein distance to get 0-1 range
      return levensteinDistance
