# -*- coding: utf-8 -*-
"""
Created on Sat May 20 13:37:05 2017
@author: junio
"""

class MongeElkan():
   '''
     The match method proposed by Monge and Elkan.  They called this
     Smith-Waterman, but actually, this uses an affine gap model, so
     it's not Smith-Waterman at all, according to the terminology in
     Durban, Sec 2.3.
     Costs are as follows: 
     mismatched char = -3, match = +5 (case insensitive), approximate match = +3,
     for pairings in {dt} {gj} {lr} {mn} {bpv} {aeiou} {,.}, start gap = +5, 
     continue gap = +1
     Metrica de caractere.
     NÃO UTILIZADA NESTE PROJETO!!!
    '''

   def __init__(self):
      self.name = 'Monge-Elkan'

   
   def __str__(self):
      return '[{}]'.format(self.name)
   
   # calculo da similaridade
   def score(self, s, t):
      if (s == '') or (t == ''): 
         return 0.0
      elif (s == t):
         return 1.0
      n = len(s)
      m = len(t)         
      # Scores used for Smith-Waterman algorithm - - - - - - - - - - - - - - - - -
      match_score =       5
      approx_score =      2
      mismatch_score =   -5
      gap_penalty =       5
      extension_penalty = 1
      common_divisor = 'average'
      # Calculate the divisor - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (common_divisor == 'average'):
         divisor = 0.5*(n+m)*match_score  # Average maximum score
      elif (common_divisor == 'shortest'):
         divisor = min(n,m)*match_score
      else:  # Longest
         divisor = max(n,m)*match_score
      # Dictionary with approximate match characters mapped into numbers
      # {a,e,i,o,u} -> 0, {d,t} -> 1, {g,j} -> 2, {l,r} -> 3, {m,n} -> 4,
      # {b,p,v} -> 5
      approx_matches = {'a':0, 'b':5, 'd':1, 'e':0, 'g':2, 'i':0, 'j':2, 'l':3,
                        'm':4, 'n':4, 'o':0, 'p':5, 'r':3, 't':1, 'u':0, 'v':5}
      best_score = 0  # Keep the best score while calculating table
      d = [x[:] for x in [[0.0] * int(m+1)] * int(n+1)]
      i = 1
      while (i <= n):
         #-- Step 4
         j = 1          
         while (j <= m):
            match = d[i-1][j-1]
            if (s[i-1] == t[j-1]):
               match += match_score
            else:
               approx_match1 = approx_matches.get(s[i-1],-1)
               approx_match2 = approx_matches.get(t[j-1],-1)
               if (approx_match1 >= 0)and(approx_match2 >= 0)and(approx_match1 == approx_match2):
                  match += approx_score
               else:
                  match += mismatch_score            
            insert = 0
            for k in range(1,i):
               score = d[i-k][j] - gap_penalty - k*extension_penalty
               insert = max(insert, score)
            delete = 0
            for l in range(1,j):
               score = d[i][j-l] - gap_penalty - l*extension_penalty
               delete = max(delete, score)
            d[i][j] = max(match, insert, delete, 0)
            best_score = max(d[i][j], best_score) 
            j +=1
         i += 1
      # a sub-string ofd the other string).
      # The lower best_score the less similar the sequences are.
      w = float(best_score) / float(divisor)
      assert (w >= 0.0) and (w <= 1.0), 'Similarity weight outside 0-1: %f' % (w)      
      return w
      
      
      
      
      
      
      