# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:18:30 2017
@author: junio
"""

class Jaro():
   '''
    Jaro distance metric. From 'An Application of the Fellegi-Sunter
    Model of Record Linkage to the 1990 U.S. Decennial Census' by
    William E. Winkler and Yves Thibaudeau.
    Metrica baseada em caractere.
    Descricao detalhada desta funcao no relatorio.
   '''
   
   def __init__(self):
      self.name = 'Jaro'

   
   def __str__(self):
      return '[{}]'.format(self.name)

   # Calcula o numero de transposicoes para trandformar uma string em outra
   def transpositions(self, common1, common2):
      transpositions = 0
      for i, c in enumerate(common1):		    
         if(common1[i]!=common2[i]):           				
            transpositions +=1
      transpositions /= 2 
      return transpositions

   # Calcula o numero de caracteres em comum
   def commonChars(self, s, t, halflen):			
      common = ''
      copy = t		
      for i,c in enumerate(s): 			
         ch = s[i]
         foundIt = False			
         j = max(0,i-halflen)
         while not foundIt and j<min(i+halflen+1,len(t)):   			
            if (copy[j]==ch):
               foundIt = True
               common += ch
               copy = copy[:j] + '*' + copy[j+1:]
            j += 1				
      return common
      
   # calcula o tamanho medio da menor string
   def halfLengthOfShorter(self, str1, str2):	
      tamstr1 = len(str1)
      tamstr2 = len(str2)	    
      return tamstr2/2 + 1 if tamstr1>tamstr2 else tamstr1/2 +1
   
   # calculo da similaridade 
   def score(self, s, t):	
      str1 = s
      str2 = t

      if (str1 == '') or (str2 == ''):
         return 0.0
      elif (str1 == str2):
         return 1.0
 
      halflen = self.halfLengthOfShorter(str1,str2)
      common1 = self.commonChars(str1, str2, halflen)
      common2 = self.commonChars(str2, str1, halflen)
      lcommon1 = len(common1)
      lcommon2 = len(common2)
      
      if (lcommon1!=lcommon2): return 0.0
      if (lcommon1==0 or lcommon2==0): return 0.0
		
      transpositions = self.transpositions(common1,common2);
		
      dist = (
         lcommon1/(float(len(str1))) + 
         lcommon2/(float(len(str2))) + 
         (lcommon1-transpositions)/(float(lcommon1) )) / 3.0;
      
      return dist;
	   
   