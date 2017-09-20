# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:05:32 2017
@author: junio
"""

from funcoes.jarowinkler import JaroWinkler

class Soundex():
   '''
    Phonetic measure such as soundex match string based on their sound. These
    measures have been especially effective in matching names, since names are
    often spelled in different ways that sound the same. For example, Meyer, Meier,
    and Mire sound the same, as do Smith, Smithe, and Smythe.
    Soundex is used primarily to match surnames. It does not work as well for names
    of East Asian origins, because much of the discriminating power of these names
    resides in the vowel sounds, which the code ignores.
    Metrica baseada em caractere.
    Descricao detalhada desta funcao no relatorio.    
   '''

   def __init__(self):
      self.name = 'Soundex'

   
   def __str__(self):
      return '[{}]'.format(self.name)
   
   # gera o codigo soundex
   def getSoundex(self,name):
      name = name.upper()
      soundex = ""
      soundex += name[0]
      dictionary = {"BFPV": "1", "CGJKQSXZ":"2", "DT":"3", "L":"4", "MN":"5", "R":"6", 
                    "AEIOUHWY":"."}
      for char in name[1:]:
         for key in dictionary.keys():
            if char in key:
               code = dictionary[key]
               if code != soundex[-1]:
                  soundex += code
      soundex = soundex.replace(".", "")
      soundex = soundex[:4].ljust(4, "0")
      return soundex      
   
   # calculo da similaridade usando Jaro
   def score(self, s, t):
      ss = self.getSoundex(s)
      st = self.getSoundex(t)
      f = JaroWinkler()
      return f.score(ss,st)
   
   
   