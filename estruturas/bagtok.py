# -*- coding: utf-8 -*-
"""
Created on Sat May 06 12:39:49 2017
@author: junio
"""

class BagTok():
   '''
    Esta classe implementa uma estrutura que representa um conjunto de tokens.
    Cada token está associado a um peso. Cada Bag é criado a partir de um 
    processo de tokenização de um texto. Ex.: 
       - Texto: 'O que é que cai em pé e corre deitado'
       - Bag: [('o',1),('que',2),('é',1),('cai',1),('em',1),('pé',1),('e',1),
               ('corre',1),('deitado',1)]
    Neste exemplo, o texto foi tokenizado por espaço em branco (' '), gerando o 
    bag, tal que, cada peso associado a um token corresponde a frequência em que
    o token ocorre no texto.
   '''
   
   def __init__(self, tokens=None, dvalue=1.0):
      self.bag = {}
      self.totalw = 0.0
      self.prep = False
      
      if tokens is None:
         self.totalw = 0.0
      else:   
         for t in tokens:
            self.bag[t] = self.getWeight(t)+dvalue
            self.totalw += 1
      
   def __str__(self):
      return "Bag("+str(self.totalw)+"):"+str(self.bag.items())

   
   def ver(self):
      return self.totalw != 0.0


   def contains(self,t):
      return self.bag.has_key(t)

   
   def getWeight(self,t):
      w = self.bag.get(t)
      return w if w else 0.0


   def setWeight(self,t,w):
      w = w if w>=0 else 0
      ow = self.getWeight(t)
      self.totalw += w if ow == None else (w - ow) 
      self.bag[t] = w   

      
   def size(self):
      return len(self.bag)

   
   def getTotalWeight(self):
      return self.totalw

      
   def listToken(self):
      return self.bag.keys()
   
   
   def common(self,b):
      lb1 = set(self.listToken())
      lb2 = set(b.listToken())
      return lb1 & lb2

      
   def minus(self,b):
      lb1 = set(self.listToken())
      lb2 = set(b.listToken())
      return lb1 - lb2


   def union(self,b):
      lb1 = set(self.listToken())
      lb2 = set(b.listToken())
      return lb1 | lb2
   