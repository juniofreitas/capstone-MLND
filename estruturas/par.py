# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:14:33 2017
@author: junio
"""
import numpy as np

class Par():
   '''
    Esta classe implementa um par de itens e processa a similaridade 
    do par gerando seu vetor de similaridade. 
   '''
   
   V = 1 # rótulo verdadeiro
   F = 0 # rótulo falso   
    
   def __init__(self, iddoc1, iddoc2):
       self.iddoc1 = iddoc1 # refere-se ao index da lista de bagtok
       self.iddoc2 = iddoc2 # refere-se ao index da lista de bagtok
       self.vetsim = None # vetor de similaridade
       self.weight = 0.0 # Peso do par que é calculado pelo comite
       self.label = 0 # Rótulo real do par
       self.classi = self.F # Rótulo estimado do par
       self.pross = self.isprep = False # Flag que indica se o par foi processado 

       
   def __str__(self):
      return '({},{})={}'.format(self.iddic1,self.iddoc2,self.label)

                   
   def getPar(self):
      return self.iddoc1, self.iddoc2

   
   def getParUm(self):
      return self.iddoc1


   def getParDois(self):
      return self.iddoc2


   def getWeight(self):
      return self.weight

   
   def setWeight(self, w):
      self.weight = w

      
   def getLabel(self):
      return self.label

   
   def setLabel(self,l):
      self.label = l

      
   def getClassi(self):
      return self.classi

   
   def setClassi(self,c):
      self.classi = c      

      
   def getPross(self):
      return self.pross

   
   def setPross(self,p):
      self.pross = p      

      
   def getPrep(self):
      return self.isprep

   
   def setPrep(self,p):
      self.isprep = p  

   # Recebe a lista de metricas de similaridade e calcula a similaridade do par   
   def process(self, col, funcs):
      if(not col or not funcs): 
         print("Parametros invalidos")
         raise
      
      if(self.pross): return self.vetsim
      self.vetsim = np.zeros(len(funcs), dtype= float)      
      self.isprep = True
      if not self.isprep:
         return None
      bag1 = col.calcTFIDF(self.iddoc1)
      bag2 = col.calcTFIDF(self.iddoc2)
      
      i = 0;
      for func in funcs:
         self.vetsim[i] = func.score(bag1,bag2)
         i += 1
      self.calcWeight()
      self.pross = True
      return self.vetsim


   def getVetSim(self):
      if(len(self.vetsim)>0): return self.vetsim
      
      
   def calcWeight(self):
      if(len(self.vetsim)>0):
         self.weight = max(self.vetsim.max() - self.vetsim.min(),0.01)
                  