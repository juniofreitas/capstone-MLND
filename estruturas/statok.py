# -*- coding: utf-8 -*-
"""
Created on Sat May 06 12:39:49 2017
@author: junio
"""

class StaTok():
   '''
    Esta classe armazena estatísticas de um token da coleção.
    É utilizada em conjunto com a classe Colecao
   '''
   
   def __init__(self):
      self.totalfreq = 0
      self.liddoc = []
      self.lfreqdoc = []
      # self.mappos = {} <-- não utilizado para esse projeto
   
   
   def __str__(self):
      return '<{}-{},{}>'.format(self.totalfreq,self.liddoc,self.lfreqdoc)
   
   
   def verIdDoc(self,iddoc):
      return iddoc in self.liddoc
   
   
   def getTotalFreq(self):
      return self.totalfreq
   
   
   def getFreq(self):
      return len(self.liddoc)
   
   
   def getFreqDoc(self,iddoc):
      lidx = [i for i in range(len(self.liddoc)) if self.liddoc[i]==iddoc]
      if(len(lidx)>0):
         return self.lfreqdoc[lidx[0]]
      else:
         return -1
   
   
   def addIdDoc(self, iddoc, freqdoc):
      self.liddoc.append(iddoc)
      self.lfreqdoc.append(freqdoc)
      self.totalfreq +=freqdoc
      return len(self.liddoc)
   
   
   def upFreqDoc(self,iddoc, f):
      lidx = [i for i in range(len(self.liddoc)) if self.liddoc[i]==iddoc]         
      if(len(lidx)>0):
         of = self.lfreqdoc[lidx[0]]
         self.lfreqdoc[lidx[0]] = f
         self.totalfreq += f - of


   def listIdDocs(self):
      return self.liddoc

         
   def listFreqDocs(self):
      return self.lfreqdoc

      
   