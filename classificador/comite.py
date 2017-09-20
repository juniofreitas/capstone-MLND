# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:10:26 2017
@author: junio
"""

from collections import namedtuple
import numpy as np
import pandas as pd

class Comite():
   '''
    Esta classe representa um comite de classificadores conforme descrito no
    relatorio. Esta implementacao trata apensa de classicacao binario (0 ou 1)
   '''

   # Estrutura que representa um item do comite: nome, modelo e peso
   ItemCom = namedtuple('ItemCom',['model', 'name', 'weight'])
   V = 1 # indica valor verdadeiro
   F = 0 # indica valor falso        
   
   def __init__(self, models, names, w = 0.0):
      self.models = [] 
      self.weight = w # peso medio do comite
      self.ncertos = 0 # numero de acertos do comite
      self.models = [self.ItemCom(m,n,0.0) for m,n in zip(models,names)] # membros 
      self.DEFLIM = 0.5 # Threshold padrao  
      self.LIM = 0.5 # Threshold padrao

   # Imprime um comite       
   def __str__(self):
      strp = 'COMITE({},{}):\n'.format(self.weight, self.ncertos)
      for m in self.models:
         strp += '[{}={}]\n'.format(m.name,m.weight)
      return strp

   # Obtem os membros do comite
   def getMembers(self):
      return [m.name for m in self.models]
   
   # Obtem o peso medio do comite
   def getWeight(self):
      return self.weight
   
   # Atribui um peso medio ao comite
   def setWeight(self, w):
      self.weight = w
      
   # Obtem o numero de acertos do comite  
   def getNcertos(self):
      return self.ncertos

   # Ajuste da rotulagem baseada no peso do membro quando o rotulo for 0 (falso)
   def getLabelWeight0(self):
      lim = self.LIM
      return [lim - (m.weight*self.LIM) for m in self.models]

   # Ajuste da rotulagem baseada no peso do membro quando o rotulo for 1 (verdadeiro)
   def getLabelWeight1(self):
      lim = 1.0 - self.LIM
      return [lim + (m.weight*self.LIM) for m in self.models]

   # Ordena os membros do comite por peso    
   def sortWeight(self):
      return sorted(self.models, key=lambda itemcom: itemcom.weight)
   
   # Calcula o peso de um determinado par a partir do vetor de similaridade
   def calcPairWeight(self, df):
      return df.loc[:].apply(lambda x: max(max(x)-min(x),0.01), axis=1)

   # Estima o rotulo dos pares
   def predict(self, x):
      dfvotos = self.classify(x)
      qv = len(dfvotos.columns)
      ntrue = (dfvotos == self.V).sum(axis=1) #ntrue
      return np.array(map(int,ntrue/float(qv)>=self.DEFLIM))
            
   # Estima o rotulo dos pares apos ajuste conforme descrito no relatorio
   def predict_ajust(self, x):
      dfvotos = self.ajust(self.classify(x))
      qv = len(dfvotos.columns)
      tots = (dfvotos).sum(axis=1)
      return np.array(map(int,tots/float(qv)>=self.DEFLIM))

   # Calcula a probabilidade dos pares (como verdadeiro ou falso) com ajuste
   def predict_proba(self, xtest):
      y_proba = np.full((len(xtest),2),0.0)
      dfvotos = self.ajust(self.classify(xtest))
      qv = len(dfvotos.columns)   
      y_proba[:,0] = (dfvotos == self.V).sum(axis=1)/float(qv)
      y_proba[:,1] = 1.0 - y_proba[:,0]
      return y_proba

   # Calcula a probabilidade dos pares (como verdadeiro ou falso)
   def predict_proba_original(self, xtest):
      y_proba = np.full((len(xtest),2),0.0)
      for m in self.models:
         y_proba += np.dot(m.model.predict_proba(xtest),m.weight)
      return y_proba
   
   # Processo de votacao dos membros sobre os pares 
   def classify(self, x):
      votos = np.full((len(x),len(self.models)),None)
      for i,m in enumerate(self.models):
         votos[:,i] = m.model.predict(x)
      dfvotos = pd.DataFrame(votos, columns=[self.getMembers()])
      return dfvotos

   # Processo de ajuste dos votos dos membros a partir do pesos calculados
   def ajust(self, dfvotos):
      lw0 = self.getLabelWeight0()
      lw1 = self.getLabelWeight1()
      adf = dfvotos
      for i,m in enumerate(self.models):
         adf[m.name] = dfvotos[m.name].replace([0,1],[lw0[i],lw1[i]])
      return adf
      
   # Treinamento do comite com os pares de treino: votação e calculo dos pesos
   def treinoPar(self, xtrain, ytrain, wtrain):
         dfvotos = self.classify(xtrain)
         qv = len(dfvotos.columns)
         ncerto = np.full((len(xtrain)),0)
         ntrue = (dfvotos == self.V).sum(axis=1) #ntrue
         ncerto[:] = 0 # ncerto              
         for column in dfvotos:
            dfcol = dfvotos[column]
            ncerto += map(int,dfcol.reset_index(drop=True)==ytrain.reset_index(drop=True))
         nerro =  qv - ncerto# nerro
         #clabel = np.array(map(int,ntrue>(qv/2))) #label comite
         clabel = np.array(map(int,ntrue/float(qv)>=self.DEFLIM))
         self.ncertos =  list(clabel==ytrain).count(True)
         
         # Calculo dos pesos dos membros!!!!
         self.weight = 0.0
         for i,m in enumerate(self.models):
            vet1 = [w*nc for w,nc,y,l,v in zip(wtrain,ncerto,ytrain.reset_index(drop=True),clabel,dfvotos[m.name]) if(l==y and v==y)]
            vet2 = [w/float(nc) for w,nc,y,l,v in zip(wtrain,ncerto,ytrain.reset_index(drop=True),clabel,dfvotos[m.name]) if(l==y and v<>y)]
            vet3 = [w*ne for w,ne,y,l,v in zip(wtrain,nerro,ytrain.reset_index(drop=True),clabel,dfvotos[m.name]) if(l<>y and v==y)]
            vet4 = [w/float(ne) for w,ne,y,l,v in zip(wtrain,nerro,ytrain.reset_index(drop=True),clabel,dfvotos[m.name]) if(l<>y and v<>y)]
            self.models[i] = self.ItemCom(m.model,m.name,m.weight+sum(vet1) + sum(vet2) + sum(vet3) + sum(vet4))

         # Normaliza ...         
         for i,m in enumerate(self.models):
            self.models[i] = self.ItemCom(m.model,m.name,(m.weight/float(len(ytrain)*len(self.models))) )
            self.weight += self.models[i].weight
         self.weight = self.weight / float(len(self.models))
         self.LIM = self.weight+0.1
         self.DEFLIM = self.weight
         print 'Limiar={}'.format(self.weight)
         
         '''
         if(clabel==par.getLabel()):
            if(votos[i]==par.getLabel()):
               self.models[i] = self.ItemCom(m.model,m.name,m.weight + par.getWeight()*ncerto)  
            else:
               self.models[i] = self.ItemCom(m.model,m.name,m.weight + par.getWeight()/float(ncerto))   
               
         else:#(clabel!=par.getLabel()):
            if(votos[i]==par.getLabel()):
               self.models[i] = self.ItemCom(m.model,m.name,m.weight + par.getWeight()*nerro)  
            else:
               self.models[i] = self.ItemCom(m.model,m.name,m.weight + par.getWeight()/float(nerro))
         self.weight += self.models[i].weight
         '''            
   
   # Reune todo o processo de treinamento
   def fit(self, xtrain, ytrain):
      wtrain = self.calcPairWeight(xtrain)
      self.treinoPar(xtrain,ytrain,wtrain)

