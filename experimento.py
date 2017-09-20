# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 09:14:52 2017
@author: junio
Script de execução do experimento.
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.pyplot.switch_backend('agg')
# METRICAS
from sklearn.model_selection import train_test_split
from sklearn.metrics import  log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
# Transformacao e normalização
from sklearn import preprocessing
# CLASSIFICADORES 
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import VotingClassifier
# MINHAS ESTRUTURAS
from estruturas.colecao import Colecao
from estruturas.infinityarray import InfinityArray
from estruturas.par import Par
from classificador.comite import Comite
# MINHAS FUNCOES 
from funcoes.soundex import Soundex
from funcoes.jaccard import Jaccard
from funcoes.fullsimilarity import FullSimilarity
from funcoes.bagsim import BagSim
from funcoes.soft_tfidf import softTFIDF
from funcoes.tfidf import TFIDF
from funcoes.levenstein import Levenstein
from funcoes.coseno import Coseno
from funcoes.mongelkan import MongeElkan
from funcoes.smithwaterman import SmithWaterman

#----------------------------------------
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1500)
#----------------------------------------

class Experimento():
   ### CONSTANTES
   ddir = 'datasets'
   dname = 'train.csv'   
   dcname = 'classis'
   drname = 'res.csv'
   dlcolname = ['colecao','colecao_gram','colecao_sw','colecao_gram_sw']   
   dlparesname = ['pares','pares_gram','pares_sw','pares_gram_sw']
   dfeatures = {'Jaccard':Jaccard(),'BagSim':BagSim(),'Coseno':Coseno(),\
                'TFIDF':TFIDF(),'SoftTFIDF':softTFIDF(),'STFIDFSound':softTFIDF(func=Soundex()),\
                'STFIDFLeven':softTFIDF(func=Levenstein()),'STFIDFMonge':softTFIDF(func=MongeElkan()),\
                'STFIDFSmith':softTFIDF(func=SmithWaterman()),'FullSim':FullSimilarity(),\
                'FullSimSound':FullSimilarity(func=Soundex()),'FullSimLeven':FullSimilarity(func=Levenstein()),\
                'FullSimMonge':FullSimilarity(func=MongeElkan()),'FullSimSmith':FullSimilarity(func=SmithWaterman())}
   dclassis = {'MultinomialNB':MultinomialNB(),'GaussianNB':GaussianNB(),\
              'SGDClassifier':SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=10, learning_rate='optimal'),\
              'MLPClassifier':MLPClassifier(activation='relu',solver='lbfgs'),\
              'DecisionTreeClassifier':DecisionTreeClassifier(criterion='entropy',max_depth=20),\
              'RandomForestClassifier':RandomForestClassifier(criterion='entropy',max_depth=20, n_estimators=15),\
              'GradientBoostingClassifier':GradientBoostingClassifier(n_estimators=15, learning_rate=1.0, max_depth=20)}
   dmetricas = {'log_loss':log_loss,'f1_score':f1_score,'accuracy_score':accuracy_score,\
                'classification_report':classification_report,'precision_recall_curve':precision_recall_curve,\
                'average_precision_score':average_precision_score,'precision_score':precision_score,\
                'recall_score':recall_score,'confusion_matrix':confusion_matrix}
   
   def __init__(self):
      # Parametros do experimento
      self.name = None # nome do dataset
      self.ds = None # caminho completo do dataset
      self.data = None # dados do dataset
      self.listadocs = None # Lista de documentos obtidos do dataset
      self.n = None # número de rodadas
      self.features = None # features do dataset (funcoes de similaridade)
      self.classis = None # classificadores (tree, naive, svm, ...)
      self.classisname = None # nome do arquivo que aramazena os classificadores
      self.lclassis = None # array persistente que armazena os classificadores
      self.ntreino  = None # tamanho do treino
      self.metricas = None # metricas aplicadas ao resultado (f1, precision, recall, log_loss, ...)
      self.comite = None # parâmetros do comitê
      self.stopword = None # indica se stop words serao removidos dos dados
      self.ngram = None # indica se sera usado ngram
      self.colname = None # nome do arquivo da colecao de documentos
      self.colecao = None # Colecao de documentos com suas respectivas listas invertidas
      self.paresname = None # nome do arquivo da lista de pares
      self.pares = None # lista de pares de documentos
      self.vetsim = None # Lista de features (similaridade) dos pares
      self.X_train = None # Conjunto de Treino
      self.X_test = None #  Conjunto de Teste
      self.y_train = None # Conjunto dos rótulos do treino
      self.y_test  = None # Conjunto dos rótulos do teste
      self.tamcom = None # Tamanho do comite
      self.comite = None # Comite de classificadores
      self.resultados = None # Resultados dos testes
      self.resultados_report = None # Report dos resultados dos testes
      self.resultados_curve = None # Resultados dos testes de precision recall curve
      self.resultados_confusion = None # Resultados dos testes de Matriz de Confusao
      self.resname = None # Nome do arquivo de resultados
      self.seed = None # Semente de aleatoreidade

   # Seta o nome do dataset
   def set_name(self, nome):      
      self.name = nome      
   
   # Obtem o caminho completo do dataset
   def set_dataset(self):
      self.ds = os.sep.join([os.path.dirname(os.path.realpath(__file__)), self.ddir, self.name])
   
   # Carrga os dados do dataset
   def load_dataset(self):
      print "Carrega dataset..."
      if(self.ds is None): self.set_dataset()
      self.data = pd.read_csv(self.ds)
      self.data = self.data.fillna("null")

   # Obtem lista de documentos do dataset
   def load_listadocs(self):
      print "Carrega lista de pares de documentos..."
      if(self.data is None): self.load_dataset()
      ldoc1 = self.data[['qid1','question1']].rename(columns = {'qid1':'qid', 'question1':'question'})
      ldoc2 = self.data[['qid2','question2']].rename(columns = {'qid2':'qid', 'question2':'question'})
      self.listadocs = ldoc1.append(ldoc2).sort_values('qid').drop_duplicates()
      
   # Seta se removera stop words
   def set_stopword(self, sw):
      self.stopword = sw
   
   # Seta se os documentos serao divididos em ngram
   def set_ngram(self, ng):
      self.ngram = ng
   
   # Seta nome do arquivo de documentos 
   def set_colname(self, cn):
      self.colname = cn

   # Gera a colecao de documentos com suas respectivas listas invertidas
   def load_colecao(self):
      if(self.listadocs is None): self.load_listadocs()
      print "Carrega colecao..."
      self.colecao = Colecao(list(self.listadocs['question']),self.colname,stopw=self.stopword,ngram=self.ngram)         
   
   # Seta nome do arquivo de documentos 
   def set_paresname(self, pn):
      self.paresname = pn

   # Define as features
   def set_features(self,feat=dfeatures):
      self.features = feat
   
   # Obtem lista de nomes defeatures ordenada
   def get_fetures_name(self):
      if(self.features is None):
         return None
      else:
         return sorted(self.features.keys())      
      
   # Obtem lista de features ordenada
   def get_fetures(self):
      if(self.features is None):
         return None
      else:
         return [self.features[k] for k in sorted(self.features.keys())]
   
   # Obtem lista de pares e tambem processa suas respectivas similaridades
   def load_listapares(self):
      print "Carrega lista pares..."
      self.pares = InfinityArray(self.paresname,'x'+self.paresname)
      self.pares.open()
      if(not self.pares.ready()):
         for index, row in self.data[['id','qid1','qid2','is_duplicate']].iterrows():
            par = Par(int(row['qid1'])-1, int(row['qid2'])-1)
            par.setLabel(row['is_duplicate'])
            par.process(self.colecao,self.get_fetures())
            self.pares.append(par)   
            if(index%1000==0):
               print "\rProcessados {} pares: ({},{})={}".format(index,par.getLabel(),par.getWeight(),par.getVetSim())
   
   # Obtem o vetsim: lista de vetores de similaridade dos pares
   def load_vetsim(self):
      print "Carrega vetsim.."
      if(self.pares is None): self.load_listapares()
      self.pares.end()
      self.pares.start()
      self.vetsim = pd.DataFrame(data=np.array([self.pares.getx(i).getVetSim() for i in xrange(self.pares.size)], \
                     dtype = float), columns=list(self.get_fetures_name()))    
      self.pares.end()
   
   # Define a semente da raiz para os processos aleatorios
   def set_seed(self,s):
      self.seed = s
      random.seed(s)
      
   # Carrega as classes do arquivo
   def load_classis(self, cn):
      self.set_classisname(cn)
      self.lclassis = InfinityArray(self.classisname, self.classisname) 
      self.lclassis.open()
      if(self.lclassis.ready()):
         self.classis = {}
         for i in xrange(self.lclassis.size):
            k,v = self.lclassis.getx(i)
            self.classis[k] = v
         self.lclassis.end()
      else:
         self.lclassis.end()
         self.lclassis = None               
   
   # Salva os classificadores
   def save_classis(self,cn):
      if self.lclassis is None:
         self.set_classisname(cn)
         self.lclassis = InfinityArray(self.classisname, self.classisname) 
         #self.lclassis.open()
      self.lclassis.new()
      if not self.lclassis.ready() or self.lclassis.size==0:
         for nome, classe in self.classis.iteritems():   
            self.lclassis.append((nome, classe))
      self.lclassis.end()       
      
   # Define o nome do arquivo que armazena os classificador   
   def set_classisname(self, cn=dcname):
      self.classisname = cn
      
   # Define os classificadores
   def set_classis(self,cls=dclassis):      
      if(self.classis is None):
         self.classis = cls

   # Obtem lista de nomes defeatures ordenada
   def get_classis_name(self):
      if(self.classis is None):
         return None
      else:
         return sorted(self.classis.keys())      
      
   # Obtem lista de features ordenada
   def get_classis(self):
      if(self.classis is None):
         return None
      else:
         return [self.classis[k] for k in sorted(self.classis.keys())]   
   
   # Define as metricas
   def set_metricas(self,m=dmetricas):
      self.metricas = m

   # Obtem lista de nomes das metricas ordenada
   def get_matricas_name(self):
      if(self.metricas is None):
         return None
      else:
         return sorted(self.metricas.keys())  

   # Obtem lista de metricas ordenada
   def get_metricas(self):
      if(self.metricas is None):
         return None
      else:
         return [self.metricas[k] for k in sorted(self.metricas.keys())]   
      
   # Define o tamanho do treino em % 
   def set_ntreino(self, nt):
      self.ntreino = nt
   
   # Obtem o conjunto de dados de treino e teste
   def split_dados(self):
      print "Divide dataset entre treino e teste..."
      self.X_train,self.X_test,self.y_train,self.y_test = \
      train_test_split(self.vetsim,self.data.is_duplicate,train_size=self.ntreino,\
                       stratify=self.data.is_duplicate, random_state = self.seed)
      # Normaliza as features
      self.X_train.data = preprocessing.normalize(self.X_train, norm='l2')
      self.X_test.data  = preprocessing.normalize(self.X_test , norm='l2')
      

   # Define o tamanho do comite
   def set_tamcom(self, tc):
      self.tamcom = tc
   
   # Cria o comite 
   def set_comite(self, m, n):
      self.comite = Comite(m,n)
      
   # Seleciona os membros do comite: os classificadores com melhor acuracia
   def sel_membros(self):
      print "Seleciona membros para formar o comite..."
      membros = {}
      for nome, classe in sorted(self.classis.iteritems()):
         y_pred = classe.predict(self.X_train)
         score = self.metricas['accuracy_score'](self.y_train,y_pred)
         membros[nome]=score
      nm = sorted(membros, key=membros.get, reverse=True)[:self.tamcom]
      cm = [self.classis[n] for n in nm]
      self.set_comite(cm,n)  
      
   # Realiza o treino dos classificadores
   def treinamento(self):
      print "Treinamento dos classificadores..."
      if(self.lclassis is None):
         for nome, classe in self.classis.iteritems():   
            classe = classe.fit(self.X_train, self.y_train)
      self.sel_membros()   
      # Treina o comite
      self.treina_comite() 
         
   # Realiza o treino do comite conforme descrito no relatorio     
   def treina_comite(self):
      print "Treinamento do comite.."
      self.comite.fit(self.X_train, self.y_train)

   # Realiza o teste dos classificadores e gera os resultados
   def avaliacao(self):
      print "Gera os resultados.."
      self.resultados = {} # <classificador> => <dados metricas>
      self.resultados_confusion = {} # <classificador> => <matriz de confulsao>
      self.resultados_curve = {} # <classificador> => <curva de precisao revocação>
      self.resultados_report = {} # <classificador> => <report>      
      for nome, classe in sorted(self.classis.iteritems()):
         y_pred = classe.predict(self.X_test)
         y_proba = classe.predict_proba(self.X_test)         
         dados = {}
         self.resultados[nome] = dados
         for nmet, met in sorted(self.metricas.iteritems()):
            if nmet in ('log_loss'): 
               dados[nmet] = met(self.y_test, y_proba)
            elif nmet in ('precision_recall_curve'): 
               self.resultados_curve[nome] = met(self.y_test, y_proba[:,1])
            elif nmet in ('confusion_matrix'): 
               self.resultados_confusion[nome] = met(self.y_test,y_pred)
            elif nmet in ('classification_report'): 
               self.resultados_report[nome] = met(self.y_test,y_pred)
            else:   
               dados[nmet] = met(self.y_test,y_pred)
      # Realiza o teste e prepara resultados do comite
      y_pred = self.comite.predict_ajust(self.X_test)
      y_proba = self.comite.predict_proba_original(self.X_test)         
      dados = {}
      nome = 'COMITE'
      self.resultados[nome] = dados
      for nmet, met in sorted(self.metricas.iteritems()):
         if nmet in ('log_loss'): 
            dados[nmet] = met(self.y_test, y_proba)
         elif nmet in ('precision_recall_curve'):             
            self.resultados_curve[nome] = met(self.y_test, y_proba[:,1])
         elif nmet in ('confusion_matrix'): 
            self.resultados_confusion[nome] = met(self.y_test,y_pred)
         elif nmet in ('classification_report'): 
            self.resultados_report[nome] = met(self.y_test,y_pred)
         else:   
            dados[nmet] = met(self.y_test,y_pred)
      
   # Define o nome do arquivo de resultados
   def set_resname(self, rn = drname):
      self.resname = 'res_' + rn
      
   # Mostra e salva os resultados da avaliacao dos classificadores
   def mostra_resultados(self):
      if(self.resultados is None): self.avaliacao()
      cols= self.get_matricas_name()
      lvalues = [self.resultados.get(i) for i in sorted(self.resultados.keys())]
      rdf = pd.DataFrame(data=lvalues, index=sorted(self.get_classis_name()+['COMITE'])  , columns=cols)
      rdf.to_csv(self.resname, encoding='utf-8')
      print rdf

   # Mostra o resultado e gera a curva precisao-revocacao
   def mostra_resultados_curva(self):
      from itertools import cycle
      colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','blueviolet','chocolate','gold'])
      
      if(self.resultados is None): self.avaliacao()
      cols= ['Precision', 'Recall', 'Threshold']
      lvalues = [self.resultados_curve.get(i) for i in sorted(self.resultados_curve.keys())]
      lines = []
      labels = []
      plt.figure(figsize=(8, 10))  
      classes = sorted(self.get_classis_name()+['COMITE'])
      for l, color,c in zip(lvalues,colors, classes):
         (p, r, t) = l
         lin, = plt.plot(r, p, color=color, lw=2)
         lines.append(lin)
         labels.append(c)
      fig = plt.gcf()
      fig.subplots_adjust(bottom=0.35)   
      plt.ylim([0.0, 1.05])
      plt.xlim([0.0, 1.0])      
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title('Precision-Recall Curve')  
      plt.legend(lines, labels, loc=(0, 0), prop=dict(size=10))
      plt.savefig('curve_'+self.resname+'.png')   # save the figure to file
      
      rdf = pd.DataFrame(data=lvalues, index=sorted(self.get_classis_name()+['COMITE'])  , columns=cols)
      rdf.to_csv('curve_'+self.resname, index=False, encoding='utf-8',tupleize_cols=True)
      print rdf
      
   # Mostra o resultado e gera a tabela de matriz de confulsao
   def mostra_resultados_confusion(self):
      if(self.resultados is None): self.avaliacao()
      v= [ np.concatenate(a) for a in self.resultados_confusion.values()]
      rdf = pd.DataFrame(data=v, index=sorted(self.get_classis_name()+['COMITE']) )
      rdf.to_csv('confusion_'+self.resname, encoding='utf-8')
      print rdf
      
   # Mostra o resultado e gera o report das metricas
   def mostra_resultados_report(self):
      if(self.resultados is None): self.avaliacao()
      cols= ['Classification Report']
      lvalues = [self.resultados_report.get(i) for i in sorted(self.resultados_report.keys())]
      rdf = pd.DataFrame(data=lvalues, index=sorted(self.get_classis_name()+['COMITE']) , columns=cols)
      rdf.to_csv('report_'+self.resname, encoding='utf-8')
      print rdf
            
   # Executa uma rodada completa de experimento
   def rodada(self, s, nome, sw, ng, cn, pn, feat, cls, met, nt, tc, rn, fcn):
      self.set_seed(s)
      random.seed(s)
      self.set_name(nome)
      self.set_stopword(sw)
      self.set_ngram(ng)
      self.set_colname(cn)
      self.set_paresname(pn)
      self.load_colecao()
      self.set_features(feat)
      self.load_vetsim()
      self.load_classis(fcn)
      self.set_classis(cls)      
      self.set_metricas(met)
      self.set_ntreino(nt)
      self.split_dados()
      self.set_tamcom(tc)
      self.treinamento()
      self.save_classis(fcn)
      self.avaliacao()
      self.set_resname(rn)
      self.mostra_resultados()
      self.mostra_resultados_curva()
      self.mostra_resultados_confusion()
      self.mostra_resultados_report()      
      
            
# --------------------------------------------------------------------------------------------------------
# PRINCIPAL
# --------------------------------------------------------------------------------------------------------      
if __name__ == '__main__':
   print("***EXPERIMENTO OFICIAL QUORA***")   
   
   exp = Experimento()       
   sem = 0
   print "\nRodada semente {}: STOPWORD-TOKEN-FULL:\n".format(sem)
   exp.rodada(sem,exp.dname,True,False,'colecao-ws-tok.bin','par-ws-tok.bin',\
              exp.dfeatures,exp.dclassis, exp.dmetricas,0.6, 7, 'ws-tok-01.csv','classis-wstk-01')
   exp = None
   
   exp = Experimento()    
   sem = 20
   print "\nRodada semente {}: STOPWORD-TOKEN-FULL:\n".format(sem)
   exp.rodada(sem,exp.dname,True,False,'colecao-ws-tok.bin','par-ws-tok.bin',\
              exp.dfeatures,exp.dclassis, exp.dmetricas,0.6, 7, 'ws-tok-02.csv','classis-wstk-02')
   exp = None
   exp = Experimento()    
   sem = 45
   print "\nRodada semente {}: STOPWORD-TOKEN-FULL:\n".format(sem)
   exp.rodada(sem,exp.dname,True,False,'colecao-ws-tok.bin','par-ws-tok.bin',\
              exp.dfeatures,exp.dclassis, exp.dmetricas,0.6, 7, 'ws-tok-03.csv','classis-wstk-03')
   exp = None
   
   exp = Experimento()    
   sem = 0
   print "\nRodada semente {}: MINSTOPWORD-TOKEN-FULL:\n".format(sem)
   exp.rodada(sem,exp.dname,2,False,'colecao-miws-tok.bin','par-miws-tok.bin',\
              exp.dfeatures,exp.dclassis, exp.dmetricas,0.6, 7, 'miws-tok-01.csv','classis-miwstk-01')
   exp = None
   
   exp = Experimento()    
   sem = 20
   print "\nRodada semente {}: MINSTOPWORD-TOKEN-FULL:\n".format(sem)
   exp.rodada(sem,exp.dname,2,False,'colecao-miws-tok.bin','par-miws-tok.bin',\
              exp.dfeatures,exp.dclassis, exp.dmetricas,0.6, 7, 'miws-tok-02.csv','classis-miwstk-02')
   exp = None
   exp = Experimento()    
   sem = 45
   print "\nRodada semente {}: MINSTOPWORD-TOKEN-FULL:\n".format(sem)
   exp.rodada(sem,exp.dname,2,False,'colecao-miws-tok.bin','par-miws-tok.bin',\
              exp.dfeatures,exp.dclassis, exp.dmetricas,0.6, 7, 'miws-tok-03.csv','classis-miwstk-03')
   exp = None
   
   exp = Experimento()    
   sem = 0   
   print "\nRodada semente {}: NOSTOPWORD-TOKEN-FULL:\n".format(sem)
   exp.rodada(sem,exp.dname,False,False,'colecao-nows-tok.bin','par-nows-tok.bin',\
              exp.dfeatures,exp.dclassis, exp.dmetricas,0.6, 7, 'nows-tok-01.csv','classis-nows-01')
   exp = None
   exp = Experimento()    
   sem = 20
   print "\nRodada semente {}: NOSTOPWORD-TOKEN-FULL:\n".format(sem)
   exp.rodada(sem,exp.dname,False,False,'colecao-nows-tok.bin','par-nows-tok.bin',\
              exp.dfeatures,exp.dclassis, exp.dmetricas,0.6, 7, 'nows-tok-02.csv','classis-nows-02')
   exp = None
   exp = Experimento()    
   sem = 45
   print "\nRodada semente {}: NOSTOPWORD-TOKEN-FULL:\n".format(sem)
   exp.rodada(sem,exp.dname,False,False,'colecao-nows-tok.bin','par-nows-tok.bin',\
              exp.dfeatures,exp.dclassis, exp.dmetricas,0.6, 7, 'nows-tok-03.csv','classis-nows-03')
   exp = None   
   
   print "\nEXPERIMENTO FINALIZADO!!!"      
   