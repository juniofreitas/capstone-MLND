# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import  log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

from estruturas.infinityarray import InfinityArray
from estruturas.par import Par
from estruturas.colecao import Colecao
from classificador.comite import Comite

from funcoes.soundex import Soundex
from funcoes.jaccard import Jaccard
from funcoes.fullsimilarity import FullSimilarity
from funcoes.bagsim import BagSim
from funcoes.soft_tfidf import softTFIDF
#from funcoes.levenstein import Levenstein

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1500)
###################################################################################################
# TREINO
###################################################################################################
path_data_treino = os.sep.join([os.path.dirname(os.path.realpath(__file__)), 'datasets',  'train.csv'])
path_data_teste  = os.sep.join([os.path.dirname(os.path.realpath(__file__)), 'datasets',  'test.csv'])

print "Carregando dados de TREINO...\n"
dados_treino = pd.read_csv(path_data_treino)
dados_treino = dados_treino.fillna("null")

print "Gerando Lista de Documentos de TREINO...\n"
txt1 = dados_treino[['qid1','question1']].rename(columns = {'qid1':'qid', 'question1':'question'})
txt2 = dados_treino[['qid2','question2']].rename(columns = {'qid2':'qid', 'question2':'question'})
txts = txt1.append(txt2).sort_values('qid').drop_duplicates()
ltxt = list(txts['question'])
del txt1, txt2

print "Processando Colecao de Documentos de TREINO..."
#colecao = Colecao(ltxt,'coltreino')
colecao = Colecao(ltxt,'coltreino')
print 'Tamanho Colecao Documentos Treino={}'.format(colecao.getTotalDoc())
print 'Tamanho Vocabulario={}'.format(colecao.size())
print 'Total Tokens={}\n'.format(colecao.getTotalTok())

print "Processando PARES DE TREINO.."
funcs = [Jaccard(), softTFIDF(func=Soundex()) , softTFIDF(), FullSimilarity(), BagSim()]
cols = ['Jaccard', 'softSoundex','softTFIDF', 'fullSim', 'bagSim']
pares = dados_treino[['id','qid1','qid2','is_duplicate']]
target = dados_treino.is_duplicate

print "Gerando Lista de Vetores de Similaridade de TREINO..."
lpares = InfinityArray('pares_treino.bin','pares_treino.idx')
lpares.open()
if(not lpares.ready()):
   for index, row in pares.iterrows():
      par = Par(int(row['qid1'])-1, int(row['qid2'])-1)
      par.setLabel(row['is_duplicate'])
      par.process(colecao,funcs)
      lpares.append(par)   
      if(index%1000==0):
         print "\rProcessados {} pares: ({},{})={}".format(index,par.getLabel(),par.getWeight(),par.getVetSim())
      
print 'Quantidade Pares de Treino: {}\n'.format(lpares.size)   
lpares.end()
del pares
del dados_treino
del ltxt[:]
del colecao

print "Treinando classificador..."
lpares.start()

vetsim = pd.DataFrame(data=np.array([lpares.getx(i).getVetSim() for i in xrange(lpares.size)], \
                      dtype = float), columns=list(cols))
print "Vetores de Similaridae de Texto preparados!"
lpares.end()

# Treino em si
print "Treinando..."
names = [
      "Nearest Neighbors", 
      "Decision Tree", 
      "Random Forest", 
      "Neural Net", 
      "Naive Bayes", 
      "SGD",
      "GradientBoosting"
      ]
classifiers = [
    KNeighborsClassifier(algorithm='kd_tree'),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    GaussianNB(),
    SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=5, random_state=0),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    ]

X_train, X_test, y_train, y_test = train_test_split(vetsim, target, test_size=0.33, random_state=84)

metrics = []
scores = []
f1s = []
for nome, classe in zip(names,classifiers):   
   #if hasattr(classe, "predict_proba"):
   print nome,"..."
   classe = classe.fit(X_train, y_train)
   y_proba = classe.predict_proba(X_test)
   score = classe.score(X_test,y_test)
   log_loss_score = log_loss(y_test, y_proba)
   f1 = f1_score(y_test,classe.predict(X_test))
   metrics.append([nome,log_loss_score])
   scores.append([nome,score])   
   f1s.append([nome,f1])

print "OK!\n"
#ll_mean = np.mean(metrics[:,1])
#print 'Log loss mean=',ll_mean

print "Treinamento do Comite..."
#eclf = VotingClassifier(estimators=[(names[0], classifiers[0]), (names[1], classifiers[1]), (names[2], classifiers[2]), 
#                                    (names[3], classifiers[3]),(names[4], classifiers[4]),(names[5], classifiers[5]),(names[6], classifiers[6])],voting='hard')
#eclf.fit(X_train, y_train)
comite = Comite(classifiers,names)
print "Fit Comite..."
comite.fit(X_train, y_train)
print "Log loss do Comite..."
y_proba_com = comite.predict_proba_original(X_test)
log_loss_score_com = log_loss(y_test, y_proba_com)
#y_proba_e = eclf.predict_proba(X_test)
#log_loss_e = log_loss(y_test, y_proba_e)
metrics.append(['COMITE',log_loss_score_com])
#metrics.append(['ESSEMBLE',log_loss_e])   
print "Accuracy do Comite..."
score_com = accuracy_score(y_test, comite.predict(X_test))
ascore_com = accuracy_score(y_test, comite.predict_ajust(X_test))
print "F1 do Comite..."
f1_com = f1_score(y_test,comite.predict(X_test))
af1_com = f1_score(y_test,comite.predict_ajust(X_test))
#score_e = accuracy_score(y_test, eclf.predict(X_test))
#f1_e = f1_score(y_test,eclf.predict(X_test))
scores.append(['COMITE',score_com])
scores.append(['ACOMITE',ascore_com])
f1s.append(['COMITE',f1_com])
f1s.append(['ACOMITE',af1_com])
#scores.append(['ESSEMBLE',score_e])
#f1s.append(['ESSEMBLE',f1_e])


print "Resultados:"
result = pd.DataFrame(metrics, columns=['Nome','Logloss'])
sresult = pd.DataFrame(scores, columns=['Nome','Score'])
sf1esult = pd.DataFrame(f1s, columns=['Nome','F1'])
print result,'\n'
print sresult,'\n'
print sf1esult,'\n'
print comite

print "Calcula Metricas de Desempenho:"
print "Em contrução...\n"



###################################################################################################
# TESTE
###################################################################################################
'''
# Carrega dataset de teste
dados_teste = pd.read_csv(path_data_teste)
dados_teste = dados_teste.fillna('empty')
# Obtendo lista de questoes a partir dos pares de questoes
txt1 = dados_teste[['test_id','question1']].rename(columns = {'question1':'question'})
txt2 = dados_teste[['test_id','question2']].rename(columns = {'question2':'question'})
txts = txt1.append(txt2).sort_values('test_id')#.drop_duplicates()
ltxt = list(txts['question'])
del txt1, txt2

# Limpeza de dados
# ....

print "Processando TESTE...\n"
coltest = Colecao(ltxt,'coltest')
print 'Tamanho colecao teste: {}'.format(coltest.getTotalDoc())

print "Carregando PARES DE TESTE...\n"
parestest = dados_teste[['test_id']]
test_ids = dados_teste['test_id']

lparestest = InfinityArray('pares_test.bin','pares_test.idx',1000)
lparestest.open()
if(not lparestest.ready()):
   j=0
   for index, row in parestest.iterrows():
      lparestest.append(Par(index+j, index+j+1))   
      j += 1
print 'Tamanho pares: {}'.format(lparestest.size)   
lparestest.end()

del parestest
del dados_teste
del ltxt[:]

print "\nProcessando PARES DE TESTE...\n"
lparestest.start()
lvst = InfinityArray('vetsim_teste.bin','vetsim_teste.idx',10)
lvst.open()
#####################################
if(not lvst.ready()):
   for i in xrange(lparestest.size):
      par = lparestest.getx(i)
      par.process(coltest,funcs)
      lvst.append(par.getVetSim())
      if(i%10000==0):
         print "Processados {} pares: {}".format(i,par.getVetSim())
lvst.end()        
   
print "Lista de Vetores de Similaridade de Treino OK!"
coltest.documentos.end()
lparestest.end()

lvst.start()
print lvst
for i in xrange(100):
   print lvst.getx(i)

print "\nTestando classificador...\n"
vetsimtest = pd.DataFrame(data=np.array([lvst.getx(i) for i in xrange(lvst.size)], dtype = float), \
                          columns=list(cols)) 
del coltest

print "Resultados:"
X_test = vetsimtest
for nome, classe in zip(names,classifiers):   
   y_proba = classe.predict_proba(X_test)
   submission = pd.DataFrame(test_ids)
   prediction_set = []
   for i in xrange(len(y_proba)):
      prediction_set.append(y_proba[i][1])
   prediction_set = pd.DataFrame(prediction_set, columns=['is_duplicate'])
   submission = pd.concat([submission, prediction_set], axis=1)   
   submission.to_csv(nome +"_submission_jsf.csv", index=False)
   del submission
   del prediction_set[:]
   del y_proba
'''  

