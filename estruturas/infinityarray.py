# -*- coding: utf-8 -*-
"""
Created on Sat May 27 15:29:09 2017
@author: junio
"""

import cPickle as pickle 
import os

class InfinityArray():
   '''
    Esta classe implementa uma lista que armazenada parte de seu elementos 
    em memoria secundaria e parte em memória principal. O que fica em memória
    principal deve ser o que estiver sendo usado no momento. Todo conteúdo
    da lista é salvo em arquivo e poderá ser recuperado para ser utilizado em
    outro momento.
    [1]->[2]->[3]->[4]->[[5]->[6]->[7]]->[8]->[9]->[10]->[11]->[12]
    No exemplo acima, tem-se uma lista contento números de 1 a 12. Somente os 
    elementos 5, 6 e 7 estão em memória, os restantes ficam em disco. A parte
    da lista que fica em memória é chamada janela (ou cursor) e a medidade em que
    a janela percorre a lista os elementos vão sendo carregados para memória.
    O tamanho da janela indicará o número de elementos da lista que ficarão em 
    memória. 
   '''
   
   DEF_WIN = 2048 # tamanho padrao da janela
   DEF_FILE = '.infinityarray.bin' # nome do arquivo padrao que guarda os dados da lista
   DEF_INDEX = '.ind' # nome do arquivo de indice da lista
   OFF_SET = 256 # tamanho de sobra do bloco de dados da lista
   DEF_PADD = 32 # tamanho de dados que completa um bloco
   DEF_LENGHT = len(pickle.dumps('1'.zfill(DEF_PADD), pickle.HIGHEST_PROTOCOL)) # tamanho do bloco 
   
   def __init__(self, f=DEF_FILE, x=DEF_INDEX,  win=DEF_WIN):
      self.file=f # arquivo que armazena a lista em disco
      self.findex=x # arquivo que armazena o indice
      self.win=win # janela de elementos da lista que ficam em memória
      self.array=None # elemento da lista
      self.handle=None # Tratador do arquivo em memoria
      self.hfx=None # Tratador do arquivo de indice
      self.size=0 # tamanho da lista
      self.nwin=0 # numero de janelas da lista
      self.pos=-1 # posicao atual na lista 
      self.idx=0 # indice da janela, entre 0 .. win
      self.edit=False # verificor se houve edicao na lista em memoria
      
   
   def __str__(self):
      return 'Infinity Array<{}>.size={}.win={}'.format(self.file,self.size,self.win)
      
                               
   def vernamefile(self,f):
      if(self.file=="")and(not isinstance(f,  (basestring, str))):
         raise TypeError('Argument is expected to be a string or is null')

 
   def verfile(self):
      if(self.handle.closed):
         raise TypeError('File Handle is closed')


   def verfindex(self):
      if(self.hfx.closed):
         raise TypeError('File Handle is closed')         

  
   def verindex(self, index):
      if(index<0 or index>=self.size):
         raise TypeError('Invalide index {}. Size Array is {}'.format(index,self.size))

         
   def verindexwin(self, index):
      if(index<0 or index>=self.win):
         raise TypeError('Invalide window index {}'.format(index))

   
   def verwin(self, win):
      if(win<1):
         raise TypeError('Window={} biggest Size={}'.format(win,self.size))

   
   def prepFile(self, mode):
      if(mode not in ('rb','wb','ab','rb+','wb+','ab+')):
         raise TypeError('File mode must be rb, wb or rb+')
      try:   
         if(self.handle and not self.handle.closed): self.handle.close()
         self.handle = open(self.file,mode)   
      except IOError:
         raise TypeError('Could not open file:{}'.format(self.file))


   def prepFindex(self, mode):
      if(mode not in ('rb','wb','ab','rb+','wb+','ab+')):
         raise TypeError('File mode must be rb, wb or rb+')
      try:   
         if(self.hfx and not self.hfx.closed): self.hfx.close()
         self.hfx = open(self.findex,mode) 
      except IOError:
         raise TypeError('Could not open file:{}'.format(self.file))         


   def dopadd(self,v, t=DEF_PADD):
      s = str(v)
      return s.zfill(t)


   def unpadd(self,s):
      return int(s)

   
   def createheader(self):
      if(not self.handle or self.handle.closed or self.handle.mode != 'wb'):
         self.prepFile('wb')
      self.handle.seek(0)
      a = str(self.size)
      b = str(self.win)
      head = (a.zfill(self.OFF_SET), b.zfill(self.OFF_SET))
      pickle.dump(head,self.handle,pickle.HIGHEST_PROTOCOL)   

      
   def saveheader(self):
      if(not self.handle or self.handle.closed or self.handle.mode != 'rb+'):
         self.prepFile('rb+')
      self.handle.seek(0)   
      a = str(self.size)
      b = str(self.win)
      head = (a.zfill(self.OFF_SET), b.zfill(self.OFF_SET))      
      pickle.dump(head,self.handle,pickle.HIGHEST_PROTOCOL)

      
   def readheader(self):
      if(not self.handle or self.handle.closed or self.handle.mode != 'rb'):
         self.prepFile('rb')
      head = pickle.load(self.handle)
      a,b = head
      self.size,self.win = int(a), int(b)

      
   def jumpheader(self):
      if(self.handle is None or self.handle.closed or self.handle.mode != 'rb'):
         self.prepFile('rb')
      pickle.load(self.handle)
      
      
   def open(self):
      if(os.path.isfile(self.file)):
         self.start()
      else:
         self.new()             
	
      
   def ready(self):
      return self.pos!=-1
   

   def new(self, f=None):
      if(f): self.file = f
      self.vernamefile(self.file)
      if(self.array is not None):
         del self.array[:]
      self.array = [None]*self.win
      self.size = self.nwin = self.idx = 0
      self.pos = -1
      self.nwin = 1
      self.edit = False
      self.createheader()     
      
      
   def flush(self):
      self.verfile()
      for o in self.array:         
         if(o is not None):
            ipos = self.handle.tell()
            self.appendidx(ipos)
            pickle.dump(o,self.handle,pickle.HIGHEST_PROTOCOL)
      del self.array[:]
      self.array = [None]*self.win      
   
   
   def append(self, o):
      self.verfile()
      self.idx += 1
      self.size += 1
      self.pos += 1      
      if(self.idx==self.win):
         self.flush()
         self.nwin += 1
         self.idx = 0
      self.array[self.idx] = o
      self.edit = True
      return True      
      
   
   def appendidx(self,i):
      if(self.hfx is None or self.hfx.mode != 'wb'):
         self.prepFindex('wb')
      pickle.dump(self.dopadd(i),self.hfx, pickle.HIGHEST_PROTOCOL)
   
   
   def seek(self, nw):
      self.verfile()
      if(self.nwin == nw):
         return
      if(nw == 1):
         self.handle.seek(0)         
         self.pos = 0
         self.jumpheader()
      else:
         i = self.pos if(nw > self.nwin) else 0
         while( i<(self.win*(nw-1))-1):
            pickle.load(self.handle)
            i += 1
      self.nwin = nw
      self.pos = self.win*(nw)-1
      
      
   def load(self, nw):
      self.seek(nw)
      i = 0
      lim = self.win-1 if(nw<(self.size/self.win)+1) else (self.size-1)%self.win
      while i <= lim:         
         self.array[i] = pickle.load(self.handle)
         i += 1

         
   def loadx(self, nw):      
      self.pos = self.win*(nw-1)
      self.handle.seek(self.getIndex(self.pos))      
      i = 0
      lim = self.win-1 if(nw<(self.size/self.win)+1) else (self.size-1)%self.win
      while i <= lim:         
         self.array[i] = pickle.load(self.handle)
         i += 1

   
   def start(self, f=None):
      if(f): self.file = f
      self.vernamefile(self.file)
      self.readheader()
      if(self.array is not None):
         del self.array[:]
      self.array = [None]*self.win
      self.load(1)
      self.nwin = 1
      self.pos = self.win-1
      self.edit = False
      return True   
      
   
   def getIndex(self, i):
      self.verindex(i)
      if(self.hfx is None or self.hfx.mode != 'rb'):
         self.prepFindex('rb')      
      ipos = i*self.DEF_LENGHT
      self.hfx.seek(ipos)
      return long(self.unpadd(pickle.load(self.hfx)))
   
   
   def get(self, i):
      self.verindex(i)
      if(self.handle.mode != 'rb'):
         self.prepFile('rb') 
         self.jumpheader()
      self.idx = i % self.win
      nw = (i / self.win)+1   
      if(nw != self.nwin):
         self.load(nw)
      return self.array[self.idx]


   def getx(self, i):
      self.verindex(i)
      if(self.handle.mode != 'rb'):
         self.prepFile('rb') 
         self.jumpheader()
      self.idx = i % self.win
      nw = (i / self.win)+1               
      if(nw != self.nwin):
         self.loadx(nw)
         self.nwin = nw
      return self.array[self.idx]
      
   
   def end(self):
      self.verfile()
      if(self.edit):
         self.flush()
      if(self.hfx is not None): 
         self.hfx.close() 
         self.hfx = None
      self.saveheader()   
      if(self.handle is not None): 
         self.handle.close()
         self.handle = None
      self.pos=-1
      del self.array[:]