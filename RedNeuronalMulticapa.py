#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:





# In[18]:


class RedNeuronalMulticapa():
    def __init__(self, capas, activacion='Tangente'):
        if activacion == 'Sigmoide':
            self.activacion = self.sigmoide
            self.activacion_prima = self.sigmoide_derivado
        elif activacion == 'Tangente':
            self.activacion = self.tangente
            self.activacion_prima = self.tangente_derivada
            
        #Iniciarlizar pesos
        self.pesos = []
        self.deltas = []
        # capas = [2,3,2] randon entre 1, -1
        for i in range(1, len(capas) -1):
            r = 2 * np.random.random((capas[i-1] + 1, capas[i] + 1)) -1
            self.pesos.append(r)
        
        #asignar aleatorios a la capa de salida
        r = 2 * np.random.random((capas[i] + 1, capas[i + 1])) - 1
        self.pesos.append(r)

    def sigmoide(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoide_derivado(self, x):
        return self.sigmoide(x) * (1 - self.sigmoide(x))

    def tangente(self, x):
        return np.tanh(x)

    def tangente_derivada(self, x):
        return 1 - x**2
            
    def ajuste(self, X, y, factor_aprendizaje = 0.2, epocas = 1000000):
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis = 1)
        
        for k in range(epocas):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            
            for l in range(len(self.pesos)):
                dot_value = np.dot(a[l], self.pesos[l])
                activacion = self.activacion(dot_value)
                a.append(activacion)
                
            #Calculo la diferencia entre la capa de salida y el valor obtenido
            error = y[i] - a[-1]
            deltas = [error * self.activacion_prima(a[-1])]
            
            #Empezamos en la segunda capa hasta la ultima
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.pesos[l].T) * self.activacion_prima(a[l]))
            self.deltas.append(deltas)
            
            #invertir
            deltas.reverse()
            
            #Backpropagation
            for i in range(len(self.pesos)):
                capa = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.pesos[i] += factor_aprendizaje * capa.T.dot(delta)
            
            if k % 10000 == 0: print('epocas:', k)
                
    def predecir(self, x):
        unos = np.atleast_2d(np.ones(x.shape[0]))
        a = np.concatenate((np.ones(1).T, np.array(x)), axis = 0)
        for l in range(0, len(self.pesos)):
            a = self.activacion(np.dot(a, self.pesos[l]))
        return a
    
    def imprimir_pesos(self):
        print("Listado de Pesos de Conexiones")
        for i in range(len(self.pesos)):
            print(self.pesos[i])
    
    def obtener_deltas(self):
        return self.deltas    


# In[19]:


# nn = RedNeuronalMulticapa([2,3,2], activacion='tangente')
# X = np.array([[0, 0], # sin obstaculos
#              [0,1], # sin obstaculos
#              [0, -1], # sin obstaculos
#              [0.5, 1], #obstaculo detectado a la derecha
#              [0.5, -1], #obstaculo a izquierdad
#              [1, 1], # demasiado cerca a la derecha
#              [1, -1]]) # demasiado cerca a la izquierda

# y = np.array([[0, 1], #avanzar
#               [0, 1], #avanzar
#               [0, 1], #avanzar
#               [-1, 1], #giro izquierda
#               [1, 1], #giro derecha
#               [0, -1], #retroceder
#               [0, -1],]) #retroceder
# nn.ajuste(X, y, factor_aprendizaje = 0.03, epocas = 150000)

# index = 0
# for e in X:
#     print("X: ", e, "y: ", y[index], "Red: ", nn.predecir(e))
#     index = index + 1


# # In[20]:


# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

