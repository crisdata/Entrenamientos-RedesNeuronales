#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import exp, array, random, dot, tanh, sum


# In[25]:


class RedNeuronalSimple():
    def __init__(self, activacion):
        self.pesos_signaticos = 2 * random.random((3,1)) - 1

        if activacion == 'Sigmoide':
            self.activacion = self.sigmoide
            self.activacion_prima = self.sigmoide_derivado
        elif activacion == 'Tangente':
            self.activacion = self.tangente
            self.activacion_prima = self.tangente_derivada

    def sigmoide(self, x):
        return 1 / (1 + exp(-x))

    def sigmoide_derivado(self, x):
        return x * (1 - x)

    def tangente(self, x):
        return tanh(x)

    def tangente_derivada(self, x):
        return 1 - x**2

    def entrenamiento(self,entradas,salidas,numero_iteraciones):
        errores = []
        for i in range(numero_iteraciones):
            salida = self.pensar(entradas)
            error = salidas - salida
            errores.append(abs(sum(error)))
            ajuste = dot(entradas.T, error * self.activacion_prima(salida))
            self.pesos_signaticos += ajuste

        return errores

    def pensar(self,entrada):
        return self.activacion(dot(entrada, self.pesos_signaticos))

    def imprimir(self):
        print(self.pesos_signaticos)
        print(self.pensar(array([1,0,0])))

# if __name__ == '__main__':
#     red_neuronal = RedNeuronal()
#     entradas = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
#     salidas = array([[0,1,1,0]]).T
#     red_neuronal.entrenamiento(entradas,salidas,1000)
#     print(red_neuronal.pesos_signaticos)
#     print(red_neuronal.pensar(array([1,0,0])))
