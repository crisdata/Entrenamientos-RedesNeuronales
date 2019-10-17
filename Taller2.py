
from tkinter import *
from tkinter import ttk
from random import choice
from numpy import array, dot, random, exp
import matplotlib.pyplot as plt
from RedNeuronalSimple import RedNeuronalSimple
from RedNeuronalMulticapa import RedNeuronalMulticapa



class Perceptron():
	def __init__(self):
		self.ventana        = Tk()
		self.entrada1       = StringVar()
		self.entrada2       = StringVar()
		self.entrada3       = StringVar()
		self.entrada4       = StringVar()
		self.salida1        = StringVar()
		self.salida2        = StringVar()
		self.salida3        = StringVar()
		self.salida4        = StringVar()
		self.sesgoB         = DoubleVar()
		self.entrenamiento  = IntVar()
		self.elegirActvacion= StringVar()


		self.ventana.title("Taller 2")
		self.ventana.geometry("500x500")

		self.notebook = ttk.Notebook(self.ventana)
		self.notebook.pack(fill='both',expand='yes')
		self.pest0 = ttk.Frame(self.notebook)
		self.pest1 = ttk.Frame(self.notebook)
		self.pest2 = ttk.Frame(self.notebook)

		# Arreglo de entrada
		self.entradas=[]
		self.salidas=[]


	#Funcion de Activacion
	def funcionActivate(self, x):
		return 0 if x < 0 else 1


	#Enrada de Datos
	def nombreCajitas(self):
		self.notebook.add(self.pest0,text="Perceptron")
		Label(self.pest0,text="Perceptron").place(x=230,y=5)

		#Set de Entrenamiento
		#Entradas
		Entry(self.pest0,textvariable=self.entrada1).place(x=100,y=30)
		Entry(self.pest0,textvariable=self.entrada2).place(x=100,y=50)
		Entry(self.pest0,textvariable=self.entrada3).place(x=100,y=70)
		Entry(self.pest0,textvariable=self.entrada4).place(x=100,y=90)

		#Salidas
		Entry(self.pest0,textvariable=self.salida1).place(x=250,y=30)
		Entry(self.pest0,textvariable=self.salida2).place(x=250,y=50)
		Entry(self.pest0,textvariable=self.salida3).place(x=250,y=70)
		Entry(self.pest0,textvariable=self.salida4).place(x=250,y=90)

		Button(self.pest0,text="Ejecutar",command=self.entrenarPerceptron).place(x=240,y=220)
		Button(self.pest0,text="Ejecutar2",command=self.entrenarSimple).place(x=280,y=220)
		Button(self.pest0,text="Ejecutar3",command=self.entrenarMulti).place(x=320,y=220)

		# Sesgo de bahies
		Entry(self.pest0,textvariable=self.sesgoB).place(x=250,y=130)

		# Entrenamientos
		Entry(self.pest0,textvariable=self.entrenamiento).place(x=250,y=150)

		# Funciones de activacion
		ttk.Combobox(self.pest0,values=("Tangente","Sigmoide"),textvariable=self.elegirActvacion).place(x=250,y=170)


		self.ventana.mainloop()


	# Perceptron (Entrada y Salida)
	def entradaPerceptron(self):
		self.entradas = []
		self.entradas.append([
							(array([int(x) for x in self.entrada1.get().split(',')]),int(self.salida1.get())),
							(array([int(x) for x in self.entrada2.get().split(',')]),int(self.salida2.get())),
							(array([int(x) for x in self.entrada3.get().split(',')]),int(self.salida3.get())),
							(array([int(x) for x in self.entrada4.get().split(',')]),int(self.salida4.get()))
							])
		return self.entradas[0]
	# --------------------------------------

	# Red Neuronal Simple (Entrada y Salida)
	def entradaSimple(self):
		self.entradas = []
		self.entradas.append([
							[int(x) for x in self.entrada1.get().split(',')],
							[int(x) for x in self.entrada2.get().split(',')],
							[int(x) for x in self.entrada3.get().split(',')],
							[int(x) for x in self.entrada4.get().split(',')]
							])
		return array(self.entradas[0])

	def salidaSimple(self):
		self.salidas = []
		self.salidas.append([
							[int(self.salida1.get()),int(self.salida2.get()),int(self.salida3.get()),int(self.salida4.get())],
							])
		return array(self.salidas[0])
	# -------------------------------------

	# Red Neuronal Multicapa (Entrada y Salida)
	def entradaMulti(self):
		self.entradas = []
		self.entradas.append([
							[int(x) for x in self.entrada1.get().split(',')],
							[int(x) for x in self.entrada2.get().split(',')],
							[int(x) for x in self.entrada3.get().split(',')],
							[int(x) for x in self.entrada4.get().split(',')]
							])
		return array(self.entradas[0])

	def salidaMulti(self):
		self.salidas = []
		self.salidas.append([
							[int(x) for x in self.salida1.get().split(',')],
							[int(x) for x in self.salida2.get().split(',')],
							[int(x) for x in self.salida3.get().split(',')],
							[int(x) for x in self.salida4.get().split(',')]
							])
		return array(self.salidas[0])


	# Sesgo de bahies
	def entraSesgo(self):
		self.sesgoB.get()
		return self.sesgoB.get()
	#-------------------------------------


	# Epocas
	def entraEpocas(self):
		self.entrenamiento.get()
		return self.entrenamiento.get()
	#-------------------------------------


	#Entrenamiento de Neuronas
	def entrenarPerceptron(self):
		w = random.rand(3)
		errores = []
		esperados = []

		# Entrenamiento
		for i in range(self.entraEpocas()):
		    x, esperado = choice(self.entradaPerceptron())
		    resultado = dot(w, x)
		    esperados.append(esperado)
		    error = esperado - self.funcionActivate(resultado)
		    errores.append(error)
		    #Ajuste
		    w += self.entraSesgo() * error * x

		for x, _ in self.entradaPerceptron():
		    resultado = dot(w, x)
		    print("{}: {} -> {}".format(x[:3], resultado, self.funcionActivate(resultado)))

		plt.plot(errores,'-',color='red')
		plt.plot(esperados,'*', color='green')
		plt.show()


	def entrenarSimple(self):
		ob = RedNeuronalSimple(self.elegirActvacion.get())
		errores = ob.entrenamiento(self.entradaSimple(), self.salidaSimple().T, self.entraEpocas())
		ob.imprimir()
		# print(self.entradaSimple())
		# print(self.salidaSimple())
		print(len(errores))
		plt.plot(errores,'-',color='red')
		plt.show()

	def entrenarMulti(self):
		ob = RedNeuronalMulticapa([2,3,2], self.elegirActvacion.get())
		ob.ajuste(self.entradaMulti(), self.salidaMulti(), self.entraSesgo(), self.entraEpocas())

		deltas = ob.obtener_deltas()
		valores = []
		index = 0
		for arreglo in deltas:
		    valores.append(arreglo[1][0] + arreglo[1][1])
		    index = index + 1

		plt.plot(range(len(valores)), valores, color='b')
		plt.ylim([0, 1])
		plt.ylabel('Costo')
		plt.xlabel('Epocas')
		plt.tight_layout()
		plt.show()

	# --------------------------------------



problema1 = Perceptron()
problema1.nombreCajitas()
