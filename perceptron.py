import numpy as np
import pandas as pd

class Perceptron:
	def __init__(self, num_inputs, lr=0.1):
		self.w = np.random.randn(num_inputs) * 0.1
		self.lr = lr
		self.bias = 0.0

	def __str__(self):
		return f'Pesos: {self.w}'

	def forward(self, inputs):
		total = np.dot(inputs, self.w) + self.bias
		output = 1 / (1 + np.exp(-total))
		return np.round(output)

	def train(self, inputs, targets, epochs=20):
		for epoch in range(epochs):
			total_error = 0.0
			
			for x, target in zip(inputs, targets):
				prediction = self.forward(x)
				error = target - prediction
				total_error += abs(error)
				self.w += self.lr * error * np.array(x)
				self.bias += self.lr * error

			if epoch % 100 == 0:
				print(f'Época {epoch} - Erro total: {total_error}')

			if total_error == 0.0:
				print(f'Convergiu na época {epoch}')
				break

def main():
	print('Perceptron')
	p = Perceptron(num_inputs=2, lr=0.01)

	df = pd.read_csv('bmi.csv')
	

	X = np.array(df[['Height', 'Weight']])
	y = np.array(df[['Index']].replace([0, 1, 2], 0).replace([3, 4, 5], 1))


	p.train(X, y, epochs=5000)

	acerto, erro = 0, 0
	idx = 0
	for entrada in X:
		previsao = p.forward(entrada)
		if previsao == y[idx]:
			acerto += 1
		else:
			erro += 1

		idx += 1

	print(f'Acertos: {acerto}, Erros: {erro}')
	print(f'Acurácia: {acerto/500}')

	# entrada = [0, 1]
	# print(f'Previsão de {entrada}: {p.forward(entrada)}')

if __name__ == '__main__':
	main()
