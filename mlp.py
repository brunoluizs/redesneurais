import numpy as np
import pandas as pd

class PerceptronLayer:
	def __init__(self, num_inputs, num_neurons, lr=0.1):
		self.w = np.random.randn(num_inputs, num_neurons) * 0.1
		self.bias = np.zeros(num_neurons)
		self.lr = lr

	def forward(self, inputs):
		self.inputs = inputs
		self.total = np.dot(inputs, self.w) + self.bias
		self.output = 1 / (1 + np.exp(-self.total))
		return self.output

	def backward(self, grad):
		delta = grad * self.output * (1 - self.output)

		grad_w = np.outer(self.inputs, delta)
		grad_bias = delta

		grad_input = np.dot(delta, self.w.T)

		self.w += self.lr * grad_w
		self.bias += self.lr * grad_bias

		return grad_input


class MLP:
	def __init__(self, layers, lr=0.1):
		self.layers = [
			PerceptronLayer(layers[i], layers[i+1], lr=lr)
			for i in range(len(layers) - 1)
		]

	def forward(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def predict(self, x):
		return np.round(self.forward(x))

	def train(self, inputs, targets, epochs=500):
		for epoch in range(epochs):
			total_error = 0.0
			for x, target in zip(inputs, targets):
				output = self.forward(np.array(x))
				error = target - output
				total_error += np.sum(np.abs(error))

				grad = error
				for layer in reversed(self.layers):
					grad = layer.backward(grad)

			if epoch % 100 == 0:
				print(f'Época {epoch} - Erro total: {total_error}')

			



def main():
	mlp = MLP(layers=[2, 1, 1], lr=0.1)

	X = [[0, 0], [0, 1], [1, 0], [1, 1]]
	y = [0, 1, 1, 0]

	mlp.train(X, y, epochs=20000)

	for x in X:
		previsao = mlp.predict(x)
		print(f'Previsao para {x}: {previsao}')


if __name__ == "__main__":
	main()
