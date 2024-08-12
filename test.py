import numpy as np
import tensorflow as tf
from nnf import Model, CrossEntropyLoss, SGD, Linear, ReLU, Softmax


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]


model = Model()
model.add_layer(Linear(784, 128))
model.add_layer(ReLU())
model.add_layer(Linear(128, 10))
model.add_layer(Softmax())

loss = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)
model.compile(loss, optimizer)

model.train(x_train, y_train, epochs=20, batch_size=64)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

predictions = model.predict(x_test)

predicted_classes = np.argmax(predictions, axis=1)
print("Predicted classes for the first 10 test samples:", predicted_classes[:10])

