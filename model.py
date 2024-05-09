import tensorflow as tf
import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

n_inputs = 28 * 28

# Normalization of input images
data_from_dataset = np.array(x_train)
data_from_dataset = data_from_dataset.reshape(data_from_dataset.shape[0], -1)
print(data_from_dataset.shape, data_from_dataset[0])
data_from_dataset = data_from_dataset.astype('float32') / 255.

# Encoder
n_hidden_1 = 300
n_hidden_2 = 5

# Decoder
n_hidden_3 = n_hidden_1
n_outputs = n_inputs

model = keras.models.Sequential()

# Make the mat mul
model.add(keras.layers.Dense(n_hidden_1, input_dim=784, activation='elu', name='hidden_layer'))
model.add(keras.layers.Dense(n_hidden_2, activation='elu', name='coder_end'))
model.add(keras.layers.Dense(n_hidden_3, activation='elu'))
model.add(keras.layers.Dense(n_outputs, activation=None))

# Optimize and compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Structure of model
print(model.summary())

BATCH_SIZE = 125
EPOCHS = 50
# Number of batches:  length dataset / batch size
n_batches = data_from_dataset.shape[0] // BATCH_SIZE
print(n_batches)

history = model.fit(data_from_dataset, data_from_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

model.save_weights('model.weights.h5')
keras.saving.save_model(model, 'model.keras')

x_test_resized = x_test.reshape(x_test.shape[0], -1)
x_test_resized = x_test_resized.astype('float32') / 255.

n_rec = 389

plt.imshow(x_test[n_rec], cmap='gray')
plt.show()

x = x_test_resized[n_rec]
x = np.expand_dims(x, axis=0)
print(x.shape)

prediction = model.predict(x)

plt.imshow(prediction[0].reshape([28, 28]), cmap='gray')
plt.show()

loss = history.history['loss']

epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

print(model)
print(model.weights)

# model(keras.Input((784, )))
layer_inputs = model.get_layer(name='hidden_layer').input
layer_outputs = model.get_layer(name='coder_end').output
coder_model = keras.models.Model(inputs=layer_inputs, outputs=layer_outputs)

coder = coder_model.predict(x)
print(coder[0])

