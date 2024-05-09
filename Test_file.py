import tensorflow as tf
import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

n_inputs = 28 * 28

model = keras.models.load_model("model.keras")

x_test_resized = x_test.reshape(x_test.shape[0], -1)
x_test_resized = x_test_resized.astype('float32') / 255.

print(x_test.shape)
print(x_test_resized[0])

n_rec = 389

plt.imshow(x_test_resized.reshape([10000, 28, 28])[n_rec], cmap='gray')
plt.show()

x = x_test_resized[n_rec]
x = np.expand_dims(x, axis=0)
print(x.shape)

prediction = model.predict(x)

plt.imshow(prediction[0].reshape([28, 28]), cmap='gray')
plt.show()
