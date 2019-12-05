import pickle

from sklearn.datasets import load_iris
import tensorflow as tf

from preprocess import MySimpleScaler

iris = load_iris()
scaler = MySimpleScaler()
num_classes = len(iris.target_names)
X = scaler.preprocess(iris.data)
y = tf.keras.utils.to_categorical(iris.target, num_classes=num_classes)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(25, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax))
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=1)

model.save('model.h5')
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(scaler, f)
