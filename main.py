import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


train_path = "C:/Users/ASUS/Downloads/archive2/train.csv"
data = pd.read_csv(train_path)
print(data.head())

# sns.pairplot(data[['years', 'km', 'rating', 'condition', 'economy', 'top speed', 'hp', 'torque', 'current price']], diag_kind='kde')
# plt.show()

#convert data to tensor
tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data, tf.float32)

#shuffle data
tensor_data = tf.random.shuffle(tensor_data)


#specifying the columns to consider in training
X = tensor_data[:,3:-1]

#the label data
y = tensor_data[:, -1]

y = tf.expand_dims(y, axis=-1)

# set the train, validation, test data
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SIZE = len(X)

X_train = X[:int(DATASET_SIZE * TRAIN_RATIO)]
y_train = y[:int(DATASET_SIZE * TRAIN_RATIO)]
print(X_train.shape)
print(y_train.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
print(train_dataset)


# for x,y in train_dataset:
#     print(x,y)
#     break

X_val = X[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
y_val = y[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
print(X_val.shape)
print(y_val.shape)


val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
# val_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)


X_test = X[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
y_test = y[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
print(X_test.shape)
print(y_test.shape)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# test_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

normalizer = Normalization()
normalizer.adapt(X_train)
# normalizer(X)[:5]

model = tf.keras.Sequential([
                             InputLayer(input_shape = (8,)),
                             normalizer,
                             Dense(128, activation = "relu"),
                             Dense(128, activation = "relu"),
                             Dense(128, activation = "relu"),
                             Dense(1),
])
model.summary()


model.compile(optimizer = Adam(learning_rate = 0.1),
              loss = MeanAbsoluteError()
              )
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = 150, verbose = 1)
# model.predict(train_dataset.reshape(800, 9))
# model.predict(val_dataset.reshape(100, 9))
# history = model.fit(train_dataset, validation_data=val_dataset, epochs = 100, verbose = 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

model.evaluate(X_test,y_test)



model.predict(tf.expand_dims(X_test[0], axis = 0 ))
y_true = list(y_test[:,0].numpy())
y_pred = list(model.predict(X_test)[:,0])

ind = np.arange(100)
plt.figure(figsize=(40,20))

width = 0.1

plt.bar(ind, y_pred, width, label='Predicted Car Price')
plt.bar(ind + width, y_true, width, label='Actual Car Price')

plt.xlabel('Actual vs Predicted Prices')
plt.ylabel('Car Price Prices')

plt.show()



# plt.plot(history.history['root_mean_squared_error'])
# plt.plot(history.history['val_root_mean_squared_error'])
# plt.title('model performance')
# plt.ylabel('rmse')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'])
# plt.show()

