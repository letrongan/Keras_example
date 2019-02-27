from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GRU, Bidirectional
from keras.optimizers import Adam

import numpy as np

input_layer = Input(shape=(300,26), name='input' )

conv_2d = Conv1D(filters= 64, kernel_size= 5, strides= 5, padding= 'valid')(input_layer)

bi_gru = Bidirectional(GRU(units=64, activation='elu', return_sequences=False))(conv_2d)
dense_1 = Dense(units=32, activation='elu')(bi_gru)
dense_2 = Dense(units=2, activation='sigmoid')(dense_1)
model = Model(input_layer,dense_2)
print(model.summary())

np.random.seed(0)
data = np.random.rand(100,300,26)
label = np.random.randint(0,2,(100,))

optimizer = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer)

model.fit(data,label,epochs=50)
model.save('training_model.h5')

#letrongan
#update



























# -------------------------------------------------------------------------------------------------
# from keras.models import Model
# from keras.layers import Input
# from keras.layers import Conv1D, MaxPooling2D
# from keras.layers import GRU, Bidirectional
# from keras.layers import Dense
# from keras.optimizers import Adam
#
# # Build the model
# # input layers (as placeholder)
# input_layer = Input(shape=(300,26), name='input')
# # convolutional layers
# conv_2d = Conv1D(filters=64, kernel_size=5, strides=5, padding='valid')(input_layer)
# # recurrent_layers
# bi_gru = Bidirectional(GRU(units=64, activation='elu', return_sequences=False))(conv_2d)
# dense_1 = Dense(units=32, activation='elu')(bi_gru)
# dense_2 = Dense(units = 1, activation='sigmoid')(dense_1)
# model = Model(input_layer, dense_2)
# print (model.summary())
#
# import numpy as np
# # for consistency
# np.random.seed(0)
# data = np.random.rand(100, 300, 26)
# label = np.random.randint(0,2,(100,))
#
# from keras.optimizers import Adam
# optimizer = Adam(lr=0.001)
# model.compile(loss = 'binary_crossentropy', optimizer=optimizer)
#
# model.fit(data, label, epochs=10)
# model.save('trained_model.h5')
#
# from keras.models import load_model
# # Pass your .h5 file here
# model = load_model('trained_model.h5')
# test_data = np.random.rand(1,300,26)
# result = model.predict(test_data)
# print (result)
