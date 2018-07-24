from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes): #depth es el color de mis imagenes, 1 grayscale, 3 rgb
        model = Sequential() #iremos agregando layers al modelo
        inputShape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        #primer layer
        model.add(Conv2D(20, (5,5), padding="same", input_shape=inputShape)) #20 convolutional filtros de 5x5
        model.add(Activation("relu")) #funcion de activacion
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #segundo layer
        model.add(Conv2D(50, (5,5), padding="same")) #50 convolutional filtros de 5x5
        #una practica comun es incrementar el numero de convutional filters a medida que vas mas profundo
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #para hacer el FC a que sean relu layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        #clasificador softmax
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
