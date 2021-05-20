import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class CancerNet:
	@staticmethod
	def build(width,height,depth,classes):
		model=Sequential()
		shape=(height,width,depth)
		channelDim=-1

		if K.image_data_format()=="channels_first":
			shape=(depth,height,width)
			channelDim=1

		model.add(SeparableConv2D(64, (3,3), padding="same",input_shape=shape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(SeparableConv2D(192, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(SeparableConv2D(384, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))

		model.add(SeparableConv2D(256, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(SeparableConv2D(256, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2,2),strides=2))

		model.add(Flatten())
		model.add(Dropout(0.6))
		model.add(Dense(2048))
		model.add(Activation("relu"))
		model.add(Dropout(0.6))
		model.add(Dense(2048))
		model.add(Activation("relu"))
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model
