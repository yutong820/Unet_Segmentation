
import utils
import keras2onnx
import onnx
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

model = load_model('unet_lung_seg.hdf5', custom_objects={'dice_coef': utils.dice_coef, 'dice_coef_loss': utils.dice_coef_loss}, compile=False)  
model.compile(optimizer=Adam(learning_rate=1e-5), loss=utils.dice_coef_loss, metrics=[utils.dice_coef, 'binary_accuracy'])
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = './convert_model/result.onnx'
onnx.save_model(onnx_model, temp_model_file)
# file = open("model.onnx", "wb")
# file.write(onnx_model.SerializeToString())
# file.close()

# import tensorflow as tf
# import utils
# model_path = './unet_lung_seg.hdf5'
# model = tf.keras.models.load_model(model_path, custom_objects={'dice_coef': utils.dice_coef, 'dice_coef_loss': utils.dice_coef_loss})
# model.save('tfmodel', save_format = 'tf')
# python -m tf2onnx.convert --saved-model ./tfmodel/ --output ./models/model.onnx --opset 12 --verbose