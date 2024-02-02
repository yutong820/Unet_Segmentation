import os
from glob import glob
import cv2
import random
from tqdm import tqdm
import numpy as np
import glob
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from tensorflow.keras.models import *
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from keras.models import *
from keras.models import load_model
import utils
from glob import glob as glob_function
import matplotlib.pyplot as plt


DILATE_KERNEL = np.ones((15, 15), np.uint8)

INPUT = r'C:\Users\z004vp1t\Desktop\lung_unet'
INPUT_DIR = os.path.join(INPUT, "input")
SHENZHEN_MASK_DIR = os.path.join(INPUT_DIR, "shcxr-lung-mask", "mask", "mask")
shenzhen_mask_dir = os.path.join(SHENZHEN_MASK_DIR, '*.png') # lung_unet\input\shcxr-lung-mask\mask\mask
select_data = glob.glob(shenzhen_mask_dir)
# shenzhen_datasets = random.sample(select_data, 10)
shenzhen_datasets = select_data[:10]
SEGMENTATION_SOURCE_DIR = os.path.join(INPUT_DIR, \
                                       "pulmonary-chest-xray-abnormalities")
SHENZHEN_TRAIN_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, "ChinaSet_AllFiles", \
                                  "ChinaSet_AllFiles")
SHENZHEN_IMAGE_DIR = os.path.join(SHENZHEN_TRAIN_DIR, "CXR_png")

for mask_file in tqdm(shenzhen_datasets):
    base_file = os.path.basename(mask_file).replace("_mask", "")
    image_file = os.path.join(SHENZHEN_IMAGE_DIR, base_file)   # lung_unet\input\pulmonary-chest-xray-abnormalities\ChinaSet_AllFiles\ChinaSet_AllFiles\CXR_png
    print(base_file)
    image = cv2.imread(image_file)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
    image = cv2.resize(image, (512, 512))
    mask = cv2.resize(mask, (512, 512))
    mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
    
    filename, fileext = os.path.splitext(base_file)

    cv2.imwrite(os.path.join('./Shenzhen_test', base_file), \
                    image)
    cv2.imwrite(os.path.join('./Shenzhen_test', \
                                 "%s_mask%s" % (filename, fileext)), mask)
    cv2.imwrite(os.path.join('./Shenzhen_test', \
                                 "%s_dilate%s" % (filename, fileext)), mask_dilate)

    
model_path = './unet_lung_seg.hdf5' 
model = load_model(model_path, custom_objects={'dice_coef': utils.dice_coef, 'dice_coef_loss': utils.dice_coef_loss})
# model.compile(optimizer=Adam(learning_rate=1e-5), loss=utils.dice_coef_loss, metrics=[utils.dice_coef, 'binary_accuracy'])

test_files = [test_file for test_file in glob_function(os.path.join('./Shenzhen_test', "*.png"))
              if ("_mask" not in test_file and
                  "_dilate" not in test_file and
                  "_predict" not in test_file)]
test_gen = utils.test_generator(test_files, target_size=(512,512))
results = model.predict(test_gen, len(test_files), verbose=1)
utils.save_result('./Shenzhen_test', results, test_files)

dice_scores = []
for i, file_path in enumerate(test_files):
    mask_path = file_path.replace('.png', '_mask.png')  
    true_mask = load_img(mask_path, color_mode='grayscale', target_size=(512, 512))
    true_mask = img_to_array(true_mask)
    true_mask = true_mask / 255.0  
    predicted_mask = results[i]
    predicted_mask = cv2.resize(predicted_mask, (512, 512))
    dice_score = utils.dice_coef(true_mask, predicted_mask)
    dice_scores.append(dice_score)
    
dice_scores_np = np.array(dice_scores)
mean_dice = np.mean(dice_scores_np)
std_dice = np.std(dice_scores_np)
print("Average Dice:", mean_dice)
print("Dice standard devitation:", std_dice)

path = test_files[1]
image = cv2.imread(path)
predict_image = cv2.imread(path.replace('.png', '_predict.png')  )
mask_image = cv2.imread(path.replace('.png', '_dilate.png'))
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].set_title("Inference")
axs[0].imshow(utils.add_colored_mask(image, predict_image))
axs[1].set_title("Gold Std.")
axs[1].imshow(utils.add_colored_mask(image, mask_image))
axs[2].set_title("Diff.")
axs[2].imshow(utils.diff_mask(mask_image, predict_image))
plt.show()




    
 