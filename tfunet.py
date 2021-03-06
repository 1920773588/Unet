from tf_unet import unet, util, image_util
import cv2
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# data
data_provider = image_util.ImageDataProvider('data/train/*.tif')

# setup and training
net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)
trainer = unet.Trainer(net)
path = trainer.train(data_provider, output_path='train_output', training_iters=32, epochs=1000)
exit()
# verification
#val_data_provider = image_util.ImageDataProvider('data/val/*.tif')
#val_images, val_masks = val_data_provider(5)
#prediction = net.predict('train_output/model.ckpt', val_images)

#mask_sample = val_masks[0, :, :, 0]
#mask_sample[mask_sample < 0.5] = 255
#mask_sample[mask_sample < 1] = 0

#prediction_sample = prediction[0, :, :, 0]
#prediction_sample[prediction_sample < 0.5] = 255
#prediction_sample[prediction_sample < 1] = 0

#cv2.imwrite('mask.png', mask_sample)
#cv2.imwrite('prediction.jpg', prediction_sample)
#print(unet.error_rate(prediction, util.crop_to_shape(val_masks, prediction.shape)))
# img = util.combine_img_prediction(val_images, val_masks, prediction)
# util.save_image(img, "prediction.jpg")