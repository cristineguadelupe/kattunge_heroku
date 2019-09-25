from django.db import models
from django.urls import reverse

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model, model_from_json
from tensorflow.python.keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.applications import imagenet_utils
from keras import backend as K


class Classification(models.Model):
    img = models.ImageField(upload_to='images')
    prediction = models.CharField(max_length=200, blank=True)

    def predict(self):
        K.reset_uids()

        model = 'core/model/final_model.h5'

        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            with open(model, 'r') as f:
                model = load_model(model)

        # img = image.load_img(self.img, target_size=(224, 224))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        # result = model.predict(x)
        # result_decode = imagenet_utils.decode_predictions(result)
        #
        # for (i, (predId, pred, prob)) in enumerate(result_decode[0]):
        #     return "{}.-  {}: {:.2f}%".format(i + 1, pred, prob * 100)

        def load_image(filename):
            img = load_img(filename, target_size=(224, 224))
            img = img_to_array(img)
            img = img.reshape(1, 224, 224, 3)
            img = img.astype('float32')
            img = img - [123.68, 116.779, 103.939]
            return img

        img = load_image(self.img)
        result = model.predict(img)

        # img = image.load_img(self.img, target_size=(224, 224))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        # result = model.predict(x)

        if result[0] == 0:
            return 'É um gatinho!'
        else: return 'Não é um gatinho...'

    def save(self, *args, **kwargs):
        self.prediction = self.predict()
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('list')
