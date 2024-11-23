import sys
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class PrepareBaseModelConfig:
    def __init__(self):
        self.params_classes=22
        self.params_freeze_all=True
        self.params_freeze_till=None
        self.params_learning_rate=0.01
        self.input_shape=(224,224,3)
        self.params_weights='imagenet'
        self.include_top=False
        self.params_activation="softmax"
        self.updated_model_path=os.path.join("artifacts","prepare_base_model","base_model_updated.h5")
        self.base_model_path=os.path.join("artifacts","prepare_base_model","base_model.h5")

class PrepareBaseModel:
    def __init__(self):
        self.prepare_base_model=PrepareBaseModelConfig()

    def load_base_model(self):
        try:
            self.base_model = tf.keras.applications.vgg16.VGG16(
                            include_top=self.prepare_base_model.include_top,
                            weights=self.prepare_base_model.params_weights,
                            input_shape=self.prepare_base_model.input_shape
                        )
            self.save_model(path=self.prepare_base_model.base_model_path, model=self.base_model)
        
        except Exception as e:
            logging.info(f"Error in load_base_model -- {e}")
            raise CustomException(e,sys)

    def prepare_full_model(self, model, classes, freeze_all, freeze_till, learning_rate):
        try:
            if freeze_all:
                for layer in model.layers:
                    model.trainable = False
            elif (freeze_till is not None) and (freeze_till > 0):
                for layer in model.layers[:-freeze_till]:
                    model.trainable = False
        
            flatten_in = tf.keras.layers.Flatten()(model.output)
            prediction = tf.keras.layers.Dense(
                units = classes,
                activation = self.prepare_base_model.params_activation
            )(flatten_in)
        
            self.full_model = tf.keras.models.Model(
                inputs = model.input,
                outputs = prediction
            )
        
            self.full_model.compile(
                optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate),
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ["accuracy"]
            )
        
            self.full_model.summary()

        except Exception as e:
            logging.info(f"Error in prepare_full_model -- {e}")
            raise CustomException(e,sys)
    
    def save_model(self, path:Path, model: tf.keras.Model):
        try:
            model.save(path)
        
        except Exception as e:
            logging.info(f"Error in save_model -- {e}")
            raise CustomException(e,sys)


    def update_base_model(self):
        try:
            self.load_base_model()
            self.prepare_full_model(
                model = self.base_model,
                classes = self.prepare_base_model.params_classes,
                freeze_all = self.prepare_base_model.params_freeze_all,
                freeze_till = self.prepare_base_model.params_freeze_till,
                learning_rate = self.prepare_base_model.params_learning_rate
            )
            self.save_model(path=self.prepare_base_model.updated_model_path, model=self.full_model) 
        
        except Exception as e:
            logging.info(f"Error in update_base_model -- {e}")
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    prepare_base_model=PrepareBaseModel()
    prepare_base_model.update_base_model()