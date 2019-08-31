# ICDAR2019RecTS 
This is my code about the task1 and task2 of the [ICDAR 2019 Robust Reading Challenge on Reading Chinese Text on Signboard](http://rrc.cvc.uab.es/?ch=12 "ICDAR2019")
# Docs
## ICDARRecTs_task2 
      
   + ICDARRecTs_2Configure.py:
       - This is the configuration file, but I can't use it to configure in this code.
   + ICDARRecTs_2DataSet.py:
       - This the dataset file, the class **ICDARRecTs_2DataSet** is the subclass of the **Dataset** 
           * img_path(str): refer to the path of the image.  
           * dictionary Optional(dict): refer to the dictionary that use to decode the char to number,but it isn't deprecated to use. 
           * coordinates_path Optional(str): refer to the path of the coordinate of the textline.
           * img_transform(Callable): refer to the image's transfomation function.
           * train(bool): refer to the model is train or test.
           * load_img(Callable): refer to the load a image function.
   + ICDARRecTs_2NN.py:
       - This is the model file, it includes 'DenseLSTM','VGGLSTM','DenseCNN','VGGFC','ResNetLSTM'.
   + ICDARRecTs_2Preprocessing.py:
       - This is the preprocessing file, it mainly includes the function of **rotate_crop**, **get_dictionary**
           * rotate_crop: it uses to rectify the textline according to the coordinates.
           * get_dictionary: it uses to get the dictionary of the char.
   + ICDARRecTs_2Test.py:
       - DEVICE: refer to test on a CPU or GPU device.
       - BATCH_SIZE: refer to batch size.
       - PATH: refer to the path of dictionary.
       - DICTIONARY_NAME: refer to the name of the dictionary.
       - COORDINATES_PATH: refer to the path of the coordinate of the textline.
       - IMAGE_PATH: refer to the path of the image.
       - MODEL_PATH: refer to the path of the parameters of the model.
       - MODEL_NAME: refer to the name of the parameters of the model.
       - NUM_CLASS: refer to the number of the classification.
   + ICDARRecTs_2Train.py:
       - DEVICE: refer to test on a CPU or GPU device
       - BATCH_SIZE: refer to the batch size.
       - EPOCH: refer to the loop count。
       - PATH：refer to the path of dictionary.
       - DICTIONARY_NAME：refer to the path of the coordinate of the textline.
       - IMAGE_PATH：refer to the path of the image.
       - MODEL_PATH：refer to the path of the parameters of the model.
       - MODEL_NAME：refer to the name of the parameters of the model.
       - PRETRAIN：refer to whether there is a saved parameters of the model.
       - NUM_CLASS: refer to the number of the classification.
       - LR: refer to the initial the learning rate.
       - MAX_ACCURACY: refer to the current max accuracy.
