 
##############
## Tiny Imagenet Dataset Loader Utiltiy
##########

## to load images
import os
import pathlib
import cv2
import numpy as np
import pandas as pd
import glob
import cv2


class DatasetLoader:
    def __init__(self, image_root_dir=''):
        self.IMAGE_ROOT_DIR = image_root_dir
        self.SUPPORT_FILES_DIR=''
        return

    def set_image_root_dir(self, image_root_dir):
        self.IMAGE_ROOT_DIR = image_root_dir
        return

    def set_support_files_dir(self, support_dir):
        self.SUPPORT_FILES_DIR = support_dir
        return
    '''
    get all the class ids
    '''
    def get_class_ids(self):
        class_file = os.path.join(self.SUPPORT_FILES_DIR, 'class_name_info.csv')
        class_info_df = pd.read_csv(class_file)
        return np.array(class_info_df['classID'])

    def get_class_info(self):
        class_file = os.path.join(self.SUPPORT_FILES_DIR, 'class_name_info.csv')
        class_info_df = pd.read_csv(class_file)
        return (np.array(class_info_df['classID']), np.array(class_info_df['className']))
    '''
    Given a set of class names, get the image file names.
    '''
    def get_training_file_names(self, class_name_list):
        res_file_list = []
        dir_names_list = []
        # get the dir names fist
        for class_name in class_name_list:
            dir_name = os.path.join(self.IMAGE_ROOT_DIR, 'train', class_name, 'images')
            #print(dir_name)
            dir_names_list.append(dir_name)
        # get the file names
        for dir in dir_names_list:
            files = glob.glob(os.path.join(dir, '*.JPEG'))
            res_file_list.append(files)

        return (np.asarray(res_file_list).flatten())

    '''
    For a given class id, load images.
    '''
    def load_training_images_for_a_class(self, class_id, resize_flag=False):
        res_data = []
        res_label = []

        class_id_list = []
        # get training file names
        class_id_list.append(class_id)
        file_list = self.get_training_file_names(class_id_list)

        for file_name in file_list:
            img = cv2.imread(file_name)

            if (resize_flag == True):
                img = cv2.resize(img, (32, 32),interpolation=cv2.INTER_AREA)

            res_data.append(img)
            res_label.append(class_id)

        return( np.array(res_data), np.array(res_label))

    '''
    For a given set of class ids, load images.
    '''
    def load_training_images_for_a_list_classes(self, class_id_list, resize_flag=False):
        res_data = []
        res_label = []

        for class_id in class_id_list:
            (data, label) = self.load_training_images_for_a_class(class_id, resize_flag)
            res_data.append(data)
            res_label.append(label)
        # vertically stack the data and return it.
        return ( np.array(np.vstack(res_data)), np.array(res_label).flatten())

    ############# validation ###########################################
    
    '''
    get_complete_validation_data_info()
    output: file-name-list class-ids
    '''
    def get_complete_validation_data_info(self):
        res_file_list =[]
        val_file = os.path.join(self.IMAGE_ROOT_DIR, 'val', 'val_annotations.txt')
        val_df = pd.read_csv(val_file, header=None, sep='\t',  
                                names=['FileName', 'ClassID', 'X', 'Y', 'H', 'W'])
        res_file_list = [os.path.join(self.IMAGE_ROOT_DIR,'val','images', file_name) for file_name in val_df['FileName'] ]
        return( np.array(res_file_list), np.array(val_df['ClassID']))
    
    '''
    get_validation_data_for_a_class()
    output: file-name-list class-ids
    '''
    def get_validation_data_for_a_class(self, class_id):
        res_file_list = []
        val_file = os.path.join(self.IMAGE_ROOT_DIR, 'val', 'val_annotations.txt')
        val_df = pd.read_csv(val_file, header=None, sep='\t',  
                                names=['FileName', 'ClassID', 'X', 'Y', 'H', 'W'])
        result_df = val_df[ val_df['ClassID'] == class_id]
        res_file_list = [os.path.join(
            self.IMAGE_ROOT_DIR, 'val', 'images', file_name) for file_name in result_df['FileName']]
        return(np.array(res_file_list), np.array(result_df['ClassID']))

    '''
    get_validation_data_info( given classIDList)
    output: file-name-list class-ids
    '''
    def get_validation_data_info(self, class_id_list):
        res_file_list = []
        res_label_list = []
        for class_id in class_id_list:
            (file_list, class_id_list) = self.get_validation_data_for_a_class(class_id)
            res_file_list.append(file_list)
            res_label_list.append(class_id_list)

        return( np.array(res_file_list).flatten(), np.array(res_label_list).flatten())

    '''
    load_validation_images_for_a_class
    output: val-images, val-labels
    '''
    ## load validation images
    def load_validation_images_for_a_class(self, class_id, resize_flag=False):
        res_data = []
        res_label = []

        # get validation file names
        (file_list, label_list) = self.get_validation_data_for_a_class(class_id)

        #print(file_list)
        # go thru each file and load them.
        for file_name in file_list:
            img = cv2.imread(file_name)
            if (resize_flag == True):
                img = cv2.resize(img, (32, 32),interpolation=cv2.INTER_AREA)

            res_data.append(img)
             
        res_label.append(label_list)

        return( np.array(res_data), np.array(res_label))

      
    '''
    load_validation_images_for_a_list_classes
    output: val-images, val-labels
    '''
    def load_validation_images_for_a_list_classes(self, class_id_list, resize_flag=False):
        res_data = []
        res_label = []

        for class_id in class_id_list:
            (data, label) = self.load_validation_images_for_a_class(class_id, resize_flag)
            res_data.append(data)
            res_label.append(label)
        # vertically stack the data and return it.
        return (np.array(np.vstack(res_data)), np.array(res_label).flatten())
