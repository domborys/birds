import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.io import read_image

class Birds270Dataset(Dataset):
    def make_labels(self, csv_table):
        bird_str_labels = csv_table["labels"].unique()
        self.labels_str_to_int = {label:i  for i, label in enumerate(bird_str_labels)}
        self.labels_int_to_str = {i:label  for i, label in enumerate(bird_str_labels)}
        self.label_set = LabelSet(bird_str_labels)
    def find_set_type(self, set_type):
        if type(set_type) is str:
            return [set_type]
        else:
            return set_type
        
    def __init__(self, dataset_dir, set_type=None, transform=None, selected_birds=None):
        set_type_list = self.find_set_type(set_type)
        csv_table = pd.read_csv(os.path.join(dataset_dir, "birds.csv"))
        if selected_birds != None:
            csv_table = csv_table[csv_table["labels"].isin(selected_birds)]
        if set_type == None:
            self.img_data = csv_table
        else:
            self.img_data = csv_table[csv_table["data set"].isin(set_type_list)]
        self.make_labels(self.img_data)
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        filepaths_col = self.img_data.columns.get_loc("filepaths")
        labels_col = self.img_data.columns.get_loc("labels")
        img_path = os.path.join(self.dataset_dir, self.img_data.iat[idx, filepaths_col])
        image = read_image(img_path).float()
        label = self.img_data.iat[idx, labels_col]
        if self.transform:
            image = self.transform(image)
        int_label = self.labels_str_to_int[label]
        return image, int_label

    def get_label_set(self):
        return self.label_set

class LabelSet:
    def __init__(self, str_labels):
        self.labels_str_to_int = {label:i  for i, label in enumerate(str_labels)}
        self.labels_int_to_str = {i:label  for i, label in enumerate(str_labels)}
    
    def get_str_labels(self):
        return self.labels_str_to_int.keys()
    
    def get_int_labels(self):
        return self.labels_int_to_str.keys()
    
    def str_to_int(self, str_label):
        return self.labels_str_to_int[str_label]

    def int_to_str(self, int_label):
        return self.labels_int_to_str[int_label]
    
