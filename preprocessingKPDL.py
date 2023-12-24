import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder

class LoadData():
    def __init__(self, _data_path) -> None:
        self._data_path = _data_path
        self.data, self.shape = self.readData()

    def readData(self):
        data = pd.read_csv(self._data_path)
        shape = data.shape
        return data, shape
    
class Preprocessing(LoadData):
    def __init__(self, _data_path) -> None:
        super().__init__(_data_path)

    def viewData(self, n):
        return self.data.head(n)    
    
    def infor(self):
        return self.data.info()

    def isNull(self):
        isnull = self.data.isnull().sum()
        return isnull
    
    def describe_col(self, column):
        return self.data[column].describe()
    
    def fill(self, column):
        return self.data[column].fillna(self.data[column].mean())
    
    def dropnull(self):
        self.data = self.data.dropna()
        return self.data

    
    def dropcol(self, col):
        if self.data is not None:
            self.data = self.data.drop(columns=col, axis=1)
            return self.data
        else:
            return "No data available to drop columns from."
    
    def encoder(self, column, method='LabelEncoder'):
        methods = {'LabelEncoder': LabelEncoder(), 'LabelBinarizer': LabelBinarizer(), 'OneHotEncoder': OneHotEncoder()}
        encoder = methods.get(method)
        if encoder:
            encoded_data = encoder.fit_transform(self.data[column].values.reshape(-1, 1)).flatten()
            self.data[column] = encoded_data
        return self.data[column].values

    def balance_data(self, label_column):
        count_labels = self.data[label_column].value_counts()
        minority_label = count_labels.idxmin()
        majority_label = count_labels.idxmax()
        minority_count = count_labels[minority_label]

        majority_data = self.data[self.data[label_column] == majority_label].sample(n=minority_count, random_state=42)
        minority_data = self.data[self.data[label_column] == minority_label]

        balanced_data = pd.concat([majority_data, minority_data])
        return balanced_data

