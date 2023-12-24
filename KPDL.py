from preprocessingKPDL import Preprocessing
from modelKPDL import SelectModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import pandas as pd


pre = Preprocessing('D:\AI\heart_disease.csv')
data = pre.data

# View data
print('5 dong dau Data: \n', pre.viewData(5))
print('Data shape: \n', pre.shape)
print('Info: \n', pre.infor())
print('Is null: \n', pre.isNull())

# Cột glucose đang có quá nhiều missing hơn những cột khác, xem cột
print('Mô tả cột glucose: ', pre.describe_col(column='glucose'))

# Điền khuyết bằng gái trị mean
data['glucose'] = pre.fill('glucose')


# Các cột còn lại không miss quá nhiều, xóa miss
data = pre.dropnull()

# Các cột Gender, Education, prevalentStroke không ảnh hưởng tới độ chính xác mô hình, tiến hành drop
cols_to_drop = ['Gender', 'education', 'prevalentStroke']
data = pre.dropcol(cols_to_drop)

# Label Encoder cột Heart_stroke
data['Heart_ stroke'] = pre.encoder(column='Heart_ stroke', method='LabelEncoder')


# Bây giờ 'balanced_data' chứa dữ liệu đã được cân bằng giữa hai nhãn
label_column = 'Heart_ stroke'
balanced_data = pre.balance_data(label_column)

X = balanced_data.drop(columns=[label_column])  # Features
y = balanced_data[label_column]  # Target variable

# Chia data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)
print(len(X_train))
model = LogisticRegression(max_iter=1000)
model = SelectModel(model=model, X=X_train, y=y_train, save_path='model.sav')
model.train_model()

# Load model từ file để sử dụng
loaded_model = pickle.load(open('model.sav', 'rb'))
y_pred = loaded_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_maxtrix = confusion_matrix(y_test, y_pred)
print(acc)
print(pre)
print(f1)
print(conf_maxtrix)
