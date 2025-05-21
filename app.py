import os
os.system("pip install openpyxl")

# Import Library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import os
import streamlit as st

st.write('app.py', 'dataset_siga.xlsx', 'requirements.txt')
st.write(os.listdir('.'))


# Connect to Google Drive

data = pd.read_excel('dataset_siga.xlsx')


print('Jumlah Baris dan Kolom Kasus Kekerasan Provinsi Sumatera Utara Tahun 2018 :\n', data.shape)
print('Jumlah Baris Kasus Kekerasan Provinsi Sumatera Utara Tahun 2018 : \n', data.shape[0])
print('Jumlah Kolom Kasus Kekerasan Provinsi Sumatera Utara Tahun 2018 :\n', data.shape[1])

# Data Exploratory Analysis

# show object variable

categorical = [var for var in data.columns if data[var].dtype=='O']
print('Terdapat {} categorical variabel\n'.format(len(categorical)))
print('Variabel categorical adalah:\n', categorical)

# show categorical and numerical variable

numerikal =[]
for num in data.columns:
  if data[num].dtypes == 'int64':
    numerikal.append(num)
print('\nAda {} variabel numerik:\n'.format(len(numerikal)))
print('Variabel numerik adalah:\n', numerikal)

print('Ringkasan Data Kekerasan Provinsi Sumatera Utara: \n')
data.info()

data['Tahun'] = pd.to_datetime(data['Tahun'].astype(str) + '-01-01')

print('Statistik Deskriptif Data Kasus Kekerasan Provinsi Sumatera Utara: \n')
desc_data = data.describe()
np.transpose(desc_data)

print('Korelasi dataset: \n')
# Drop the non-numeric 'Kabupaten/Kota' and 'Jenis Kelamin' columns before calculating correlation
corr = data.drop(['Kabupaten/Kota', 'Jenis Kelamin'], axis=1).corr(method='pearson')
corr

plt.figure(figsize=(24,15))
sns.heatmap(data=corr, annot=True)
plt.title('Korelasi Pearson Heatmap', pad=20, fontsize=25)
plt.savefig('korelasi.png')
plt.show()

print('Mengecek keseimbangan data: \n')
print(data['Jenis Kelamin'].value_counts())

laki = len(data[data['Jenis Kelamin'] == 'Laki-laki'])
perempuan = len(data[data['Jenis Kelamin'] == 'Perempuan'])

lakipersen = laki/(laki+perempuan)
print('Persentase dari Laki-laki:', lakipersen * 100)

perempuanpersen = perempuan/(laki+perempuan)
print('Persentase dari Perempuan:', perempuanpersen*100)

print('rata-rata korban dan pelaku berdasarkan jenis kelamin: \n')
observ = data.copy()
# Drop the non-numeric 'Tahun' and 'Kabupaten/Kota' columns before calculating the mean
observ.drop(['Tahun', 'Kabupaten/Kota'], axis=1, inplace=True)
observ.groupby(['Jenis Kelamin']).mean()

print('rata-rata banyaknya kasus berdasarkan tahun:\n')
# Select only the numeric columns before calculating the mean
numeric_data = data.select_dtypes(include=np.number)
numeric_data.groupby(data['Tahun']).mean()

datatahun = data.loc[:,'Tahun':'Kekerasan Lainnya']
datatahun.drop(['Kabupaten/Kota'], axis=1, inplace=True)
datatahun['Tahun'] = datatahun['Tahun'].dt.year.astype(str)
datatahun['Total'] = datatahun.loc[:,'Kekerasan Fisik':'Kekerasan Lainnya'].sum(axis=1)
datagroup = datatahun.groupby(['Tahun']).sum().reset_index()
datagroup

import matplotlib.dates as mdates

plt.figure(figsize=(14,7))
sns.set()
var = datagroup['Tahun']
num = datagroup['Total']

# Menggunakan lineplot di sini
lines = sns.lineplot(x=var, y=num, data=datagroup, palette='Blues_r', marker='o')  # menambahkan marker untuk titik data

lines.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.xlabel('Tahun', fontsize=18, labelpad=10)
plt.ylabel('Jumlah Kasus', fontsize=18)
plt.title('Kasus Kekerasan Provinsi Sumatera Utara', pad=40, fontsize=25, color='blue')
plt.xticks(datagroup['Tahun'].unique())
plt.savefig('jumlah kasus berdasarkan tahun.png', bbox_inches='tight')  # Memperbaiki typo dari bbox_tight menjadi bbox_inches='tight'
plt.show()

datagroup=data.loc[:,'Tahun':'Kekerasan Lainnya']
datagroup.drop(['Kabupaten/Kota'], axis=1, inplace=True)
datagroup['Tahun'] = datagroup['Tahun'].dt.year
datagroup['Total'] = datagroup.loc[:,'Kekerasan Fisik':'Kekerasan Lainnya'].sum(axis=1)
datagrouptahun = datagroup.groupby(['Tahun'])[['Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual','Eksploitasi','Trafficking','Penelantaran','Kekerasan Lainnya']].sum()
datagrouptahun

fig, ax = plt.subplots(figsize=(12,8))
plt.style.use('ggplot')
labels = ['Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual',
          'Eksploitasi','Trafficking','Penelantaran','Kekerasan Lainnya']
x = np.arange(7)
datagroup_tjenis = np.transpose(datagrouptahun)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(datagroup_tjenis.columns):
  bar = plt.bar(x+(a*width), datagroup_tjenis[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x-0, labels=datagroup_tjenis.index, rotation=15)
ax.set_xticklabels(labels)
ax.set_title('Kasus Kekerasan Berdasarkan Jenis Kekerasan \nProvinsi Sumatera Utara', fontsize=22, pad=25, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Jenis Kekerasan', fontsize=14)
plt.legend(ncol=3, fancybox=True, shadow=True, loc='upper right', title='Tahun')
plt.savefig('jumlah kasus berdasrkan jenis kekerasan.png')
plt.show()

datagroupusia=data[['Tahun','0-5 tahun','6-12 tahun','13-17 tahun']]
datagroupusia['Tahun'] = datagroupusia['Tahun'].dt.year
datausia = datagroupusia.groupby(['Tahun'])[['0-5 tahun','6-12 tahun','13-17 tahun']].sum()
datausia


fig, ax = plt.subplots(figsize=(11,7))
plt.style.use('ggplot')
labels = ['0-5 tahun','6-12 tahun','13-17 tahun']
x = np.arange(3)
datagroup_age = np.transpose(datausia)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(datagroup_age.columns):
  bar = plt.bar(x+(a*width), datagroup_age[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.2, labels=datagroup_age.index)
ax.set_xticklabels(labels)
ax.set_title('Kasus Kekerasan Berdasarkan Usia \n Provinsi Sumatera Utara', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Usia Korban', fontsize=14, labelpad=10)
plt.legend(ncol=3, fancybox=True, shadow=True, loc='upper left', title='Tahun')
plt.savefig('jumlah kasus berdasarkan usia.png')
plt.show()

datagroupjk=data[['Tahun','Jenis Kelamin','Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual','Penelantaran','Kekerasan Lainnya']]
datagroupjk['Total'] = datagroupjk[['Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual','Penelantaran','Kekerasan Lainnya']].sum(axis=1)
datagroupjk = datagroupjk.groupby(['Jenis Kelamin'])[['Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual','Penelantaran','Kekerasan Lainnya']].sum()
datagroupjk

fig, ax = plt.subplots(figsize=(12,7))
plt.style.use('ggplot')
labels = ['Kekerasan Fisik','Kekerasan Psikis','Kekerasan Seksual','Penelantaran','Lainnya']
x = np.arange(5)
datagroup_jkt = np.transpose(datagroupjk)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(datagroup_jkt.columns):
  bar = plt.bar(x+(a*width), datagroup_jkt[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.2, labels=datagroup_jkt.index)
ax.set_xticklabels(labels)
ax.set_title('Jenis Kekerasan \nBerdasarkan Jenis Kelamin Provinsi Sumatera Utara', fontsize=25, pad=25, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Jenis Kekerasan', fontsize=14, labelpad=10)
plt.legend(ncol=3, fancybox=True)

data17 = data[data['Tahun'] == '2017-01-01']
usia17=data17[['Jenis Kelamin','0-5 tahun','6-12 tahun','13-17 tahun']]
usia17 = usia17.groupby(['Jenis Kelamin'])[['0-5 tahun','6-12 tahun','13-17 tahun']].sum()


fig, ax = plt.subplots(figsize=(10,7))
plt.style.use('ggplot')
labels = ['0-5 tahun','6-12 tahun','13-17 tahun']
x = np.arange(3)
usia17t = np.transpose(usia17)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(usia17t.columns):
  bar = plt.bar(x+(a*width), usia17t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=usia17t.index)
ax.set_xticklabels(labels)
ax.set_title('Kasus Kekerasan Berdasarkan Usia Korban Tahun 2017 \n Provinsi Sumatera Utara', fontsize=22, pad=25, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Range Usia', fontsize=14)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.68,-0.12), title='Jenis Kelamin')
# Change bbox_tight=True to bbox_inches='tight'
plt.savefig('jumlah kasus berdasrkan usia korban 2017.png', bbox_inches='tight')
plt.show()

data18 = data[data['Tahun'] == '2018-01-01']
usia18=data18[['Jenis Kelamin','0-5 tahun','6-12 tahun','13-17 tahun']]
usia18 = usia18.groupby(['Jenis Kelamin'])[['0-5 tahun','6-12 tahun','13-17 tahun']].sum()
data18[['Jenis Kelamin','0-5 tahun','6-12 tahun','13-17 tahun']]

fig, ax = plt.subplots(figsize=(10,7))
plt.style.use('ggplot')
labels = ['0-5 tahun','6-12 tahun','13-17 tahun']
x = np.arange(3)
usia18t = np.transpose(usia18)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(usia18t.columns):
  bar = plt.bar(x+(a*width), usia18t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=usia18t.index)
ax.set_xticklabels(labels)
ax.set_title('Kasus Kekerasan Berdasarkan Usia Korban Tahun 2018 \n Provinsi Sumatera Utara', fontsize=21, pad=25, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Range Usia', fontsize=14, labelpad=3)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.69,-0.12), title='Jenis Kelamin')
# Changed bbox_tight=True to bbox_inches='tight'
plt.savefig('jumlah kasus berdasrkan usia korban 2018.png', bbox_inches='tight')
plt.show()

data19 = data[data['Tahun'] == '2019-01-01']
usia19=data19[['Jenis Kelamin','0-5 tahun','6-12 tahun','13-17 tahun']]
usia19 = usia19.groupby(['Jenis Kelamin'])[['0-5 tahun','6-12 tahun','13-17 tahun']].sum()

fig, ax = plt.subplots(figsize=(10,7))
plt.style.use('ggplot')
labels = ['0-5 tahun','6-12 tahun','13-17 tahun']
x = np.arange(3)
usia19t = np.transpose(usia19)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)) , # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(usia19t.columns):
  bar = plt.bar(x+(a*width), usia19t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=usia19t.index)
ax.set_xticklabels(labels)
ax.set_title('Kasus Kekerasan Berdasarkan Usia Korban Tahun 2019 \n Provinsi Sumatera Utara', fontsize=21, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Range Usia', fontsize=14)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.69,-0.12), title='Jenis Kelamin')
# Changed bbox_tight=True to bbox_inches='tight'
plt.savefig('jumlah kasus berdasrkan usia korban 2019.png', bbox_inches='tight')
plt.show()

datatempat17 = data[data['Tahun'] == '2017-01-01']
tempat17=datatempat17[['Jenis Kelamin','Rumah','Tempat Bekerja','Lingkungan Masyarakat','Sekolah','Lainnya']]
tempat17 = tempat17.groupby(['Jenis Kelamin'])[['Rumah','Lingkungan Masyarakat','Sekolah','Lainnya']].sum()

fig, ax = plt.subplots(figsize=(13,8))
plt.style.use('ggplot')
labels = ['Rumah','Lingkungan Masyarakat','Sekolah','Lainnya']
x = np.arange(4)
tempat17_t = np.transpose(tempat17)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(tempat17_t.columns):
  bar = plt.bar(x+(a*width), tempat17_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=tempat17_t.index)
ax.set_xticklabels(labels)
ax.set_title('Jumlah Kasus Berdasarkan Tempat Kejadian Tahun 2017', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Tempat Kejadian', fontsize=14, labelpad=10)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.63,-0.12), title='Jenis Kelamin')
plt.show()

datatempat18 = data[data['Tahun'] == '2018-01-01']
tempat18=datatempat17[['Jenis Kelamin','Rumah','Tempat Bekerja','Lingkungan Masyarakat','Sekolah','Lainnya']]
tempat18 = tempat18.groupby(['Jenis Kelamin'])[['Rumah','Lingkungan Masyarakat','Sekolah','Lainnya']].sum()

fig, ax = plt.subplots(figsize=(13,8))
plt.style.use('ggplot')
labels = ['Rumah','Lingkungan Masyarakat','Sekolah','Lainnya']
x = np.arange(4)
tempat18_t = np.transpose(tempat18)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(tempat18_t.columns):
  bar = plt.bar(x+(a*width), tempat18_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=tempat18_t.index)
ax.set_xticklabels(labels)
ax.set_title('Jumlah Kasus Berdasarkan Tempat Kejadian Tahun 2018', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Tempat Kejadian', fontsize=14, labelpad=10)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.63,-0.12), title='Jenis Kelamin')
plt.show()

datatempat19 = data[data['Tahun'] == '2019-01-01']
tempat19=datatempat19[['Jenis Kelamin','Rumah','Tempat Bekerja','Lingkungan Masyarakat','Sekolah','Lainnya','Fasilitas Umum']]
tempat19 = tempat19.groupby(['Jenis Kelamin'])[['Rumah','Tempat Bekerja','Sekolah','Lainnya','Fasilitas Umum']].sum()

fig, ax = plt.subplots(figsize=(13,8))
plt.style.use('ggplot')
labels = ['Rumah','Tempat Bekerja','Sekolah','Lainnya','Fasilitas Umum']
x = np.arange(5)
tempat19_t = np.transpose(tempat19)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(tempat19_t.columns):
  bar = plt.bar(x+(a*width), tempat19_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=tempat19_t.index)
ax.set_xticklabels(labels)
ax.set_title('Jumlah Kasus Berdasarkan Tempat Kejadian Tahun 2019', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Tempat Kejadian', fontsize=14, labelpad=10)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.63,-0.12), title='Jenis Kelamin')
plt.show()

datalayanan17 = data[data['Tahun'] == '2017-01-01']
layanan17=datalayanan17[['Jenis Kelamin','Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']]
layanan17 = layanan17.groupby(['Jenis Kelamin'])[['Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']].sum()

fig, ax = plt.subplots(figsize=(20,8))
plt.style.use('ggplot')
labels = ['Pengaduan','Kesehatan','Bantuan Hukum\ndan Penegakan','Rehabilitasi','Pemulangan dan \nReintegrasi','Pendampingan','Mediasi']
x = np.arange(7)
layanan17_t = np.transpose(layanan17)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(layanan17_t.columns):
  bar = plt.bar(x+(a*width), layanan17_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=layanan17_t.index, rotation=0)
ax.set_xticklabels(labels)
ax.set_title('Pelayanan Yang Didapat Oleh Korban Tahun 2017', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Pelayanan', fontsize=14)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.59,-0.12), title='Jenis Kelamin')
plt.show()

datalayanan18 = data[data['Tahun'] == '2018-01-01']
layanan18=datalayanan18[['Jenis Kelamin','Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']]
layanan18 = layanan18.groupby(['Jenis Kelamin'])[['Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']].sum()

fig, ax = plt.subplots(figsize=(20,8))
plt.style.use('ggplot')
labels = ['Pengaduan','Kesehatan','Bantuan Hukum\ndan Penegakan','Rehabilitasi','Pemulangan dan \nReintegrasi','Pendampingan','Mediasi']
x = np.arange(7)
layanan18_t = np.transpose(layanan18)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(layanan18_t.columns):
  bar = plt.bar(x+(a*width), layanan18_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=layanan18_t.index, rotation=0)
ax.set_xticklabels(labels)
ax.set_title('Pelayanan Yang Didapat Oleh Korban Tahun 2018', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Pelayanan', fontsize=14)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.59,-0.12), title='Jenis Kelamin')
plt.show()

datalayanan19 = data[data['Tahun'] == '2019-01-01']
layanan19=datalayanan19[['Jenis Kelamin','Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']]
layanan19 = layanan19.groupby(['Jenis Kelamin'])[['Pengaduan','Kesehatan','Bantuan Hukum dan Penegakan','Penegakan','Rehabilitasi Sosial','Pemulangan dan Reintegrasi Sosial','Pendampingan Tokoh Agama','Mediasi']].sum()

fig, ax = plt.subplots(figsize=(20,8))
plt.style.use('ggplot')
labels = ['Pengaduan','Kesehatan','Bantuan Hukum\ndan Penegakan','Penegakan','Rehabilitasi Sosial','Pemulangan dan \nReintegrasi Sosial','Pendampingan','Mediasi']
x = np.arange(8)
layanan19_t = np.transpose(layanan19)
width = 0.2

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2, 1*height,
    str(round(height,2)), # 3 points vertical offset
    ha='center', va='bottom')

for a, b in enumerate(layanan19_t.columns):
  bar = plt.bar(x+(a*width), layanan19_t[b], width=width, label=b)
  autolabel(bar)

plt.xticks(x+0.1, labels=layanan19_t.index, rotation=0)
ax.set_xticklabels(labels)
ax.set_title('Pelayanan Yang Didapat Oleh Korban Tahun 2019', fontsize=25, pad=40, color='blue')
ax.set_ylabel('Jumlah Kasus', fontsize=14)
ax.set_xlabel('Pelayanan', fontsize=14)
plt.legend(ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.59,-0.12), title='Jenis Kelamin')
plt.show()

# Data Pre-Processing

## Data Cleaning

missing_values = data.isnull()

print('Check Missing Values:\n')
for column in missing_values.columns.values.tolist():
  print(column)
  print(missing_values[column].value_counts())
  print('')

from scipy import stats
datazscore = data.copy()
datazscore.drop(['Kabupaten/Kota','Jenis Kelamin','Rekan Kerja','Tahun','Na'], axis=1, inplace=True)

# Find the outliers using Z score
zscore = np.abs(stats.zscore(datazscore))
print('Nilai Zscore:\n',zscore)
print('\n')

threshold = 3
thres_zscore = zscore>3
loc = np.where(thres_zscore)
print('Lokasi Outliers:\n',loc)
print('Jumlah Outliers Pada Dataset:\n', thres_zscore.sum())

## Encoding

encoder = LabelEncoder()
data['Kabupaten/Kota'] = encoder.fit_transform(data['Kabupaten/Kota'])
data['Jenis Kelamin'] = encoder.fit_transform(data['Jenis Kelamin'])
print('Data setelah LabelEncoder:')
data.head(10)

# Data Transformation

## Feature Selection

data.head()

data = data.drop(['Kabupaten/Kota','Tahun','Kekerasan Lainnya','Lainnya'], axis=1)
datadrop=data.iloc[:,9:24]
data = data.drop(datadrop, axis=1)
data.head()

# Data Modelling

X = data.drop(['Jenis Kelamin'], axis=1)
y = data[['Jenis Kelamin']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

print('X_train shape :\n', X_train.shape)
print('X_test shape :\n', X_test.shape)
print('Baris X_train shape:\n', X_train.shape[0])
print('Kolom X_train shape:\n', X_train.shape[1])

print('Baris X_test shape:\n', X_test.shape[0])
print('Kolom X_test shape:\n', X_test.shape[1])

print('y_train shape :\n', y_train.shape)
print('y_test shape :\n', y_test.shape)
print('Baris y_train shape:\n', y_train.shape[0])
print('Baris y_train shape:\n', y_test.shape[0])

## Data Transformation (Normalization)

scaler = MinMaxScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

print('Normalisasi X_train 80%:\n', X_train_scaler)
print('Normalisasi X_test 20%:\n', X_test_scaler)

## Data Modelling (Implementation Algorithm)

### Split 80%:20%

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_scaler, y_train.values.ravel())

gnbscore_train = gnb.score(X_train_scaler, y_train)
print('Accuracy of Gaussian NB classifier on training set: {:.2f}'
     .format(gnbscore_train))

gnbscore_test = gnb.score(X_test_scaler, y_test)
print('Accuracy of Gaussian NB classifier on test set: {:.2f}'
     .format(gnbscore_test))

scores = cross_val_score(gnb, X, y.values.ravel(), cv=10)
print('Keakuratan Gaussian NB dengan dataset:', scores.mean())
print('%0.2f akurasi dengan standar deviasi sebesar %0.2f' %(scores.mean(), scores.std()))

print('0=Laki-laki\n1=Perempuan\n')
print('Peluang tiap class pada dataset:\n', gnb.class_prior_)

print('Mean tiap feature: \n', gnb.theta_)
print('\nVariansi tiap feature: \n', gnb.var_)

y_pred_gaussian = gnb.predict(X_test_scaler)
print('Angka kesalahan label point dari total %d points : %d' %(X_test_scaler.shape[0], (y_test.values.ravel() != y_pred_gaussian).sum()))

y_pred_gaussian_ds = gnb.predict([[0.608696,0.217391,0,0.25,0,
                                   0.041667,0,0.166667,0.125,
                                   0.170732,0,0.032609,0,0.090909,
                                   0,0,0]])
print('Prediksi Data Sampel Pertama:', y_pred_gaussian_ds)

y_pred_gauss_ds2 = gnb.predict([[1,1.086957,0.052632,0,0,0.541667,
                                 0.791667,0.6875,0.458333,0.012195,
                                 0,0.01087,0,0,0,0,0]])
print('Prediksi Data Sampel Kedua:', y_pred_gauss_ds2)

print('Prediksi seluruh data sampel:\n', y_pred_gaussian)
print('Seluruh data aktual:\n', y_test)

confusion_mat = confusion_matrix(y_test, y_pred_gaussian)
confusion_mat

fig, ax = plt.subplots()
class_names=[0,1] # name  of classes
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(confusion_mat), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix Data Split 80:20', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('confusionmatriks 8020.png')
plt.show()

print('Pengujian Evaluasi Data Split 80:20')
print(classification_report(y_test, y_pred_gaussian))

gnb.fit(X_train.values, y_train.values.ravel())
print('Training set score: {:.4f}'.format(gnb.score(X_train.values, y_train.values.ravel())))
print('Test set score: {:.4f}'.format(gnb.score(X_test.values, y_test.values.ravel())))

### Process of visualization the model

testpred = pd.DataFrame({'Nilai Test':y_test.values.ravel(),'Nilai Prediksi':y_pred_gaussian})

testpred[testpred['Nilai Prediksi']==1].count()

predtest = pd.DataFrame({'Nilai Prediksi': ['0','1'], 'Total': [26,14], 'Nilai Test': ['0','1'],'Total Test':[22,18]}, index=[0,1])
predtestgroup = predtest.groupby(['Nilai Prediksi'])[['Total']].mean().reset_index()
pred_testgroup = predtest.groupby(['Nilai Test'])[['Total Test']].mean().reset_index()

valuespred20 = predtestgroup['Total']
valuesact20 = pred_testgroup['Total Test']

fig, ax = plt.subplots(figsize=(12,8))
explode = (0,0.1)
my_colors = ['#66b3ff','#ff9999']
piechart = ax.pie(predtestgroup['Total'], labels=predtestgroup['Nilai Prediksi'],
          autopct=lambda p:f'{p:.2f}%, {p*sum(valuespred20)/100 :.0f} Anak', explode=explode, shadow=True, colors=my_colors,
          startangle=0)

ax.set_title('Hasil Prediksi Analisis Data Kasus Kekerasan Anak \nVariasi 80:20', color='black',weight='bold',fontsize=20)

plt.legend(bbox_to_anchor = (1,0.6),labels=['Laki-laki','Perempuan'])

plt.savefig('visualisasi data akhir 8020.png')
plt.show()
