import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

st.title('Weather Dataset Analysis')

full_data = pd.read_csv('weather.csv')
st.subheader('Dataset Overview')
st.write(full_data.head())
st.write('Shape of the dataset:', full_data.shape)

full_data['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
full_data['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

st.subheader('Class Distribution Before Oversampling')
fig, ax = plt.subplots(figsize=(8,5))
full_data.RainTomorrow.value_counts(normalize=True).plot(kind='bar', color=['skyblue','navy'], alpha=0.9, rot=0, ax=ax)
ax.set_title('RainTomorrow Indicator No(0) and Yes(1) - Imbalanced Dataset')
st.pyplot(fig)

no = full_data[full_data.RainTomorrow == 0]
yes = full_data[full_data.RainTomorrow == 1]
yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)
oversampled = pd.concat([no, yes_oversampled])

st.subheader('Class Distribution After Oversampling')
fig, ax = plt.subplots(figsize=(8,5))
oversampled.RainTomorrow.value_counts(normalize=True).plot(kind='bar', color=['skyblue','navy'], alpha=0.9, rot=0, ax=ax)
ax.set_title('RainTomorrow Indicator No(0) and Yes(1) - Balanced Dataset')
st.pyplot(fig)

st.subheader('Missing Data Heatmap')
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(oversampled.isnull(), cbar=False, cmap='PuBu', ax=ax)
st.pyplot(fig)

total = oversampled.isnull().sum().sort_values(ascending=False)
percent = (oversampled.isnull().sum()/oversampled.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
st.subheader('Missing Data Statistics')
st.write(missing.head(4))

lencoders = {}
for col in oversampled.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    oversampled[col] = lencoders[col].fit_transform(oversampled[col])

MiceImputed = oversampled.copy(deep=True) 
simple_imputer = SimpleImputer(strategy='mean')
MiceImputed.iloc[:, :] = simple_imputer.fit_transform(oversampled)

Q1 = MiceImputed.quantile(0.25)
Q3 = MiceImputed.quantile(0.75)
IQR = Q3 - Q1
MiceImputed = MiceImputed[~((MiceImputed < (Q1 - 1.5 * IQR)) | (MiceImputed > (Q3 + 1.5 * IQR))).any(axis=1)]

st.subheader('Correlation Heatmap')
corr = MiceImputed.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(250, 25, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, annot=True, linewidths=.5, cbar_kws={'shrink': .9}, ax=ax)
st.pyplot(fig)

st.subheader('Pairplot')
sns.pairplot(data=MiceImputed, vars=['MaxTemp', 'MinTemp', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Evaporation'], hue='RainTomorrow')
st.pyplot()