#!/usr/bin/env python
# coding: utf-8

# ## Filling Missing Values in Python
# 
# This notebook demonstrates various methods to handle missing values in Python using popular libraries like pandas and scikit-learn.
# 

# In[1]:
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4]
})

# Display initial DataFrame
print("Initial DataFrame:")
print(df)


# In[2]:


# Fill with a specific value
df_fillna_0 = df.fillna(0)
print("\nFilled with 0:")
print(df_fillna_0)


# In[3]:


# Fill with the mean
df_fillna_mean = df.copy()
df_fillna_mean['A'].fillna(df_fillna_mean['A'].mean(), inplace=True)
print("\nFilled with mean of 'A':")
print(df_fillna_mean)

# Fill with the median
df_fillna_median = df.copy()
df_fillna_median['B'].fillna(df_fillna_median['B'].median(), inplace=True)
print("\nFilled with median of 'B':")
print(df_fillna_median)

# Fill with the mode
df_fillna_mode = df.copy()
df_fillna_mode['A'].fillna(df_fillna_mode['A'].mode()[0], inplace=True)
print("\nFilled with mode of 'A':")
print(df_fillna_mode)


# In[4]:


# Forward fill
df_ffill = df.fillna(method='ffill')
print("\nForward fill:")
print(df_ffill)

# Backward fill
df_bfill = df.fillna(method='bfill')
print("\nBackward fill:")
print(df_bfill)


# In[5]:


# Interpolate missing values
df_interpolated = df.interpolate()
print("\nInterpolated DataFrame:")
print(df_interpolated)


# In[6]:


# Simple Imputer
data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
simple_imputer = SimpleImputer(strategy='mean')
imputed_data_simple = simple_imputer.fit_transform(data)
print("\nSimple Imputer (mean):")
print(imputed_data_simple)


# In[7]:


# Iterative Imputer
iterative_imputer = IterativeImputer()
imputed_data_iterative = iterative_imputer.fit_transform(data)
print("\nIterative Imputer:")
print(imputed_data_iterative)


# In[8]:


# KNN Imputer
knn_imputer = KNNImputer(n_neighbors=2)
imputed_data_knn = knn_imputer.fit_transform(data)
print("\nKNN Imputer:")
print(imputed_data_knn)


# # Deploy with streamlit

# In[15]:


# Function to fill missing values
def fill_missing_values(data, columns, method, value=None):
    if method == 'Mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'Median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'Most Frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    elif method == 'Constant':
        if value is not None:
            imputer = SimpleImputer(strategy='constant', fill_value=value)
        else:
            st.error("Please provide a constant value to fill missing values.")
            return data
    else:
        st.error("Unsupported method")
        return data
    
    try:
        data[columns] = imputer.fit_transform(data[columns])
    except Exception as e:
        st.error(f"Error while filling missing values: {e}")
        st.write("Data preview:")
        st.write(data)
        return data
    return data


# In[16]:


# Streamlit app
st.title("Fill Missing Values in Your Dataset")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data)

    # Select columns to fill missing values
    all_columns = data.columns.tolist()
    columns = st.multiselect("Select columns to fill missing values", options=all_columns, default=all_columns)

    # Select method to fill missing values
    method = st.selectbox(
        "Select a method to fill missing values",
        ('Mean', 'Median', 'Most Frequent', 'Constant')
    )
    # Input for constant value if method is 'Constant'
    constant_value = None
    if method == 'Constant':
        constant_value = st.text_input("Enter a constant value to fill missing values")

    if st.button("Fill Missing Values"):
        filled_data = fill_missing_values(data, columns, method, constant_value)
        st.write("Data after filling missing values:")
        st.write(filled_data)
        # Download link for the filled dataset
        st.download_button(
            label="Download filled data as CSV",
            data=filled_data.to_csv(index=False),
            file_name='filled_data.csv',
            mime='text/csv'
        )

