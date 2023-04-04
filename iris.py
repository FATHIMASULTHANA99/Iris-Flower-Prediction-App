import streamlit as st
import pandas as pd
import numpy as np
from  sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write ('IRIS FLOWER PREDICTION APP' )
st.sidebar.header('User input parameters')

def user_input_features():
    sepal_length=st.sidebar.slider('sepal length',4.3,7.9,5.4)
    sepal_width=st.sidebar.slider('sepal width',2.0,4.4,3.4)
    petal_length = st.sidebar.slider('petal length', 2.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('petal width', 0.1, 2.5, 0.2)

    data={'sepal_length':sepal_length,
          'sepal_width':sepal_width,
          'petal_length': petal_length,
          'petal_width': petal_width}

    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_features()

st.subheader('User input parameters')
st.write(df)

iris=datasets.load_iris()
x=iris.data
y=iris.target

clf=RandomForestClassifier()
clf.fit(x,y)

##prediction

prediction =clf.predict(df)
prediction_proba =clf.predict_proba(df)

st.subheader('class labels and their cureesponding index numbers')
st.write(iris.target_names)

st.subheader('prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('prediction probability')
st.write(prediction_proba)








