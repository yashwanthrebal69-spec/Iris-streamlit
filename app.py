import streamlit as st
from sklearn.datasets import load_iris
data = load_iris()
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
x = data.data
y = data.target
model.fit(x,y)
st.header("Iris Flower Classification")
sl = st.number_input("enter sepal Length")
sw = st.number_input("enter sepal Width")
pl = st.number_input("enter petal Length")
pw = st.number_input("enter petal width")
y_pred = model.predict([[sl,sw,pl,pw]])
op = data.target_names[y_pred[0]]
st.write(op)
