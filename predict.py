import streamlit as st
import pandas as pd
import torch
from model import Net
from sklearn.preprocessing import MinMaxScaler

model_path = 'designer_state.pth'


def load_model(file_path):
    """function to load saved state of model"""
    trained_model = Net()
    trained_model.load_state_dict(torch.load(file_path))
    trained_model.eval()

    return trained_model


# Load saved model
model = load_model(model_path)


st.title('AI Beam Designer')

st.sidebar.header('User Input Parameters')


def user_input_features():
    span = st.sidebar.number_input('Span (m)')
    ultimate_load = st.sidebar.number_input('Ultimate load (kN/m)')
    steel_grade = st.sidebar.selectbox(
        'Steel Grade', ['Fe-215', 'Fe-415', 'Fe-500'])
    concrete_strength = st.sidebar.selectbox(
        'Concrete Strength', ['M15', 'M20', 'M25', 'M30'])

    data = {'span': span,
            'ultimate_load': ultimate_load,
            'steel_grade': steel_grade,
            'concrete_strength': concrete_strength}

    features = pd.DataFrame(data)
    return features


df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)


def inference(df):
    input_params = MinMaxScaler().fit_transform(df)
    with torch.no_grad():
        output = model(input_params)

        steel_area = output[0]
        depth = output[1]
        width = output[2]
        cost = output[3]
        moment_capacity = output[4]

        data = {
            'Steel area (cm^2)': steel_area,
            'Depth (cm)': depth,
            'Width (cm)': width,
            'Cost per meter (₦/m)': cost,
            'Moment capacity (kNmm)': moment_capacity
        }
    predictors = pd.DataFrame(data)
    return predictors


prediction = inference(df)
st.header('Model Output')
st.write(prediction)
