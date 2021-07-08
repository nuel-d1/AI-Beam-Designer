import streamlit as st
import pandas as pd
import torch
import joblib
from model import Net
from sklearn.preprocessing import MinMaxScaler

model_path = 'designer_state.pth'


def load_model(file_path):
    """function to load saved state of model"""
    model_class = Net()
    model_class.load_state_dict(torch.load(file_path))
    model_class.eval()

    return model_class


# Load saved model
model = load_model(model_path)

st.title('AI Beam Designer')
st.sidebar.header('User Input Parameters')

# user input variables
span = st.sidebar.number_input('Span (m)')
ultimate_load = st.sidebar.number_input('Ultimate load (kN/m)')
steel_grade = st.sidebar.selectbox(
    'Steel Grade', ['Fe-215', 'Fe-415', 'Fe-500'])
concrete_strength = st.sidebar.selectbox(
    'Concrete Strength', ['M15', 'M20', 'M25', 'M30'])

generate_button = st.sidebar.button('generate initial design parameters')

user_input = {'span': span,
              'ultimate load': ultimate_load,
              'steel grade': steel_grade,
              'concrete strength': concrete_strength
              }

st.subheader('User Input Parameters')
st.write(pd.DataFrame(user_input, index=[0]))


def user_input_features(user_input):
    raw_data = {'span': user_input['span'],
                'ultimate_load': user_input['ultimate load'],
                'steel_Fe_215': 1 if user_input['steel grade'] == 'Fe-215' else 0,
                'steel_Fe_415': 1 if user_input['steel grade'] == 'Fe-415' else 0,
                'steel_Fe_500': 1 if user_input['steel grade'] == 'Fe-500' else 0,
                'concrete M15': 1 if user_input['concrete strength'] == 'M15' else 0,
                'concrete M20': 1 if user_input['concrete strength'] == 'M20' else 0,
                'concrete M25': 1 if user_input['concrete strength'] == 'M25'else 0,
                'concrete M30': 1 if user_input['concrete strength'] == 'M30' else 0
                }

    features = pd.DataFrame(raw_data, index=[0])

    return features


def inference(dict):
    sc_x = joblib.load('scaler_x')
    sc_y = joblib.load('scaler_y')
    user_input = user_input_features(dict)
    features = torch.tensor(sc_x.fit_transform(
        user_input), dtype=torch.float32)
    with torch.no_grad():
        output = model(features)
        output = output.numpy().reshape([-1, 1])

        data = {
            'Steel area (cm^2)': output[0],
            'Depth (cm)': output[1],
            'Width (cm)': output[2],
            'Cost per meter (₦/m)': output[3],
            'Moment capacity (kNmm)': output[4]
        }
    predicted = pd.DataFrame(data, index=[0])
    predicted[predicted.columns] = sc_y.inverse_transform(predicted)

    return predicted


st.subheader('Model Output')
if generate_button:
    prediction = inference(user_input)
    st.write(prediction)
