import streamlit as st
import pandas as pd
import torch
import joblib
from model import Net
from sklearn.preprocessing import MinMaxScaler

model_path = 'designer_state.pth'
sc_x = joblib.load('scaler_x')
sc_y = joblib.load('scaler_y')

def load_model(file_path):
    """function to load saved state of model"""
    model_class = Net()
    model_class.load_state_dict(torch.load(file_path))
    model_class.eval()

    return model_class


# Load saved model
model = load_model(model_path)


def user_input_features(input_dict):
    raw_data = {
        'span': input_dict['span'],
        'ultimate load': input_dict['ultimate load'],
        'steel_Fe_215': 1 if input_dict['steel grade'] == 'Fe-215' else 0,
        'steel_Fe_415': 1 if input_dict['steel grade'] == 'Fe-415' else 0,
        'steel_Fe_500': 1 if input_dict['steel grade'] == 'Fe-500' else 0,
        'concrete M15': 1 if input_dict['concrete strength'] == 'M15' else 0,
        'concrete M20': 1 if input_dict['concrete strength'] == 'M20' else 0,
        'concrete M25': 1 if input_dict['concrete strength'] == 'M25'else 0,
        'concrete M30': 1 if input_dict['concrete strength'] == 'M30' else 0}

    features = pd.DataFrame(raw_data, index=[0])

    features[['span', 'ultimate load']] = sc_x.transform(features[['span', 'ultimate load']])

    return features


def inference(df):
    input_data = torch.tensor(df.values, dtype=torch.float32)
    
    loader = torch.utils.data.DataLoader(input_data)
    with torch.no_grad():
        for features in loader:
            output = model(features)
            output = output.numpy().reshape([-1, 1])

            data = {
            'Steel area (cm^2)': output[0],
            'Depth (cm)': output[1],
            'Width (cm)': output[2],
            'Cost per meter (₦/m)': output[3],
            'Moment capacity (kNm)': output[4]
            }

            predicted = pd.DataFrame(data, index=[0])
            predicted[predicted.columns] = sc_y.inverse_transform(predicted)

            return predicted


st.title('AI Beam Designer')
st.sidebar.header('User Input Parameters')

# input variables
def variables():
    input_dict = {
    'span': st.sidebar.number_input('Span (m)'),
    'ultimate load': st.sidebar.number_input('Ultimate load (kN/m)'),
    'steel grade': st.sidebar.selectbox('steel grade', ['Fe-215', 'Fe-415', 'Fe-500']),
    'concrete strength': st.sidebar.selectbox('Concrete Strength', ['M15', 'M20', 'M25', 'M30'])}

    return input_dict


st.subheader('Model Input')
input_dict = variables()
st.write(pd.DataFrame(input_dict, index=[0]))
    
generate_button = st.sidebar.button('generate initial design parameters')

st.subheader('Model Output')
if generate_button:
    user_input = user_input_features(input_dict)
    prediction = inference(user_input)
    st.write(prediction)
