import pandas as pd
import streamlit as st
import pickle

st.title('Model Deployment: Linear Regression')
st.sidebar.header('User Input Parameters')

def user_input_features():
    avg_session =st.sidebar.text_input('Avg Session Length')
    app_time =st.sidebar.text_input('Time on App')
    web_time =st.sidebar.text_input('Time on Website')
    mem_len =st.sidebar.text_input('Length of Membership')
    try:
        avg_session = float(avg_session)
        app_time = float(app_time)
        web_time = float( web_time)
        mem_len = float(mem_len)
    except ValueError:
        st.error('Please enter numeric values for features.')
        return
    data = {'Avg Session Length': avg_session,
            'Time on App': app_time,
             'Time on Website': web_time,
            'Length of Membership': mem_len}
    features = pd.DataFrame(data,index = [0])
    return features
df =user_input_features()
st.subheader('User Input Parameters')
st.write(df) 
# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
#prediction
prediction = model.predict(df)
#output
st.subheader('Prediction Result')
st.write(prediction)
 
        

