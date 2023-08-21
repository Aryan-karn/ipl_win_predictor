import pickle

import pandas as pd
import streamlit as st
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# import sklearn
# from sklearn.linear_model import LogisticRegression
def stylish_probability_bar(label1, label2, probability1, probability2):
    st.write(f'<div style="position: relative; width: 100%; height: 10px; border: 1px solid #ddd;">\
                <div style="position: absolute; width: {probability1*100:.2f}%; height: 100%; background-color: green;"></div>\
                <div style="position: absolute; width: {probability2*100:.2f}%; height: 100%; background-color: red; left: {probability1*100:.2f}%"></div>\
                <div style="position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-weight: bold;padding-top: 40px;">\
                    <div>{label1}</div>\
                    <div style="margin-left: auto; margin-right: auto;">vs</div>\
                    <div>{label2}</div>\
                </div>\
              </div>', unsafe_allow_html=True)

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl','rb'))
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)


with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    st.table(input_df)
    # input_df.to_csv('input.csv')
    result = pipe.predict_proba(input_df)
    print(result)
    st.text(result)
    prediction_probabilities = result[0]  # Extract the probabilities from the result

    # Create a stylish progress bar for each team's probability
    stylish_probability_bar(bowling_team, batting_team, prediction_probabilities[0], prediction_probabilities[1])






