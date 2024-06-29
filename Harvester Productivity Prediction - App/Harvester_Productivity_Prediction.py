#----------------------------------------------------------------#
import os, pickle
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
#----------------------------------------------------------------#


#----------------------------------------------------------------#
st.set_page_config(
    page_title = 'Harvester Productivity Prediction - 2024',
    layout = 'wide'
)

hide_streamlit_style = """
<style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html = True)
#----------------------------------------------------------------#


#----------------------------------------------------------------#
st.subheader(':blue[Feature adjustments]')

#1
with st.container():
    st.write('---')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        weather_data = st.radio(
            'Use weather data', 
            options = ['Yes', 'No'], horizontal = True
        )

    with col2:
        average_individual_tree_volume = st.radio(
            '**Average Individual Tree Volume** (m³)', options = ['0.15 < x ≤ 0.20', '0.20 < x ≤ 0.25', '0.25 < x ≤ 0.30'], horizontal = False
        )

    with col3:
        forest_age = st.radio(
        '**Forest age** (years)', options = [6, 7, 8, 9], horizontal = False
        )

    with col4:
        stand_density = st.radio(
            '**Stand density** (unit/ha)', options = ['700 < x ≤ 900', '900 < x ≤ 1200', '1200 < x ≤ 1350'], horizontal = False
        )



#2
if weather_data == 'Yes':

    with st.container():
        st.write('---')
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            dew_point_temperature_mean = st.radio(
            '**Dew point temperature** (°C)', options = ['0 < x ≤ 10', '10 < x ≤ 15', '15 < x ≤ 20', '20 < x ≤ 30'], horizontal = False
            )

        with col2:
            gusts_of_wind_mean = st.radio(
            '**Gust of wind** (m/s)', options = ['0 < x ≤ 3', '3 < x ≤ 4', '4 < x ≤ 7'], horizontal = False
            )

        with col3:
            relativy_air_humidity_mean = st.radio(
            '**Relativy air humidity** (%)', options = ['10 < x ≤ 30', '30 < x ≤ 60', '60 < x ≤ 80', '80 < x ≤ 99'], horizontal = False
            )

        with col4:
            atmospheric_pressure_mean = st.radio(
            '**Atmospheric pressure** (mbar)', options = ['950 < x ≤ 960', '960 < x ≤ 965', '965 < x ≤ 975'], horizontal = False
            )



#3
with st.container():
    st.write('---')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        work_shift = st.radio(
            '**Work shift**',
            options = ['morning - afternoon', 'afternoon - nigth', 'night - morning'], horizontal = False
        )

    with col2:
        operator_experience = st.radio(
        '**Operator experience** (years)', options = ['0.1 < x ≤ 1', '1 < x ≤ 2.5', '2.5 < x ≤ 15', '15 < x ≤ 40'], horizontal = False
        )

    with col3:
        material = st.radio(
            '**Timber assortment**',
            options = ['log with bark', 'log to energy', 'log without bark'], horizontal = False
        )

    with col4:
        working_hours = st.slider(
        '**Working hours**', min_value = 1.0, step = 0.25, max_value = 8.0, value = 5.0, format = '%f'
        )

st.write('---')

#----------------------------------------------------------------#


if weather_data == 'Yes':
    ######################
    if material == 'log with bark':
        material = 0
    if material == 'log to energy':
        material = 1
    if material == 'log without bark':
        material = 2
    ######################

    ######################
    if work_shift == 'morning - afternoon':
        work_shift = 0
    if work_shift == 'afternoon - nigth':
        work_shift = 1
    if work_shift == 'night - morning':
        work_shift = 2
    ######################

    ######################
    if operator_experience == '0.1 < x ≤ 1':
        operator_experience = 0.5
    if operator_experience == '1 < x ≤ 2.5':
        operator_experience = 2.5
    if operator_experience == '2.5 < x ≤ 15':
        operator_experience = 3.5
    if operator_experience == '15 < x ≤ 40':
        operator_experience = 21.0
    ######################

    ######################
    if average_individual_tree_volume == '0.15 < x ≤ 0.20':
        average_individual_tree_volume = 0.19
    if average_individual_tree_volume == '0.20 < x ≤ 0.25':
        average_individual_tree_volume = 0.24
    if average_individual_tree_volume == '0.25 < x ≤ 0.30':
        average_individual_tree_volume = 0.28
    ######################

    ######################
    if stand_density == '700 < x ≤ 900':
        stand_density = 800
    if stand_density == '900 < x ≤ 1200':
        stand_density = 900
    if stand_density == '1200 < x ≤ 1350':
        stand_density = 1200
    ######################

    ######################
    if dew_point_temperature_mean == '0 < x ≤ 10':
        dew_point_temperature_mean = 10.0
    if dew_point_temperature_mean == '10 < x ≤ 15':
        dew_point_temperature_mean = 15.0
    if dew_point_temperature_mean == '15 < x ≤ 20':
        dew_point_temperature_mean = 19.5
    if dew_point_temperature_mean == '20 < x ≤ 30':
        dew_point_temperature_mean = 20.5
    ######################

    ######################
    if gusts_of_wind_mean == '0 < x ≤ 3':
        gusts_of_wind_mean = 0
    if gusts_of_wind_mean == '3 < x ≤ 4':
        gusts_of_wind_mean = 3
    if gusts_of_wind_mean == '4 < x ≤ 7':
        gusts_of_wind_mean = 4
    ######################

    ######################
    if relativy_air_humidity_mean == '10 < x ≤ 30':
        relativy_air_humidity_mean = 30.0
    if relativy_air_humidity_mean == '30 < x ≤ 60':
        relativy_air_humidity_mean = 55.0
    if relativy_air_humidity_mean == '60 < x ≤ 80':
        relativy_air_humidity_mean = 79.0
    if relativy_air_humidity_mean == '80 < x ≤ 99':
        relativy_air_humidity_mean = 85.0
    ######################

    ######################
    if atmospheric_pressure_mean == '950 < x ≤ 960':
        atmospheric_pressure_mean = 952
    if atmospheric_pressure_mean == '960 < x ≤ 965':
        atmospheric_pressure_mean = 964
    if atmospheric_pressure_mean == '965 < x ≤ 975':
        atmospheric_pressure_mean = 970
    ######################

    DF = pd.DataFrame({
        'material':[material],
        'work shift':[work_shift],

        'working hours':[working_hours],
        'operator experience':[operator_experience],
        'average individual tree volume':[average_individual_tree_volume],

        'forest age':[forest_age],
        'wind speed mean':[0.0],
        'dew point temperature mean':[dew_point_temperature_mean],

        'gusts of wind mean':[gusts_of_wind_mean], 
        'relativy air humidity mean':[relativy_air_humidity_mean],
        'stand density':[stand_density],
        'atmospheric pressure mean':[atmospheric_pressure_mean],
    })

    model = pickle.load(open('/home/rodrigo/Dropbox/DABL/download/harvester_productivity_prediction_final_model_timber_harvest_data_combined_with_weather_data.pkl', 'rb'))



if weather_data == 'No':

    ######################
    if material == 'log with bark':
        material = 0
    if material == 'log to energy':
        material = 1
    if material == 'log without bark':
        material = 2
    ######################

    ######################
    if work_shift == 'morning - afternoon':
        work_shift = 0
    if work_shift == 'afternoon - nigth':
        work_shift = 1
    if work_shift == 'night - morning':
        work_shift = 2
    ######################

    ######################
    if operator_experience == '0.1 < x ≤ 1':
        operator_experience = 0.5
    if operator_experience == '1 < x ≤ 2.5':
        operator_experience = 2.5
    if operator_experience == '2.5 < x ≤ 15':
        operator_experience = 3.5
    if operator_experience == '15 < x ≤ 40':
        operator_experience = 21.0
    ######################

    ######################
    if average_individual_tree_volume == '0.15 < x ≤ 0.20':
        average_individual_tree_volume = 0.19
    if average_individual_tree_volume == '0.20 < x ≤ 0.25':
        average_individual_tree_volume = 0.24
    if average_individual_tree_volume == '0.25 < x ≤ 0.30':
        average_individual_tree_volume = 0.28
    ######################

    ######################
    if stand_density == '700 < x ≤ 900':
        stand_density = 800
    if stand_density == '900 < x ≤ 1200':
        stand_density = 900
    if stand_density == '1200 < x ≤ 1350':
        stand_density = 1200
    ######################

    DF = pd.DataFrame({
        'material':[material],
        'work shift':[work_shift],

        'working hours':[working_hours],
        'average individual tree volume':[average_individual_tree_volume],
        'operator experience':[operator_experience],

        'forest age':[forest_age],
        'stand density':[stand_density],
    })

    model = pickle.load(open('/home/rodrigo/Dropbox/DABL/download/harvester_productivity_prediction_final_model_only_timber_harvest_data.pkl', 'rb'))

#----------------------------------------------------------------#


result = model.predict(DF)
st.subheader(':blue[Results]')
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.write('Productivity: **:orange[' + str(round(list(result)[0], 1)) + ']**')

st.write('---')
#----------------------------------------------------------------#
