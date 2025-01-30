#----------------------------------------------------------------#
import os, pickle, itertools
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#----------------------------------------------------------------#


#----------------------------------------------------------------#
st.set_page_config(
    page_title = 'Harvester Maintenance Prediction - 2024',
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
#≥≤

#----------------------------------------------------------------#
st.subheader(':blue[Feature adjustments]')

model = pickle.load(open('/home/rodrigo/Random Forest Default.pkl', 'rb'))
encoder_class = pickle.load(open('/home/rodrigo/encoder_class.pkl', 'rb'))
status_ratio = []
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#1
with st.container():
    st.write('---')
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

    with col1:
        a1_m1_df1 = 41; a1_m2_df1 = 94; a1_s_df1 = 5
        a1_p1_df1 = list(np.arange(a1_m1_df1, a1_m2_df1, (a1_m2_df1 - a1_m1_df1)/a1_s_df1)) + [a1_m2_df1]
        for i in range(0, len(a1_p1_df1) - 1):
            a1_p1_df1[i] = str(int(a1_p1_df1[i])) + ' ≤ x < ' + str(int(a1_p1_df1[i + 1]))
        a1_p2_df1 = list(np.arange(a1_m1_df1, a1_m2_df1, (a1_m2_df1 - a1_m1_df1)/a1_s_df1))
        for i in range(0, len(a1_p2_df1)):
            a1_p2_df1[i] = round(a1_p2_df1[i] + ((a1_m2_df1 - a1_m1_df1)/a1_s_df1)/2, 1)
        
        a1_df1 = st.radio(
            '**ECU temp** (°C)', options = [a1_p1_df1[0], a1_p1_df1[1], a1_p1_df1[2], a1_p1_df1[3], a1_p1_df1[4]], key = 'a1_df1'
        )
        engine_ecu_temperature_df1 = 0
        if a1_df1 == a1_p1_df1[0]:
            engine_ecu_temperature_df1 = a1_p2_df1[0]
        elif a1_df1 == a1_p1_df1[1]:
            engine_ecu_temperature_df1 = a1_p2_df1[1]
        elif a1_df1 == a1_p1_df1[2]:
            engine_ecu_temperature_df1 = a1_p2_df1[2]
        elif a1_df1 == a1_p1_df1[3]:
            engine_ecu_temperature_df1 = a1_p2_df1[3]
        elif a1_df1 == a1_p1_df1[4]:
            engine_ecu_temperature_df1 = a1_p2_df1[4]
    
    with col2:
        a2_m1_df1 = 3; a2_m2_df1 = 37; a2_s_df1 = 5
        a2_p1_df1 = list(np.arange(a2_m1_df1, a2_m2_df1, (a2_m2_df1 - a2_m1_df1)/a2_s_df1)) + [a2_m2_df1]
        for i in range(0, len(a2_p1_df1) - 1):
            a2_p1_df1[i] = str(int(a2_p1_df1[i])) + ' ≤ x < ' + str(int(a2_p1_df1[i + 1]))
        a2_p2_df1 = list(np.arange(a2_m1_df1, a2_m2_df1, (a2_m2_df1 - a2_m1_df1)/a2_s_df1))
        for i in range(0, len(a2_p2_df1)):
            a2_p2_df1[i] = round(a2_p2_df1[i] + ((a2_m2_df1 - a2_m1_df1)/a2_s_df1)/2, 1)
        
        a2_df1 = st.radio(
            '**fuel rate** (L/h)', options = [a2_p1_df1[0], a2_p1_df1[1], a2_p1_df1[2], a2_p1_df1[3], a2_p1_df1[4]], key = 'a2_df1'
        )
        engine_fuel_rate_df1 = 0
        if a2_df1 == a2_p1_df1[0]:
            engine_fuel_rate_df1 = a2_p2_df1[0]
        elif a2_df1 == a2_p1_df1[1]:
            engine_fuel_rate_df1 = a2_p2_df1[1]
        elif a2_df1 == a2_p1_df1[2]:
            engine_fuel_rate_df1 = a2_p2_df1[2]
        elif a2_df1 == a2_p1_df1[3]:
            engine_fuel_rate_df1 = a2_p2_df1[3]
        elif a2_df1 == a2_p1_df1[4]:
            engine_fuel_rate_df1 = a2_p2_df1[4]
    
    with col3:
        a3_m1_df1 = 4; a3_m2_df1 = 80; a3_s_df1 = 5
        a3_p1_df1 = list(np.arange(a3_m1_df1, a3_m2_df1, (a3_m2_df1 - a3_m1_df1)/a3_s_df1)) + [a3_m2_df1]
        for i in range(0, len(a3_p1_df1) - 1):
            a3_p1_df1[i] = str(int(a3_p1_df1[i])) + ' ≤ x < ' + str(int(a3_p1_df1[i + 1]))
        a3_p2_df1 = list(np.arange(a3_m1_df1, a3_m2_df1, (a3_m2_df1 - a3_m1_df1)/a3_s_df1))
        for i in range(0, len(a3_p2_df1)):
            a3_p2_df1[i] = round(a3_p2_df1[i] + ((a3_m2_df1 - a3_m1_df1)/a3_s_df1)/2, 1)
        
        a3_df1 = st.radio(
            '**IMP** (bar)', options = [a3_p1_df1[0], a3_p1_df1[1], a3_p1_df1[2], a3_p1_df1[3], a3_p1_df1[4]], key = 'a3_df1'
        )
        engine_intake_manifold_1_pressure_df1 = 0
        if a3_df1 == a3_p1_df1[0]:
            engine_intake_manifold_1_pressure_df1 = a3_p2_df1[0]
        elif a3_df1 == a3_p1_df1[1]:
            engine_intake_manifold_1_pressure_df1 = a3_p2_df1[1]
        elif a3_df1 == a3_p1_df1[2]:
            engine_intake_manifold_1_pressure_df1 = a3_p2_df1[2]
        elif a3_df1 == a3_p1_df1[3]:
            engine_intake_manifold_1_pressure_df1 = a3_p2_df1[3]
        elif a3_df1 == a3_p1_df1[4]:
            engine_intake_manifold_1_pressure_df1 = a3_p2_df1[4]
            
    with col4:
        a4_m1_df1 = 192; a4_m2_df1 = 348; a4_s_df1 = 5
        a4_p1_df1 = list(np.arange(a4_m1_df1, a4_m2_df1, (a4_m2_df1 - a4_m1_df1)/a4_s_df1)) + [a4_m2_df1]
        for i in range(0, len(a4_p1_df1) - 1):
            a4_p1_df1[i] = str(int(a4_p1_df1[i])) + ' ≤ x < ' + str(int(a4_p1_df1[i + 1]))
        a4_p2_df1 = list(np.arange(a4_m1_df1, a4_m2_df1, (a4_m2_df1 - a4_m1_df1)/a4_s_df1))
        for i in range(0, len(a4_p2_df1)):
            a4_p2_df1[i] = round(a4_p2_df1[i] + ((a4_m2_df1 - a4_m1_df1)/a4_s_df1)/2, 1)
        
        a4_df1 = st.radio(
            '**oil pressure** (bar)', options = [a4_p1_df1[0], a4_p1_df1[1], a4_p1_df1[2], a4_p1_df1[3], a4_p1_df1[4]], key = 'a4_df1'
        )
        engine_oil_pressure_df1 = 0
        if a4_df1 == a4_p1_df1[0]:
            engine_oil_pressure_df1 = a4_p2_df1[0]
        elif a4_df1 == a4_p1_df1[1]:
            engine_oil_pressure_df1 = a4_p2_df1[1]
        elif a4_df1 == a4_p1_df1[2]:
            engine_oil_pressure_df1 = a4_p2_df1[2]
        elif a4_df1 == a4_p1_df1[3]:
            engine_oil_pressure_df1 = a4_p2_df1[3]
        elif a4_df1 == a4_p1_df1[4]:
            engine_oil_pressure_df1 = a4_p2_df1[4]
    
    with col5:
        a5_m1_df1 = 83; a5_m2_df1 = 101; a5_s_df1 = 5
        a5_p1_df1 = list(np.arange(a5_m1_df1, a5_m2_df1, (a5_m2_df1 - a5_m1_df1)/a5_s_df1)) + [a5_m2_df1]
        for i in range(0, len(a5_p1_df1) - 1):
            a5_p1_df1[i] = str(int(a5_p1_df1[i])) + ' ≤ x < ' + str(int(a5_p1_df1[i + 1]))
        a5_p2_df1 = list(np.arange(a5_m1_df1, a5_m2_df1, (a5_m2_df1 - a5_m1_df1)/a5_s_df1))
        for i in range(0, len(a5_p2_df1)):
            a5_p2_df1[i] = round(a5_p2_df1[i] + ((a5_m2_df1 - a5_m1_df1)/a5_s_df1)/2, 1)
        
        a5_df1 = st.radio(
            '**oil temperature** (°C)', options = [a5_p1_df1[0], a5_p1_df1[1], a5_p1_df1[2], a5_p1_df1[3], a5_p1_df1[4]], key = 'a5_df1'
        )
        engine_oil_temperature_1_df1 = 0
        if a5_df1 == a5_p1_df1[0]:
            engine_oil_temperature_1_df1 = a5_p2_df1[0]
        elif a5_df1 == a5_p1_df1[1]:
            engine_oil_temperature_1_df1 = a5_p2_df1[1]
        elif a5_df1 == a5_p1_df1[2]:
            engine_oil_temperature_1_df1 = a5_p2_df1[2]
        elif a5_df1 == a5_p1_df1[3]:
            engine_oil_temperature_1_df1 = a5_p2_df1[3]
        elif a5_df1 == a5_p1_df1[4]:
            engine_oil_temperature_1_df1 = a5_p2_df1[4]
    
    with col6:
        a6_m1_df1 = 26.1; a6_m2_df1 = 28; a6_s_df1 = 5
        a6_p1_df1 = list(np.arange(a6_m1_df1, a6_m2_df1, (a6_m2_df1 - a6_m1_df1)/a6_s_df1)) + [a6_m2_df1]
        for i in range(0, len(a6_p1_df1) - 1):
            a6_p1_df1[i] = str(int(a6_p1_df1[i])) + ' ≤ x < ' + str(int(a6_p1_df1[i + 1]))
        a6_p2_df1 = list(np.arange(a6_m1_df1, a6_m2_df1, (a6_m2_df1 - a6_m1_df1)/a6_s_df1))
        for i in range(0, len(a6_p2_df1)):
            a6_p2_df1[i] = round(a6_p2_df1[i] + ((a6_m2_df1 - a6_m1_df1)/a6_s_df1)/2, 1)
        
        a6_df1 = st.radio(
            '**battery potential** (V)', options = [a6_p1_df1[0], a6_p1_df1[1], a6_p1_df1[2], a6_p1_df1[3], a6_p1_df1[4]], key = 'a6_df1'
        )
        keyswitch_battery_potential_df1 = 0
        if a6_df1 == a6_p1_df1[0]:
            keyswitch_battery_potential_df1 = a6_p2_df1[0]
        elif a6_df1 == a6_p1_df1[1]:
            keyswitch_battery_potential_df1 = a6_p2_df1[1]
        elif a6_df1 == a6_p1_df1[2]:
            keyswitch_battery_potential_df1 = a6_p2_df1[2]
        elif a6_df1 == a6_p1_df1[3]:
            keyswitch_battery_potential_df1 = a6_p2_df1[3]
        elif a6_df1 == a6_p1_df1[4]:
            keyswitch_battery_potential_df1 = a6_p2_df1[4]
    
    with col7:
        DF1 = pd.DataFrame({
            'keyswitch battery potential':[keyswitch_battery_potential_df1],
            'engine oil temperature 1':[engine_oil_temperature_1_df1],
            'engine oil pressure':[engine_oil_pressure_df1],
            'engine intake manifold 1 pressure':[engine_intake_manifold_1_pressure_df1],
            'engine fuel rate':[engine_fuel_rate_df1],
            'engine ecu temperature':[engine_ecu_temperature_df1]
        })
        
        result_df1 = model.predict(DF1)
        status_df1 = list(encoder_class.inverse_transform(np.array(int(result_df1)).ravel()))[0]
        
        st.subheader(':green[period 1]')
        st.write('Status: **:orange[' + status_df1 + ']**')
        status_ratio.append(status_df1)
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#2
with st.container():
    st.write('---')
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

    with col1:
        a1_m1_df2 = 41; a1_m2_df2 = 94; a1_s_df2 = 5
        a1_p1_df2 = list(np.arange(a1_m1_df2, a1_m2_df2, (a1_m2_df2 - a1_m1_df2)/a1_s_df2)) + [a1_m2_df2]
        for i in range(0, len(a1_p1_df2) - 1):
            a1_p1_df2[i] = str(int(a1_p1_df2[i])) + ' ≤ x < ' + str(int(a1_p1_df2[i + 1]))
        a1_p2_df2 = list(np.arange(a1_m1_df2, a1_m2_df2, (a1_m2_df2 - a1_m1_df2)/a1_s_df2))
        for i in range(0, len(a1_p2_df2)):
            a1_p2_df2[i] = round(a1_p2_df2[i] + ((a1_m2_df2 - a1_m1_df2)/a1_s_df2)/2, 1)
        
        a1_df2 = st.radio(
            '**ECU temp** (°C)', options = [a1_p1_df2[0], a1_p1_df2[1], a1_p1_df2[2], a1_p1_df2[3], a1_p1_df2[4]], key = 'a1_df2'
        )
        engine_ecu_temperature_df2 = 0
        if a1_df2 == a1_p1_df2[0]:
            engine_ecu_temperature_df2 = a1_p2_df2[0]
        elif a1_df2 == a1_p1_df2[1]:
            engine_ecu_temperature_df2 = a1_p2_df2[1]
        elif a1_df2 == a1_p1_df2[2]:
            engine_ecu_temperature_df2 = a1_p2_df2[2]
        elif a1_df2 == a1_p1_df2[3]:
            engine_ecu_temperature_df2 = a1_p2_df2[3]
        elif a1_df2 == a1_p1_df2[4]:
            engine_ecu_temperature_df2 = a1_p2_df2[4]
    
    with col2:
        a2_m1_df2 = 3; a2_m2_df2 = 37; a2_s_df2 = 5
        a2_p1_df2 = list(np.arange(a2_m1_df2, a2_m2_df2, (a2_m2_df2 - a2_m1_df2)/a2_s_df2)) + [a2_m2_df2]
        for i in range(0, len(a2_p1_df2) - 1):
            a2_p1_df2[i] = str(int(a2_p1_df2[i])) + ' ≤ x < ' + str(int(a2_p1_df2[i + 1]))
        a2_p2_df2 = list(np.arange(a2_m1_df2, a2_m2_df2, (a2_m2_df2 - a2_m1_df2)/a2_s_df2))
        for i in range(0, len(a2_p2_df2)):
            a2_p2_df2[i] = round(a2_p2_df2[i] + ((a2_m2_df2 - a2_m1_df2)/a2_s_df2)/2, 1)
        
        a2_df2 = st.radio(
            '**fuel rate** (L/h)', options = [a2_p1_df2[0], a2_p1_df2[1], a2_p1_df2[2], a2_p1_df2[3], a2_p1_df2[4]], key = 'a2_df2'
        )
        engine_fuel_rate_df2 = 0
        if a2_df2 == a2_p1_df2[0]:
            engine_fuel_rate_df2 = a2_p2_df2[0]
        elif a2_df2 == a2_p1_df2[1]:
            engine_fuel_rate_df2 = a2_p2_df2[1]
        elif a2_df2 == a2_p1_df2[2]:
            engine_fuel_rate_df2 = a2_p2_df2[2]
        elif a2_df2 == a2_p1_df2[3]:
            engine_fuel_rate_df2 = a2_p2_df2[3]
        elif a2_df2 == a2_p1_df2[4]:
            engine_fuel_rate_df2 = a2_p2_df2[4]
    
    with col3:
        a3_m1_df2 = 4; a3_m2_df2 = 80; a3_s_df2 = 5
        a3_p1_df2 = list(np.arange(a3_m1_df2, a3_m2_df2, (a3_m2_df2 - a3_m1_df2)/a3_s_df2)) + [a3_m2_df2]
        for i in range(0, len(a3_p1_df2) - 1):
            a3_p1_df2[i] = str(int(a3_p1_df2[i])) + ' ≤ x < ' + str(int(a3_p1_df2[i + 1]))
        a3_p2_df2 = list(np.arange(a3_m1_df2, a3_m2_df2, (a3_m2_df2 - a3_m1_df2)/a3_s_df2))
        for i in range(0, len(a3_p2_df2)):
            a3_p2_df2[i] = round(a3_p2_df2[i] + ((a3_m2_df2 - a3_m1_df2)/a3_s_df2)/2, 1)
        
        a3_df2 = st.radio(
            '**IMP** (bar)', options = [a3_p1_df2[0], a3_p1_df2[1], a3_p1_df2[2], a3_p1_df2[3], a3_p1_df2[4]], key = 'a3_df2'
        )
        engine_intake_manifold_1_pressure_df2 = 0
        if a3_df2 == a3_p1_df2[0]:
            engine_intake_manifold_1_pressure_df2 = a3_p2_df2[0]
        elif a3_df2 == a3_p1_df2[1]:
            engine_intake_manifold_1_pressure_df2 = a3_p2_df2[1]
        elif a3_df2 == a3_p1_df2[2]:
            engine_intake_manifold_1_pressure_df2 = a3_p2_df2[2]
        elif a3_df2 == a3_p1_df2[3]:
            engine_intake_manifold_1_pressure_df2 = a3_p2_df2[3]
        elif a3_df2 == a3_p1_df2[4]:
            engine_intake_manifold_1_pressure_df2 = a3_p2_df2[4]
            
    with col4:
        a4_m1_df2 = 192; a4_m2_df2 = 348; a4_s_df2 = 5
        a4_p1_df2 = list(np.arange(a4_m1_df2, a4_m2_df2, (a4_m2_df2 - a4_m1_df2)/a4_s_df2)) + [a4_m2_df2]
        for i in range(0, len(a4_p1_df2) - 1):
            a4_p1_df2[i] = str(int(a4_p1_df2[i])) + ' ≤ x < ' + str(int(a4_p1_df2[i + 1]))
        a4_p2_df2 = list(np.arange(a4_m1_df2, a4_m2_df2, (a4_m2_df2 - a4_m1_df2)/a4_s_df2))
        for i in range(0, len(a4_p2_df2)):
            a4_p2_df2[i] = round(a4_p2_df2[i] + ((a4_m2_df2 - a4_m1_df2)/a4_s_df2)/2, 1)
        
        a4_df2 = st.radio(
            '**oil pressure** (bar)', options = [a4_p1_df2[0], a4_p1_df2[1], a4_p1_df2[2], a4_p1_df2[3], a4_p1_df2[4]], key = 'a4_df2'
        )
        engine_oil_pressure_df2 = 0
        if a4_df2 == a4_p1_df2[0]:
            engine_oil_pressure_df2 = a4_p2_df2[0]
        elif a4_df2 == a4_p1_df2[1]:
            engine_oil_pressure_df2 = a4_p2_df2[1]
        elif a4_df2 == a4_p1_df2[2]:
            engine_oil_pressure_df2 = a4_p2_df2[2]
        elif a4_df2 == a4_p1_df2[3]:
            engine_oil_pressure_df2 = a4_p2_df2[3]
        elif a4_df2 == a4_p1_df2[4]:
            engine_oil_pressure_df2 = a4_p2_df2[4]
    
    with col5:
        a5_m1_df2 = 83; a5_m2_df2 = 101; a5_s_df2 = 5
        a5_p1_df2 = list(np.arange(a5_m1_df2, a5_m2_df2, (a5_m2_df2 - a5_m1_df2)/a5_s_df2)) + [a5_m2_df2]
        for i in range(0, len(a5_p1_df2) - 1):
            a5_p1_df2[i] = str(int(a5_p1_df2[i])) + ' ≤ x < ' + str(int(a5_p1_df2[i + 1]))
        a5_p2_df2 = list(np.arange(a5_m1_df2, a5_m2_df2, (a5_m2_df2 - a5_m1_df2)/a5_s_df2))
        for i in range(0, len(a5_p2_df2)):
            a5_p2_df2[i] = round(a5_p2_df2[i] + ((a5_m2_df2 - a5_m1_df2)/a5_s_df2)/2, 1)
        
        a5_df2 = st.radio(
            '**oil temperature** (°C)', options = [a5_p1_df2[0], a5_p1_df2[1], a5_p1_df2[2], a5_p1_df2[3], a5_p1_df2[4]], key = 'a5_df2'
        )
        engine_oil_temperature_1_df2 = 0
        if a5_df2 == a5_p1_df2[0]:
            engine_oil_temperature_1_df2 = a5_p2_df2[0]
        elif a5_df2 == a5_p1_df2[1]:
            engine_oil_temperature_1_df2 = a5_p2_df2[1]
        elif a5_df2 == a5_p1_df2[2]:
            engine_oil_temperature_1_df2 = a5_p2_df2[2]
        elif a5_df2 == a5_p1_df2[3]:
            engine_oil_temperature_1_df2 = a5_p2_df2[3]
        elif a5_df2 == a5_p1_df2[4]:
            engine_oil_temperature_1_df2 = a5_p2_df2[4]
    
    with col6:
        a6_m1_df2 = 26.1; a6_m2_df2 = 28; a6_s_df2 = 5
        a6_p1_df2 = list(np.arange(a6_m1_df2, a6_m2_df2, (a6_m2_df2 - a6_m1_df2)/a6_s_df2)) + [a6_m2_df2]
        for i in range(0, len(a6_p1_df2) - 1):
            a6_p1_df2[i] = str(int(a6_p1_df2[i])) + ' ≤ x < ' + str(int(a6_p1_df2[i + 1]))
        a6_p2_df2 = list(np.arange(a6_m1_df2, a6_m2_df2, (a6_m2_df2 - a6_m1_df2)/a6_s_df2))
        for i in range(0, len(a6_p2_df2)):
            a6_p2_df2[i] = round(a6_p2_df2[i] + ((a6_m2_df2 - a6_m1_df2)/a6_s_df2)/2, 1)
        
        a6_df2 = st.radio(
            '**battery potential** (V)', options = [a6_p1_df2[0], a6_p1_df2[1], a6_p1_df2[2], a6_p1_df2[3], a6_p1_df2[4]], key = 'a6_df2'
        )
        keyswitch_battery_potential_df2 = 0
        if a6_df2 == a6_p1_df2[0]:
            keyswitch_battery_potential_df2 = a6_p2_df2[0]
        elif a6_df2 == a6_p1_df2[1]:
            keyswitch_battery_potential_df2 = a6_p2_df2[1]
        elif a6_df2 == a6_p1_df2[2]:
            keyswitch_battery_potential_df2 = a6_p2_df2[2]
        elif a6_df2 == a6_p1_df2[3]:
            keyswitch_battery_potential_df2 = a6_p2_df2[3]
        elif a6_df2 == a6_p1_df2[4]:
            keyswitch_battery_potential_df2 = a6_p2_df2[4]
    
    with col7:
        DF2 = pd.DataFrame({
            'keyswitch battery potential':[keyswitch_battery_potential_df2],
            'engine oil temperature 1':[engine_oil_temperature_1_df2],
            'engine oil pressure':[engine_oil_pressure_df2],
            'engine intake manifold 1 pressure':[engine_intake_manifold_1_pressure_df2],
            'engine fuel rate':[engine_fuel_rate_df2],
            'engine ecu temperature':[engine_ecu_temperature_df2]
        })
        
        result_df2 = model.predict(DF2)
        status_df2 = list(encoder_class.inverse_transform(np.array(int(result_df2)).ravel()))[0]
        
        st.subheader(':green[period 2]')
        st.write('Status: **:orange[' + status_df2 + ']**')
        status_ratio.append(status_df2)
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#3
with st.container():
    st.write('---')
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

    with col1:
        a1_m1_df3 = 41; a1_m2_df3 = 94; a1_s_df3 = 5
        a1_p1_df3 = list(np.arange(a1_m1_df3, a1_m2_df3, (a1_m2_df3 - a1_m1_df3)/a1_s_df3)) + [a1_m2_df3]
        for i in range(0, len(a1_p1_df3) - 1):
            a1_p1_df3[i] = str(int(a1_p1_df3[i])) + ' ≤ x < ' + str(int(a1_p1_df3[i + 1]))
        a1_p2_df3 = list(np.arange(a1_m1_df3, a1_m2_df3, (a1_m2_df3 - a1_m1_df3)/a1_s_df3))
        for i in range(0, len(a1_p2_df3)):
            a1_p2_df3[i] = round(a1_p2_df3[i] + ((a1_m2_df3 - a1_m1_df3)/a1_s_df3)/2, 1)
        
        a1_df3 = st.radio(
            '**ECU temp** (°C)', options = [a1_p1_df3[0], a1_p1_df3[1], a1_p1_df3[2], a1_p1_df3[3], a1_p1_df3[4]], key = 'a1_df3'
        )
        engine_ecu_temperature_df3 = 0
        if a1_df3 == a1_p1_df3[0]:
            engine_ecu_temperature_df3 = a1_p2_df3[0]
        elif a1_df3 == a1_p1_df3[1]:
            engine_ecu_temperature_df3 = a1_p2_df3[1]
        elif a1_df3 == a1_p1_df3[2]:
            engine_ecu_temperature_df3 = a1_p2_df3[2]
        elif a1_df3 == a1_p1_df3[3]:
            engine_ecu_temperature_df3 = a1_p2_df3[3]
        elif a1_df3 == a1_p1_df3[4]:
            engine_ecu_temperature_df3 = a1_p2_df3[4]
    
    with col2:
        a2_m1_df3 = 3; a2_m2_df3 = 37; a2_s_df3 = 5
        a2_p1_df3 = list(np.arange(a2_m1_df3, a2_m2_df3, (a2_m2_df3 - a2_m1_df3)/a2_s_df3)) + [a2_m2_df3]
        for i in range(0, len(a2_p1_df3) - 1):
            a2_p1_df3[i] = str(int(a2_p1_df3[i])) + ' ≤ x < ' + str(int(a2_p1_df3[i + 1]))
        a2_p2_df3 = list(np.arange(a2_m1_df3, a2_m2_df3, (a2_m2_df3 - a2_m1_df3)/a2_s_df3))
        for i in range(0, len(a2_p2_df3)):
            a2_p2_df3[i] = round(a2_p2_df3[i] + ((a2_m2_df3 - a2_m1_df3)/a2_s_df3)/2, 1)
        
        a2_df3 = st.radio(
            '**fuel rate** (L/h)', options = [a2_p1_df3[0], a2_p1_df3[1], a2_p1_df3[2], a2_p1_df3[3], a2_p1_df3[4]], key = 'a2_df3'
        )
        engine_fuel_rate_df3 = 0
        if a2_df3 == a2_p1_df3[0]:
            engine_fuel_rate_df3 = a2_p2_df3[0]
        elif a2_df3 == a2_p1_df3[1]:
            engine_fuel_rate_df3 = a2_p2_df3[1]
        elif a2_df3 == a2_p1_df3[2]:
            engine_fuel_rate_df3 = a2_p2_df3[2]
        elif a2_df3 == a2_p1_df3[3]:
            engine_fuel_rate_df3 = a2_p2_df3[3]
        elif a2_df3 == a2_p1_df3[4]:
            engine_fuel_rate_df3 = a2_p2_df3[4]
    
    with col3:
        a3_m1_df3 = 4; a3_m2_df3 = 80; a3_s_df3 = 5
        a3_p1_df3 = list(np.arange(a3_m1_df3, a3_m2_df3, (a3_m2_df3 - a3_m1_df3)/a3_s_df3)) + [a3_m2_df3]
        for i in range(0, len(a3_p1_df3) - 1):
            a3_p1_df3[i] = str(int(a3_p1_df3[i])) + ' ≤ x < ' + str(int(a3_p1_df3[i + 1]))
        a3_p2_df3 = list(np.arange(a3_m1_df3, a3_m2_df3, (a3_m2_df3 - a3_m1_df3)/a3_s_df3))
        for i in range(0, len(a3_p2_df3)):
            a3_p2_df3[i] = round(a3_p2_df3[i] + ((a3_m2_df3 - a3_m1_df3)/a3_s_df3)/2, 1)
        
        a3_df3 = st.radio(
            '**IMP** (bar)', options = [a3_p1_df3[0], a3_p1_df3[1], a3_p1_df3[2], a3_p1_df3[3], a3_p1_df3[4]], key = 'a3_df3'
        )
        engine_intake_manifold_1_pressure_df3 = 0
        if a3_df3 == a3_p1_df3[0]:
            engine_intake_manifold_1_pressure_df3 = a3_p2_df3[0]
        elif a3_df3 == a3_p1_df3[1]:
            engine_intake_manifold_1_pressure_df3 = a3_p2_df3[1]
        elif a3_df3 == a3_p1_df3[2]:
            engine_intake_manifold_1_pressure_df3 = a3_p2_df3[2]
        elif a3_df3 == a3_p1_df3[3]:
            engine_intake_manifold_1_pressure_df3 = a3_p2_df3[3]
        elif a3_df3 == a3_p1_df3[4]:
            engine_intake_manifold_1_pressure_df3 = a3_p2_df3[4]
            
    with col4:
        a4_m1_df3 = 192; a4_m2_df3 = 348; a4_s_df3 = 5
        a4_p1_df3 = list(np.arange(a4_m1_df3, a4_m2_df3, (a4_m2_df3 - a4_m1_df3)/a4_s_df3)) + [a4_m2_df3]
        for i in range(0, len(a4_p1_df3) - 1):
            a4_p1_df3[i] = str(int(a4_p1_df3[i])) + ' ≤ x < ' + str(int(a4_p1_df3[i + 1]))
        a4_p2_df3 = list(np.arange(a4_m1_df3, a4_m2_df3, (a4_m2_df3 - a4_m1_df3)/a4_s_df3))
        for i in range(0, len(a4_p2_df3)):
            a4_p2_df3[i] = round(a4_p2_df3[i] + ((a4_m2_df3 - a4_m1_df3)/a4_s_df3)/2, 1)
        
        a4_df3 = st.radio(
            '**oil pressure** (bar)', options = [a4_p1_df3[0], a4_p1_df3[1], a4_p1_df3[2], a4_p1_df3[3], a4_p1_df3[4]], key = 'a4_df3'
        )
        engine_oil_pressure_df3 = 0
        if a4_df3 == a4_p1_df3[0]:
            engine_oil_pressure_df3 = a4_p2_df3[0]
        elif a4_df3 == a4_p1_df3[1]:
            engine_oil_pressure_df3 = a4_p2_df3[1]
        elif a4_df3 == a4_p1_df3[2]:
            engine_oil_pressure_df3 = a4_p2_df3[2]
        elif a4_df3 == a4_p1_df3[3]:
            engine_oil_pressure_df3 = a4_p2_df3[3]
        elif a4_df3 == a4_p1_df3[4]:
            engine_oil_pressure_df3 = a4_p2_df3[4]
    
    with col5:
        a5_m1_df3 = 83; a5_m2_df3 = 101; a5_s_df3 = 5
        a5_p1_df3 = list(np.arange(a5_m1_df3, a5_m2_df3, (a5_m2_df3 - a5_m1_df3)/a5_s_df3)) + [a5_m2_df3]
        for i in range(0, len(a5_p1_df3) - 1):
            a5_p1_df3[i] = str(int(a5_p1_df3[i])) + ' ≤ x < ' + str(int(a5_p1_df3[i + 1]))
        a5_p2_df3 = list(np.arange(a5_m1_df3, a5_m2_df3, (a5_m2_df3 - a5_m1_df3)/a5_s_df3))
        for i in range(0, len(a5_p2_df3)):
            a5_p2_df3[i] = round(a5_p2_df3[i] + ((a5_m2_df3 - a5_m1_df3)/a5_s_df3)/2, 1)
        
        a5_df3 = st.radio(
            '**oil temperature** (°C)', options = [a5_p1_df3[0], a5_p1_df3[1], a5_p1_df3[2], a5_p1_df3[3], a5_p1_df3[4]], key = 'a5_df3'
        )
        engine_oil_temperature_1_df3 = 0
        if a5_df3 == a5_p1_df3[0]:
            engine_oil_temperature_1_df3 = a5_p2_df3[0]
        elif a5_df3 == a5_p1_df3[1]:
            engine_oil_temperature_1_df3 = a5_p2_df3[1]
        elif a5_df3 == a5_p1_df3[2]:
            engine_oil_temperature_1_df3 = a5_p2_df3[2]
        elif a5_df3 == a5_p1_df3[3]:
            engine_oil_temperature_1_df3 = a5_p2_df3[3]
        elif a5_df3 == a5_p1_df3[4]:
            engine_oil_temperature_1_df3 = a5_p2_df3[4]
    
    with col6:
        a6_m1_df3 = 26.1; a6_m2_df3 = 28; a6_s_df3 = 5
        a6_p1_df3 = list(np.arange(a6_m1_df3, a6_m2_df3, (a6_m2_df3 - a6_m1_df3)/a6_s_df3)) + [a6_m2_df3]
        for i in range(0, len(a6_p1_df3) - 1):
            a6_p1_df3[i] = str(int(a6_p1_df3[i])) + ' ≤ x < ' + str(int(a6_p1_df3[i + 1]))
        a6_p2_df3 = list(np.arange(a6_m1_df3, a6_m2_df3, (a6_m2_df3 - a6_m1_df3)/a6_s_df3))
        for i in range(0, len(a6_p2_df3)):
            a6_p2_df3[i] = round(a6_p2_df3[i] + ((a6_m2_df3 - a6_m1_df3)/a6_s_df3)/2, 1)
        
        a6_df3 = st.radio(
            '**battery potential** (V)', options = [a6_p1_df3[0], a6_p1_df3[1], a6_p1_df3[2], a6_p1_df3[3], a6_p1_df3[4]], key = 'a6_df3'
        )
        keyswitch_battery_potential_df3 = 0
        if a6_df3 == a6_p1_df3[0]:
            keyswitch_battery_potential_df3 = a6_p2_df3[0]
        elif a6_df3 == a6_p1_df3[1]:
            keyswitch_battery_potential_df3 = a6_p2_df3[1]
        elif a6_df3 == a6_p1_df3[2]:
            keyswitch_battery_potential_df3 = a6_p2_df3[2]
        elif a6_df3 == a6_p1_df3[3]:
            keyswitch_battery_potential_df3 = a6_p2_df3[3]
        elif a6_df3 == a6_p1_df3[4]:
            keyswitch_battery_potential_df3 = a6_p2_df3[4]
    
    with col7:
        DF3 = pd.DataFrame({
            'keyswitch battery potential':[keyswitch_battery_potential_df3],
            'engine oil temperature 1':[engine_oil_temperature_1_df3],
            'engine oil pressure':[engine_oil_pressure_df3],
            'engine intake manifold 1 pressure':[engine_intake_manifold_1_pressure_df3],
            'engine fuel rate':[engine_fuel_rate_df3],
            'engine ecu temperature':[engine_ecu_temperature_df3]
        })
        
        result_df3 = model.predict(DF3)
        status_df3 = list(encoder_class.inverse_transform(np.array(int(result_df3)).ravel()))[0]
        
        st.subheader(':green[period 3]')
        st.write('Status: **:orange[' + status_df3 + ']**')
        status_ratio.append(status_df3)
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#4
with st.container():
    st.write('---')
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

    with col1:
        a1_m1_df4 = 41; a1_m2_df4 = 94; a1_s_df4 = 5
        a1_p1_df4 = list(np.arange(a1_m1_df4, a1_m2_df4, (a1_m2_df4 - a1_m1_df4)/a1_s_df4)) + [a1_m2_df4]
        for i in range(0, len(a1_p1_df4) - 1):
            a1_p1_df4[i] = str(int(a1_p1_df4[i])) + ' ≤ x < ' + str(int(a1_p1_df4[i + 1]))
        a1_p2_df4 = list(np.arange(a1_m1_df4, a1_m2_df4, (a1_m2_df4 - a1_m1_df4)/a1_s_df4))
        for i in range(0, len(a1_p2_df4)):
            a1_p2_df4[i] = round(a1_p2_df4[i] + ((a1_m2_df4 - a1_m1_df4)/a1_s_df4)/2, 1)
        
        a1_df4 = st.radio(
            '**ECU temp** (°C)', options = [a1_p1_df4[0], a1_p1_df4[1], a1_p1_df4[2], a1_p1_df4[3], a1_p1_df4[4]], key = 'a1_df4'
        )
        engine_ecu_temperature_df4 = 0
        if a1_df4 == a1_p1_df4[0]:
            engine_ecu_temperature_df4 = a1_p2_df4[0]
        elif a1_df4 == a1_p1_df4[1]:
            engine_ecu_temperature_df4 = a1_p2_df4[1]
        elif a1_df4 == a1_p1_df4[2]:
            engine_ecu_temperature_df4 = a1_p2_df4[2]
        elif a1_df4 == a1_p1_df4[3]:
            engine_ecu_temperature_df4 = a1_p2_df4[3]
        elif a1_df4 == a1_p1_df4[4]:
            engine_ecu_temperature_df4 = a1_p2_df4[4]
    
    with col2:
        a2_m1_df4 = 3; a2_m2_df4 = 37; a2_s_df4 = 5
        a2_p1_df4 = list(np.arange(a2_m1_df4, a2_m2_df4, (a2_m2_df4 - a2_m1_df4)/a2_s_df4)) + [a2_m2_df4]
        for i in range(0, len(a2_p1_df4) - 1):
            a2_p1_df4[i] = str(int(a2_p1_df4[i])) + ' ≤ x < ' + str(int(a2_p1_df4[i + 1]))
        a2_p2_df4 = list(np.arange(a2_m1_df4, a2_m2_df4, (a2_m2_df4 - a2_m1_df4)/a2_s_df4))
        for i in range(0, len(a2_p2_df4)):
            a2_p2_df4[i] = round(a2_p2_df4[i] + ((a2_m2_df4 - a2_m1_df4)/a2_s_df4)/2, 1)
        
        a2_df4 = st.radio(
            '**fuel rate** (L/h)', options = [a2_p1_df4[0], a2_p1_df4[1], a2_p1_df4[2], a2_p1_df4[3], a2_p1_df4[4]], key = 'a2_df4'
        )
        engine_fuel_rate_df4 = 0
        if a2_df4 == a2_p1_df4[0]:
            engine_fuel_rate_df4 = a2_p2_df4[0]
        elif a2_df4 == a2_p1_df4[1]:
            engine_fuel_rate_df4 = a2_p2_df4[1]
        elif a2_df4 == a2_p1_df4[2]:
            engine_fuel_rate_df4 = a2_p2_df4[2]
        elif a2_df4 == a2_p1_df4[3]:
            engine_fuel_rate_df4 = a2_p2_df4[3]
        elif a2_df4 == a2_p1_df4[4]:
            engine_fuel_rate_df4 = a2_p2_df4[4]
    
    with col3:
        a3_m1_df4 = 4; a3_m2_df4 = 80; a3_s_df4 = 5
        a3_p1_df4 = list(np.arange(a3_m1_df4, a3_m2_df4, (a3_m2_df4 - a3_m1_df4)/a3_s_df4)) + [a3_m2_df4]
        for i in range(0, len(a3_p1_df4) - 1):
            a3_p1_df4[i] = str(int(a3_p1_df4[i])) + ' ≤ x < ' + str(int(a3_p1_df4[i + 1]))
        a3_p2_df4 = list(np.arange(a3_m1_df4, a3_m2_df4, (a3_m2_df4 - a3_m1_df4)/a3_s_df4))
        for i in range(0, len(a3_p2_df4)):
            a3_p2_df4[i] = round(a3_p2_df4[i] + ((a3_m2_df4 - a3_m1_df4)/a3_s_df4)/2, 1)
        
        a3_df4 = st.radio(
            '**IMP** (bar)', options = [a3_p1_df4[0], a3_p1_df4[1], a3_p1_df4[2], a3_p1_df4[3], a3_p1_df4[4]], key = 'a3_df4'
        )
        engine_intake_manifold_1_pressure_df4 = 0
        if a3_df4 == a3_p1_df4[0]:
            engine_intake_manifold_1_pressure_df4 = a3_p2_df4[0]
        elif a3_df4 == a3_p1_df4[1]:
            engine_intake_manifold_1_pressure_df4 = a3_p2_df4[1]
        elif a3_df4 == a3_p1_df4[2]:
            engine_intake_manifold_1_pressure_df4 = a3_p2_df4[2]
        elif a3_df4 == a3_p1_df4[3]:
            engine_intake_manifold_1_pressure_df4 = a3_p2_df4[3]
        elif a3_df4 == a3_p1_df4[4]:
            engine_intake_manifold_1_pressure_df4 = a3_p2_df4[4]
            
    with col4:
        a4_m1_df4 = 192; a4_m2_df4 = 348; a4_s_df4 = 5
        a4_p1_df4 = list(np.arange(a4_m1_df4, a4_m2_df4, (a4_m2_df4 - a4_m1_df4)/a4_s_df4)) + [a4_m2_df4]
        for i in range(0, len(a4_p1_df4) - 1):
            a4_p1_df4[i] = str(int(a4_p1_df4[i])) + ' ≤ x < ' + str(int(a4_p1_df4[i + 1]))
        a4_p2_df4 = list(np.arange(a4_m1_df4, a4_m2_df4, (a4_m2_df4 - a4_m1_df4)/a4_s_df4))
        for i in range(0, len(a4_p2_df4)):
            a4_p2_df4[i] = round(a4_p2_df4[i] + ((a4_m2_df4 - a4_m1_df4)/a4_s_df4)/2, 1)
        
        a4_df4 = st.radio(
            '**oil pressure** (bar)', options = [a4_p1_df4[0], a4_p1_df4[1], a4_p1_df4[2], a4_p1_df4[3], a4_p1_df4[4]], key = 'a4_df4'
        )
        engine_oil_pressure_df4 = 0
        if a4_df4 == a4_p1_df4[0]:
            engine_oil_pressure_df4 = a4_p2_df4[0]
        elif a4_df4 == a4_p1_df4[1]:
            engine_oil_pressure_df4 = a4_p2_df4[1]
        elif a4_df4 == a4_p1_df4[2]:
            engine_oil_pressure_df4 = a4_p2_df4[2]
        elif a4_df4 == a4_p1_df4[3]:
            engine_oil_pressure_df4 = a4_p2_df4[3]
        elif a4_df4 == a4_p1_df4[4]:
            engine_oil_pressure_df4 = a4_p2_df4[4]
    
    with col5:
        a5_m1_df4 = 83; a5_m2_df4 = 101; a5_s_df4 = 5
        a5_p1_df4 = list(np.arange(a5_m1_df4, a5_m2_df4, (a5_m2_df4 - a5_m1_df4)/a5_s_df4)) + [a5_m2_df4]
        for i in range(0, len(a5_p1_df4) - 1):
            a5_p1_df4[i] = str(int(a5_p1_df4[i])) + ' ≤ x < ' + str(int(a5_p1_df4[i + 1]))
        a5_p2_df4 = list(np.arange(a5_m1_df4, a5_m2_df4, (a5_m2_df4 - a5_m1_df4)/a5_s_df4))
        for i in range(0, len(a5_p2_df4)):
            a5_p2_df4[i] = round(a5_p2_df4[i] + ((a5_m2_df4 - a5_m1_df4)/a5_s_df4)/2, 1)
        
        a5_df4 = st.radio(
            '**oil temperature** (°C)', options = [a5_p1_df4[0], a5_p1_df4[1], a5_p1_df4[2], a5_p1_df4[3], a5_p1_df4[4]], key = 'a5_df4'
        )
        engine_oil_temperature_1_df4 = 0
        if a5_df4 == a5_p1_df4[0]:
            engine_oil_temperature_1_df4 = a5_p2_df4[0]
        elif a5_df4 == a5_p1_df4[1]:
            engine_oil_temperature_1_df4 = a5_p2_df4[1]
        elif a5_df4 == a5_p1_df4[2]:
            engine_oil_temperature_1_df4 = a5_p2_df4[2]
        elif a5_df4 == a5_p1_df4[3]:
            engine_oil_temperature_1_df4 = a5_p2_df4[3]
        elif a5_df4 == a5_p1_df4[4]:
            engine_oil_temperature_1_df4 = a5_p2_df4[4]
    
    with col6:
        a6_m1_df4 = 26.1; a6_m2_df4 = 28; a6_s_df4 = 5
        a6_p1_df4 = list(np.arange(a6_m1_df4, a6_m2_df4, (a6_m2_df4 - a6_m1_df4)/a6_s_df4)) + [a6_m2_df4]
        for i in range(0, len(a6_p1_df4) - 1):
            a6_p1_df4[i] = str(int(a6_p1_df4[i])) + ' ≤ x < ' + str(int(a6_p1_df4[i + 1]))
        a6_p2_df4 = list(np.arange(a6_m1_df4, a6_m2_df4, (a6_m2_df4 - a6_m1_df4)/a6_s_df4))
        for i in range(0, len(a6_p2_df4)):
            a6_p2_df4[i] = round(a6_p2_df4[i] + ((a6_m2_df4 - a6_m1_df4)/a6_s_df4)/2, 1)
        
        a6_df4 = st.radio(
            '**battery potential** (V)', options = [a6_p1_df4[0], a6_p1_df4[1], a6_p1_df4[2], a6_p1_df4[3], a6_p1_df4[4]], key = 'a6_df4'
        )
        keyswitch_battery_potential_df4 = 0
        if a6_df4 == a6_p1_df4[0]:
            keyswitch_battery_potential_df4 = a6_p2_df4[0]
        elif a6_df4 == a6_p1_df4[1]:
            keyswitch_battery_potential_df4 = a6_p2_df4[1]
        elif a6_df4 == a6_p1_df4[2]:
            keyswitch_battery_potential_df4 = a6_p2_df4[2]
        elif a6_df4 == a6_p1_df4[3]:
            keyswitch_battery_potential_df4 = a6_p2_df4[3]
        elif a6_df4 == a6_p1_df4[4]:
            keyswitch_battery_potential_df4 = a6_p2_df4[4]
    
    with col7:
        DF4 = pd.DataFrame({
            'keyswitch battery potential':[keyswitch_battery_potential_df4],
            'engine oil temperature 1':[engine_oil_temperature_1_df4],
            'engine oil pressure':[engine_oil_pressure_df4],
            'engine intake manifold 1 pressure':[engine_intake_manifold_1_pressure_df4],
            'engine fuel rate':[engine_fuel_rate_df4],
            'engine ecu temperature':[engine_ecu_temperature_df4]
        })
        
        result_df4 = model.predict(DF4)
        status_df4 = list(encoder_class.inverse_transform(np.array(int(result_df4)).ravel()))[0]
        
        st.subheader(':green[period 4]')
        st.write('Status: **:orange[' + status_df4 + ']**')
        status_ratio.append(status_df4)
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#5
with st.container():
    st.write('---')
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

    with col1:
        a1_m1_df5 = 41; a1_m2_df5 = 94; a1_s_df5 = 5
        a1_p1_df5 = list(np.arange(a1_m1_df5, a1_m2_df5, (a1_m2_df5 - a1_m1_df5)/a1_s_df5)) + [a1_m2_df5]
        for i in range(0, len(a1_p1_df5) - 1):
            a1_p1_df5[i] = str(int(a1_p1_df5[i])) + ' ≤ x < ' + str(int(a1_p1_df5[i + 1]))
        a1_p2_df5 = list(np.arange(a1_m1_df5, a1_m2_df5, (a1_m2_df5 - a1_m1_df5)/a1_s_df5))
        for i in range(0, len(a1_p2_df5)):
            a1_p2_df5[i] = round(a1_p2_df5[i] + ((a1_m2_df5 - a1_m1_df5)/a1_s_df5)/2, 1)
        
        a1_df5 = st.radio(
            '**ECU temp** (°C)', options = [a1_p1_df5[0], a1_p1_df5[1], a1_p1_df5[2], a1_p1_df5[3], a1_p1_df5[4]], key = 'a1_df5'
        )
        engine_ecu_temperature_df5 = 0
        if a1_df5 == a1_p1_df5[0]:
            engine_ecu_temperature_df5 = a1_p2_df5[0]
        elif a1_df5 == a1_p1_df5[1]:
            engine_ecu_temperature_df5 = a1_p2_df5[1]
        elif a1_df5 == a1_p1_df5[2]:
            engine_ecu_temperature_df5 = a1_p2_df5[2]
        elif a1_df5 == a1_p1_df5[3]:
            engine_ecu_temperature_df5 = a1_p2_df5[3]
        elif a1_df5 == a1_p1_df5[4]:
            engine_ecu_temperature_df5 = a1_p2_df5[4]
    
    with col2:
        a2_m1_df5 = 3; a2_m2_df5 = 37; a2_s_df5 = 5
        a2_p1_df5 = list(np.arange(a2_m1_df5, a2_m2_df5, (a2_m2_df5 - a2_m1_df5)/a2_s_df5)) + [a2_m2_df5]
        for i in range(0, len(a2_p1_df5) - 1):
            a2_p1_df5[i] = str(int(a2_p1_df5[i])) + ' ≤ x < ' + str(int(a2_p1_df5[i + 1]))
        a2_p2_df5 = list(np.arange(a2_m1_df5, a2_m2_df5, (a2_m2_df5 - a2_m1_df5)/a2_s_df5))
        for i in range(0, len(a2_p2_df5)):
            a2_p2_df5[i] = round(a2_p2_df5[i] + ((a2_m2_df5 - a2_m1_df5)/a2_s_df5)/2, 1)
        
        a2_df5 = st.radio(
            '**fuel rate** (L/h)', options = [a2_p1_df5[0], a2_p1_df5[1], a2_p1_df5[2], a2_p1_df5[3], a2_p1_df5[4]], key = 'a2_df5'
        )
        engine_fuel_rate_df5 = 0
        if a2_df5 == a2_p1_df5[0]:
            engine_fuel_rate_df5 = a2_p2_df5[0]
        elif a2_df5 == a2_p1_df5[1]:
            engine_fuel_rate_df5 = a2_p2_df5[1]
        elif a2_df5 == a2_p1_df5[2]:
            engine_fuel_rate_df5 = a2_p2_df5[2]
        elif a2_df5 == a2_p1_df5[3]:
            engine_fuel_rate_df5 = a2_p2_df5[3]
        elif a2_df5 == a2_p1_df5[4]:
            engine_fuel_rate_df5 = a2_p2_df5[4]
    
    with col3:
        a3_m1_df5 = 4; a3_m2_df5 = 80; a3_s_df5 = 5
        a3_p1_df5 = list(np.arange(a3_m1_df5, a3_m2_df5, (a3_m2_df5 - a3_m1_df5)/a3_s_df5)) + [a3_m2_df5]
        for i in range(0, len(a3_p1_df5) - 1):
            a3_p1_df5[i] = str(int(a3_p1_df5[i])) + ' ≤ x < ' + str(int(a3_p1_df5[i + 1]))
        a3_p2_df5 = list(np.arange(a3_m1_df5, a3_m2_df5, (a3_m2_df5 - a3_m1_df5)/a3_s_df5))
        for i in range(0, len(a3_p2_df5)):
            a3_p2_df5[i] = round(a3_p2_df5[i] + ((a3_m2_df5 - a3_m1_df5)/a3_s_df5)/2, 1)
        
        a3_df5 = st.radio(
            '**IMP** (bar)', options = [a3_p1_df5[0], a3_p1_df5[1], a3_p1_df5[2], a3_p1_df5[3], a3_p1_df5[4]], key = 'a3_df5'
        )
        engine_intake_manifold_1_pressure_df5 = 0
        if a3_df5 == a3_p1_df5[0]:
            engine_intake_manifold_1_pressure_df5 = a3_p2_df5[0]
        elif a3_df5 == a3_p1_df5[1]:
            engine_intake_manifold_1_pressure_df5 = a3_p2_df5[1]
        elif a3_df5 == a3_p1_df5[2]:
            engine_intake_manifold_1_pressure_df5 = a3_p2_df5[2]
        elif a3_df5 == a3_p1_df5[3]:
            engine_intake_manifold_1_pressure_df5 = a3_p2_df5[3]
        elif a3_df5 == a3_p1_df5[4]:
            engine_intake_manifold_1_pressure_df5 = a3_p2_df5[4]
            
    with col4:
        a4_m1_df5 = 192; a4_m2_df5 = 348; a4_s_df5 = 5
        a4_p1_df5 = list(np.arange(a4_m1_df5, a4_m2_df5, (a4_m2_df5 - a4_m1_df5)/a4_s_df5)) + [a4_m2_df5]
        for i in range(0, len(a4_p1_df5) - 1):
            a4_p1_df5[i] = str(int(a4_p1_df5[i])) + ' ≤ x < ' + str(int(a4_p1_df5[i + 1]))
        a4_p2_df5 = list(np.arange(a4_m1_df5, a4_m2_df5, (a4_m2_df5 - a4_m1_df5)/a4_s_df5))
        for i in range(0, len(a4_p2_df5)):
            a4_p2_df5[i] = round(a4_p2_df5[i] + ((a4_m2_df5 - a4_m1_df5)/a4_s_df5)/2, 1)
        
        a4_df5 = st.radio(
            '**oil pressure** (bar)', options = [a4_p1_df5[0], a4_p1_df5[1], a4_p1_df5[2], a4_p1_df5[3], a4_p1_df5[4]], key = 'a4_df5'
        )
        engine_oil_pressure_df5 = 0
        if a4_df5 == a4_p1_df5[0]:
            engine_oil_pressure_df5 = a4_p2_df5[0]
        elif a4_df5 == a4_p1_df5[1]:
            engine_oil_pressure_df5 = a4_p2_df5[1]
        elif a4_df5 == a4_p1_df5[2]:
            engine_oil_pressure_df5 = a4_p2_df5[2]
        elif a4_df5 == a4_p1_df5[3]:
            engine_oil_pressure_df5 = a4_p2_df5[3]
        elif a4_df5 == a4_p1_df5[4]:
            engine_oil_pressure_df5 = a4_p2_df5[4]
    
    with col5:
        a5_m1_df5 = 83; a5_m2_df5 = 101; a5_s_df5 = 5
        a5_p1_df5 = list(np.arange(a5_m1_df5, a5_m2_df5, (a5_m2_df5 - a5_m1_df5)/a5_s_df5)) + [a5_m2_df5]
        for i in range(0, len(a5_p1_df5) - 1):
            a5_p1_df5[i] = str(int(a5_p1_df5[i])) + ' ≤ x < ' + str(int(a5_p1_df5[i + 1]))
        a5_p2_df5 = list(np.arange(a5_m1_df5, a5_m2_df5, (a5_m2_df5 - a5_m1_df5)/a5_s_df5))
        for i in range(0, len(a5_p2_df5)):
            a5_p2_df5[i] = round(a5_p2_df5[i] + ((a5_m2_df5 - a5_m1_df5)/a5_s_df5)/2, 1)
        
        a5_df5 = st.radio(
            '**oil temperature** (°C)', options = [a5_p1_df5[0], a5_p1_df5[1], a5_p1_df5[2], a5_p1_df5[3], a5_p1_df5[4]], key = 'a5_df5'
        )
        engine_oil_temperature_1_df5 = 0
        if a5_df5 == a5_p1_df5[0]:
            engine_oil_temperature_1_df5 = a5_p2_df5[0]
        elif a5_df5 == a5_p1_df5[1]:
            engine_oil_temperature_1_df5 = a5_p2_df5[1]
        elif a5_df5 == a5_p1_df5[2]:
            engine_oil_temperature_1_df5 = a5_p2_df5[2]
        elif a5_df5 == a5_p1_df5[3]:
            engine_oil_temperature_1_df5 = a5_p2_df5[3]
        elif a5_df5 == a5_p1_df5[4]:
            engine_oil_temperature_1_df5 = a5_p2_df5[4]
    
    with col6:
        a6_m1_df5 = 26.1; a6_m2_df5 = 28; a6_s_df5 = 5
        a6_p1_df5 = list(np.arange(a6_m1_df5, a6_m2_df5, (a6_m2_df5 - a6_m1_df5)/a6_s_df5)) + [a6_m2_df5]
        for i in range(0, len(a6_p1_df5) - 1):
            a6_p1_df5[i] = str(int(a6_p1_df5[i])) + ' ≤ x < ' + str(int(a6_p1_df5[i + 1]))
        a6_p2_df5 = list(np.arange(a6_m1_df5, a6_m2_df5, (a6_m2_df5 - a6_m1_df5)/a6_s_df5))
        for i in range(0, len(a6_p2_df5)):
            a6_p2_df5[i] = round(a6_p2_df5[i] + ((a6_m2_df5 - a6_m1_df5)/a6_s_df5)/2, 1)
        
        a6_df5 = st.radio(
            '**battery potential** (V)', options = [a6_p1_df5[0], a6_p1_df5[1], a6_p1_df5[2], a6_p1_df5[3], a6_p1_df5[4]], key = 'a6_df5'
        )
        keyswitch_battery_potential_df5 = 0
        if a6_df5 == a6_p1_df5[0]:
            keyswitch_battery_potential_df5 = a6_p2_df5[0]
        elif a6_df5 == a6_p1_df5[1]:
            keyswitch_battery_potential_df5 = a6_p2_df5[1]
        elif a6_df5 == a6_p1_df5[2]:
            keyswitch_battery_potential_df5 = a6_p2_df5[2]
        elif a6_df5 == a6_p1_df5[3]:
            keyswitch_battery_potential_df5 = a6_p2_df5[3]
        elif a6_df5 == a6_p1_df5[4]:
            keyswitch_battery_potential_df5 = a6_p2_df5[4]
    
    with col7:
        DF5 = pd.DataFrame({
            'keyswitch battery potential':[keyswitch_battery_potential_df5],
            'engine oil temperature 1':[engine_oil_temperature_1_df5],
            'engine oil pressure':[engine_oil_pressure_df5],
            'engine intake manifold 1 pressure':[engine_intake_manifold_1_pressure_df5],
            'engine fuel rate':[engine_fuel_rate_df5],
            'engine ecu temperature':[engine_ecu_temperature_df5]
        })
        
        result_df5 = model.predict(DF5)
        status_df5 = list(encoder_class.inverse_transform(np.array(int(result_df5)).ravel()))[0]
        
        st.subheader(':green[period 5]')
        st.write('Status: **:orange[' + status_df5 + ']**')
        status_ratio.append(status_df5)
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#6
with st.container():
    st.write('---')
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

    with col1:
        a1_m1_df6 = 41; a1_m2_df6 = 94; a1_s_df6 = 5
        a1_p1_df6 = list(np.arange(a1_m1_df6, a1_m2_df6, (a1_m2_df6 - a1_m1_df6)/a1_s_df6)) + [a1_m2_df6]
        for i in range(0, len(a1_p1_df6) - 1):
            a1_p1_df6[i] = str(int(a1_p1_df6[i])) + ' ≤ x < ' + str(int(a1_p1_df6[i + 1]))
        a1_p2_df6 = list(np.arange(a1_m1_df6, a1_m2_df6, (a1_m2_df6 - a1_m1_df6)/a1_s_df6))
        for i in range(0, len(a1_p2_df6)):
            a1_p2_df6[i] = round(a1_p2_df6[i] + ((a1_m2_df6 - a1_m1_df6)/a1_s_df6)/2, 1)
        
        a1_df6 = st.radio(
            '**ECU temp** (°C)', options = [a1_p1_df6[0], a1_p1_df6[1], a1_p1_df6[2], a1_p1_df6[3], a1_p1_df6[4]], key = 'a1_df6'
        )
        engine_ecu_temperature_df6 = 0
        if a1_df6 == a1_p1_df6[0]:
            engine_ecu_temperature_df6 = a1_p2_df6[0]
        elif a1_df6 == a1_p1_df6[1]:
            engine_ecu_temperature_df6 = a1_p2_df6[1]
        elif a1_df6 == a1_p1_df6[2]:
            engine_ecu_temperature_df6 = a1_p2_df6[2]
        elif a1_df6 == a1_p1_df6[3]:
            engine_ecu_temperature_df6 = a1_p2_df6[3]
        elif a1_df6 == a1_p1_df6[4]:
            engine_ecu_temperature_df6 = a1_p2_df6[4]
    
    with col2:
        a2_m1_df6 = 3; a2_m2_df6 = 37; a2_s_df6 = 5
        a2_p1_df6 = list(np.arange(a2_m1_df6, a2_m2_df6, (a2_m2_df6 - a2_m1_df6)/a2_s_df6)) + [a2_m2_df6]
        for i in range(0, len(a2_p1_df6) - 1):
            a2_p1_df6[i] = str(int(a2_p1_df6[i])) + ' ≤ x < ' + str(int(a2_p1_df6[i + 1]))
        a2_p2_df6 = list(np.arange(a2_m1_df6, a2_m2_df6, (a2_m2_df6 - a2_m1_df6)/a2_s_df6))
        for i in range(0, len(a2_p2_df6)):
            a2_p2_df6[i] = round(a2_p2_df6[i] + ((a2_m2_df6 - a2_m1_df6)/a2_s_df6)/2, 1)
        
        a2_df6 = st.radio(
            '**fuel rate** (L/h)', options = [a2_p1_df6[0], a2_p1_df6[1], a2_p1_df6[2], a2_p1_df6[3], a2_p1_df6[4]], key = 'a2_df6'
        )
        engine_fuel_rate_df6 = 0
        if a2_df6 == a2_p1_df6[0]:
            engine_fuel_rate_df6 = a2_p2_df6[0]
        elif a2_df6 == a2_p1_df6[1]:
            engine_fuel_rate_df6 = a2_p2_df6[1]
        elif a2_df6 == a2_p1_df6[2]:
            engine_fuel_rate_df6 = a2_p2_df6[2]
        elif a2_df6 == a2_p1_df6[3]:
            engine_fuel_rate_df6 = a2_p2_df6[3]
        elif a2_df6 == a2_p1_df6[4]:
            engine_fuel_rate_df6 = a2_p2_df6[4]
    
    with col3:
        a3_m1_df6 = 4; a3_m2_df6 = 80; a3_s_df6 = 5
        a3_p1_df6 = list(np.arange(a3_m1_df6, a3_m2_df6, (a3_m2_df6 - a3_m1_df6)/a3_s_df6)) + [a3_m2_df6]
        for i in range(0, len(a3_p1_df6) - 1):
            a3_p1_df6[i] = str(int(a3_p1_df6[i])) + ' ≤ x < ' + str(int(a3_p1_df6[i + 1]))
        a3_p2_df6 = list(np.arange(a3_m1_df6, a3_m2_df6, (a3_m2_df6 - a3_m1_df6)/a3_s_df6))
        for i in range(0, len(a3_p2_df6)):
            a3_p2_df6[i] = round(a3_p2_df6[i] + ((a3_m2_df6 - a3_m1_df6)/a3_s_df6)/2, 1)
        
        a3_df6 = st.radio(
            '**IMP** (bar)', options = [a3_p1_df6[0], a3_p1_df6[1], a3_p1_df6[2], a3_p1_df6[3], a3_p1_df6[4]], key = 'a3_df6'
        )
        engine_intake_manifold_1_pressure_df6 = 0
        if a3_df6 == a3_p1_df6[0]:
            engine_intake_manifold_1_pressure_df6 = a3_p2_df6[0]
        elif a3_df6 == a3_p1_df6[1]:
            engine_intake_manifold_1_pressure_df6 = a3_p2_df6[1]
        elif a3_df6 == a3_p1_df6[2]:
            engine_intake_manifold_1_pressure_df6 = a3_p2_df6[2]
        elif a3_df6 == a3_p1_df6[3]:
            engine_intake_manifold_1_pressure_df6 = a3_p2_df6[3]
        elif a3_df6 == a3_p1_df6[4]:
            engine_intake_manifold_1_pressure_df6 = a3_p2_df6[4]
            
    with col4:
        a4_m1_df6 = 192; a4_m2_df6 = 348; a4_s_df6 = 5
        a4_p1_df6 = list(np.arange(a4_m1_df6, a4_m2_df6, (a4_m2_df6 - a4_m1_df6)/a4_s_df6)) + [a4_m2_df6]
        for i in range(0, len(a4_p1_df6) - 1):
            a4_p1_df6[i] = str(int(a4_p1_df6[i])) + ' ≤ x < ' + str(int(a4_p1_df6[i + 1]))
        a4_p2_df6 = list(np.arange(a4_m1_df6, a4_m2_df6, (a4_m2_df6 - a4_m1_df6)/a4_s_df6))
        for i in range(0, len(a4_p2_df6)):
            a4_p2_df6[i] = round(a4_p2_df6[i] + ((a4_m2_df6 - a4_m1_df6)/a4_s_df6)/2, 1)
        
        a4_df6 = st.radio(
            '**oil pressure** (bar)', options = [a4_p1_df6[0], a4_p1_df6[1], a4_p1_df6[2], a4_p1_df6[3], a4_p1_df6[4]], key = 'a4_df6'
        )
        engine_oil_pressure_df6 = 0
        if a4_df6 == a4_p1_df6[0]:
            engine_oil_pressure_df6 = a4_p2_df6[0]
        elif a4_df6 == a4_p1_df6[1]:
            engine_oil_pressure_df6 = a4_p2_df6[1]
        elif a4_df6 == a4_p1_df6[2]:
            engine_oil_pressure_df6 = a4_p2_df6[2]
        elif a4_df6 == a4_p1_df6[3]:
            engine_oil_pressure_df6 = a4_p2_df6[3]
        elif a4_df6 == a4_p1_df6[4]:
            engine_oil_pressure_df6 = a4_p2_df6[4]
    
    with col5:
        a5_m1_df6 = 83; a5_m2_df6 = 101; a5_s_df6 = 5
        a5_p1_df6 = list(np.arange(a5_m1_df6, a5_m2_df6, (a5_m2_df6 - a5_m1_df6)/a5_s_df6)) + [a5_m2_df6]
        for i in range(0, len(a5_p1_df6) - 1):
            a5_p1_df6[i] = str(int(a5_p1_df6[i])) + ' ≤ x < ' + str(int(a5_p1_df6[i + 1]))
        a5_p2_df6 = list(np.arange(a5_m1_df6, a5_m2_df6, (a5_m2_df6 - a5_m1_df6)/a5_s_df6))
        for i in range(0, len(a5_p2_df6)):
            a5_p2_df6[i] = round(a5_p2_df6[i] + ((a5_m2_df6 - a5_m1_df6)/a5_s_df6)/2, 1)
        
        a5_df6 = st.radio(
            '**oil temperature** (°C)', options = [a5_p1_df6[0], a5_p1_df6[1], a5_p1_df6[2], a5_p1_df6[3], a5_p1_df6[4]], key = 'a5_df6'
        )
        engine_oil_temperature_1_df6 = 0
        if a5_df6 == a5_p1_df6[0]:
            engine_oil_temperature_1_df6 = a5_p2_df6[0]
        elif a5_df6 == a5_p1_df6[1]:
            engine_oil_temperature_1_df6 = a5_p2_df6[1]
        elif a5_df6 == a5_p1_df6[2]:
            engine_oil_temperature_1_df6 = a5_p2_df6[2]
        elif a5_df6 == a5_p1_df6[3]:
            engine_oil_temperature_1_df6 = a5_p2_df6[3]
        elif a5_df6 == a5_p1_df6[4]:
            engine_oil_temperature_1_df6 = a5_p2_df6[4]
    
    with col6:
        a6_m1_df6 = 26.1; a6_m2_df6 = 28; a6_s_df6 = 5
        a6_p1_df6 = list(np.arange(a6_m1_df6, a6_m2_df6, (a6_m2_df6 - a6_m1_df6)/a6_s_df6)) + [a6_m2_df6]
        for i in range(0, len(a6_p1_df6) - 1):
            a6_p1_df6[i] = str(int(a6_p1_df6[i])) + ' ≤ x < ' + str(int(a6_p1_df6[i + 1]))
        a6_p2_df6 = list(np.arange(a6_m1_df6, a6_m2_df6, (a6_m2_df6 - a6_m1_df6)/a6_s_df6))
        for i in range(0, len(a6_p2_df6)):
            a6_p2_df6[i] = round(a6_p2_df6[i] + ((a6_m2_df6 - a6_m1_df6)/a6_s_df6)/2, 1)
        
        a6_df6 = st.radio(
            '**battery potential** (V)', options = [a6_p1_df6[0], a6_p1_df6[1], a6_p1_df6[2], a6_p1_df6[3], a6_p1_df6[4]], key = 'a6_df6'
        )
        keyswitch_battery_potential_df6 = 0
        if a6_df6 == a6_p1_df6[0]:
            keyswitch_battery_potential_df6 = a6_p2_df6[0]
        elif a6_df6 == a6_p1_df6[1]:
            keyswitch_battery_potential_df6 = a6_p2_df6[1]
        elif a6_df6 == a6_p1_df6[2]:
            keyswitch_battery_potential_df6 = a6_p2_df6[2]
        elif a6_df6 == a6_p1_df6[3]:
            keyswitch_battery_potential_df6 = a6_p2_df6[3]
        elif a6_df6 == a6_p1_df6[4]:
            keyswitch_battery_potential_df6 = a6_p2_df6[4]
    
    with col7:
        DF6 = pd.DataFrame({
            'keyswitch battery potential':[keyswitch_battery_potential_df6],
            'engine oil temperature 1':[engine_oil_temperature_1_df6],
            'engine oil pressure':[engine_oil_pressure_df6],
            'engine intake manifold 1 pressure':[engine_intake_manifold_1_pressure_df6],
            'engine fuel rate':[engine_fuel_rate_df6],
            'engine ecu temperature':[engine_ecu_temperature_df6]
        })
        
        result_df6 = model.predict(DF6)
        status_df6 = list(encoder_class.inverse_transform(np.array(int(result_df6)).ravel()))[0]
        
        st.subheader(':green[period 6]')
        st.write('Status: **:orange[' + status_df6 + ']**')
        status_ratio.append(status_df6)
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#7
with st.container():
    st.write('---')
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

    with col1:
        a1_m1_df7 = 41; a1_m2_df7 = 94; a1_s_df7 = 5
        a1_p1_df7 = list(np.arange(a1_m1_df7, a1_m2_df7, (a1_m2_df7 - a1_m1_df7)/a1_s_df7)) + [a1_m2_df7]
        for i in range(0, len(a1_p1_df7) - 1):
            a1_p1_df7[i] = str(int(a1_p1_df7[i])) + ' ≤ x < ' + str(int(a1_p1_df7[i + 1]))
        a1_p2_df7 = list(np.arange(a1_m1_df7, a1_m2_df7, (a1_m2_df7 - a1_m1_df7)/a1_s_df7))
        for i in range(0, len(a1_p2_df7)):
            a1_p2_df7[i] = round(a1_p2_df7[i] + ((a1_m2_df7 - a1_m1_df7)/a1_s_df7)/2, 1)
        
        a1_df7 = st.radio(
            '**ECU temp** (°C)', options = [a1_p1_df7[0], a1_p1_df7[1], a1_p1_df7[2], a1_p1_df7[3], a1_p1_df7[4]], key = 'a1_df7'
        )
        engine_ecu_temperature_df7 = 0
        if a1_df7 == a1_p1_df7[0]:
            engine_ecu_temperature_df7 = a1_p2_df7[0]
        elif a1_df7 == a1_p1_df7[1]:
            engine_ecu_temperature_df7 = a1_p2_df7[1]
        elif a1_df7 == a1_p1_df7[2]:
            engine_ecu_temperature_df7 = a1_p2_df7[2]
        elif a1_df7 == a1_p1_df7[3]:
            engine_ecu_temperature_df7 = a1_p2_df7[3]
        elif a1_df7 == a1_p1_df7[4]:
            engine_ecu_temperature_df7 = a1_p2_df7[4]
    
    with col2:
        a2_m1_df7 = 3; a2_m2_df7 = 37; a2_s_df7 = 5
        a2_p1_df7 = list(np.arange(a2_m1_df7, a2_m2_df7, (a2_m2_df7 - a2_m1_df7)/a2_s_df7)) + [a2_m2_df7]
        for i in range(0, len(a2_p1_df7) - 1):
            a2_p1_df7[i] = str(int(a2_p1_df7[i])) + ' ≤ x < ' + str(int(a2_p1_df7[i + 1]))
        a2_p2_df7 = list(np.arange(a2_m1_df7, a2_m2_df7, (a2_m2_df7 - a2_m1_df7)/a2_s_df7))
        for i in range(0, len(a2_p2_df7)):
            a2_p2_df7[i] = round(a2_p2_df7[i] + ((a2_m2_df7 - a2_m1_df7)/a2_s_df7)/2, 1)
        
        a2_df7 = st.radio(
            '**fuel rate** (L/h)', options = [a2_p1_df7[0], a2_p1_df7[1], a2_p1_df7[2], a2_p1_df7[3], a2_p1_df7[4]], key = 'a2_df7'
        )
        engine_fuel_rate_df7 = 0
        if a2_df7 == a2_p1_df7[0]:
            engine_fuel_rate_df7 = a2_p2_df7[0]
        elif a2_df7 == a2_p1_df7[1]:
            engine_fuel_rate_df7 = a2_p2_df7[1]
        elif a2_df7 == a2_p1_df7[2]:
            engine_fuel_rate_df7 = a2_p2_df7[2]
        elif a2_df7 == a2_p1_df7[3]:
            engine_fuel_rate_df7 = a2_p2_df7[3]
        elif a2_df7 == a2_p1_df7[4]:
            engine_fuel_rate_df7 = a2_p2_df7[4]
    
    with col3:
        a3_m1_df7 = 4; a3_m2_df7 = 80; a3_s_df7 = 5
        a3_p1_df7 = list(np.arange(a3_m1_df7, a3_m2_df7, (a3_m2_df7 - a3_m1_df7)/a3_s_df7)) + [a3_m2_df7]
        for i in range(0, len(a3_p1_df7) - 1):
            a3_p1_df7[i] = str(int(a3_p1_df7[i])) + ' ≤ x < ' + str(int(a3_p1_df7[i + 1]))
        a3_p2_df7 = list(np.arange(a3_m1_df7, a3_m2_df7, (a3_m2_df7 - a3_m1_df7)/a3_s_df7))
        for i in range(0, len(a3_p2_df7)):
            a3_p2_df7[i] = round(a3_p2_df7[i] + ((a3_m2_df7 - a3_m1_df7)/a3_s_df7)/2, 1)
        
        a3_df7 = st.radio(
            '**IMP** (bar)', options = [a3_p1_df7[0], a3_p1_df7[1], a3_p1_df7[2], a3_p1_df7[3], a3_p1_df7[4]], key = 'a3_df7'
        )
        engine_intake_manifold_1_pressure_df7 = 0
        if a3_df7 == a3_p1_df7[0]:
            engine_intake_manifold_1_pressure_df7 = a3_p2_df7[0]
        elif a3_df7 == a3_p1_df7[1]:
            engine_intake_manifold_1_pressure_df7 = a3_p2_df7[1]
        elif a3_df7 == a3_p1_df7[2]:
            engine_intake_manifold_1_pressure_df7 = a3_p2_df7[2]
        elif a3_df7 == a3_p1_df7[3]:
            engine_intake_manifold_1_pressure_df7 = a3_p2_df7[3]
        elif a3_df7 == a3_p1_df7[4]:
            engine_intake_manifold_1_pressure_df7 = a3_p2_df7[4]
            
    with col4:
        a4_m1_df7 = 192; a4_m2_df7 = 348; a4_s_df7 = 5
        a4_p1_df7 = list(np.arange(a4_m1_df7, a4_m2_df7, (a4_m2_df7 - a4_m1_df7)/a4_s_df7)) + [a4_m2_df7]
        for i in range(0, len(a4_p1_df7) - 1):
            a4_p1_df7[i] = str(int(a4_p1_df7[i])) + ' ≤ x < ' + str(int(a4_p1_df7[i + 1]))
        a4_p2_df7 = list(np.arange(a4_m1_df7, a4_m2_df7, (a4_m2_df7 - a4_m1_df7)/a4_s_df7))
        for i in range(0, len(a4_p2_df7)):
            a4_p2_df7[i] = round(a4_p2_df7[i] + ((a4_m2_df7 - a4_m1_df7)/a4_s_df7)/2, 1)
        
        a4_df7 = st.radio(
            '**oil pressure** (bar)', options = [a4_p1_df7[0], a4_p1_df7[1], a4_p1_df7[2], a4_p1_df7[3], a4_p1_df7[4]], key = 'a4_df7'
        )
        engine_oil_pressure_df7 = 0
        if a4_df7 == a4_p1_df7[0]:
            engine_oil_pressure_df7 = a4_p2_df7[0]
        elif a4_df7 == a4_p1_df7[1]:
            engine_oil_pressure_df7 = a4_p2_df7[1]
        elif a4_df7 == a4_p1_df7[2]:
            engine_oil_pressure_df7 = a4_p2_df7[2]
        elif a4_df7 == a4_p1_df7[3]:
            engine_oil_pressure_df7 = a4_p2_df7[3]
        elif a4_df7 == a4_p1_df7[4]:
            engine_oil_pressure_df7 = a4_p2_df7[4]
    
    with col5:
        a5_m1_df7 = 83; a5_m2_df7 = 101; a5_s_df7 = 5
        a5_p1_df7 = list(np.arange(a5_m1_df7, a5_m2_df7, (a5_m2_df7 - a5_m1_df7)/a5_s_df7)) + [a5_m2_df7]
        for i in range(0, len(a5_p1_df7) - 1):
            a5_p1_df7[i] = str(int(a5_p1_df7[i])) + ' ≤ x < ' + str(int(a5_p1_df7[i + 1]))
        a5_p2_df7 = list(np.arange(a5_m1_df7, a5_m2_df7, (a5_m2_df7 - a5_m1_df7)/a5_s_df7))
        for i in range(0, len(a5_p2_df7)):
            a5_p2_df7[i] = round(a5_p2_df7[i] + ((a5_m2_df7 - a5_m1_df7)/a5_s_df7)/2, 1)
        
        a5_df7 = st.radio(
            '**oil temperature** (°C)', options = [a5_p1_df7[0], a5_p1_df7[1], a5_p1_df7[2], a5_p1_df7[3], a5_p1_df7[4]], key = 'a5_df7'
        )
        engine_oil_temperature_1_df7 = 0
        if a5_df7 == a5_p1_df7[0]:
            engine_oil_temperature_1_df7 = a5_p2_df7[0]
        elif a5_df7 == a5_p1_df7[1]:
            engine_oil_temperature_1_df7 = a5_p2_df7[1]
        elif a5_df7 == a5_p1_df7[2]:
            engine_oil_temperature_1_df7 = a5_p2_df7[2]
        elif a5_df7 == a5_p1_df7[3]:
            engine_oil_temperature_1_df7 = a5_p2_df7[3]
        elif a5_df7 == a5_p1_df7[4]:
            engine_oil_temperature_1_df7 = a5_p2_df7[4]
    
    with col6:
        a6_m1_df7 = 26.1; a6_m2_df7 = 28; a6_s_df7 = 5
        a6_p1_df7 = list(np.arange(a6_m1_df7, a6_m2_df7, (a6_m2_df7 - a6_m1_df7)/a6_s_df7)) + [a6_m2_df7]
        for i in range(0, len(a6_p1_df7) - 1):
            a6_p1_df7[i] = str(int(a6_p1_df7[i])) + ' ≤ x < ' + str(int(a6_p1_df7[i + 1]))
        a6_p2_df7 = list(np.arange(a6_m1_df7, a6_m2_df7, (a6_m2_df7 - a6_m1_df7)/a6_s_df7))
        for i in range(0, len(a6_p2_df7)):
            a6_p2_df7[i] = round(a6_p2_df7[i] + ((a6_m2_df7 - a6_m1_df7)/a6_s_df7)/2, 1)
        
        a6_df7 = st.radio(
            '**battery potential** (V)', options = [a6_p1_df7[0], a6_p1_df7[1], a6_p1_df7[2], a6_p1_df7[3], a6_p1_df7[4]], key = 'a6_df7'
        )
        keyswitch_battery_potential_df7 = 0
        if a6_df7 == a6_p1_df7[0]:
            keyswitch_battery_potential_df7 = a6_p2_df7[0]
        elif a6_df7 == a6_p1_df7[1]:
            keyswitch_battery_potential_df7 = a6_p2_df7[1]
        elif a6_df7 == a6_p1_df7[2]:
            keyswitch_battery_potential_df7 = a6_p2_df7[2]
        elif a6_df7 == a6_p1_df7[3]:
            keyswitch_battery_potential_df7 = a6_p2_df7[3]
        elif a6_df7 == a6_p1_df7[4]:
            keyswitch_battery_potential_df7 = a6_p2_df7[4]
    
    with col7:
        DF7 = pd.DataFrame({
            'keyswitch battery potential':[keyswitch_battery_potential_df7],
            'engine oil temperature 1':[engine_oil_temperature_1_df7],
            'engine oil pressure':[engine_oil_pressure_df7],
            'engine intake manifold 1 pressure':[engine_intake_manifold_1_pressure_df7],
            'engine fuel rate':[engine_fuel_rate_df7],
            'engine ecu temperature':[engine_ecu_temperature_df7]
        })
        
        result_df7 = model.predict(DF7)
        status_df7 = list(encoder_class.inverse_transform(np.array(int(result_df7)).ravel()))[0]
        
        st.subheader(':green[period 7]')
        st.write('Status: **:orange[' + status_df7 + ']**')
        status_ratio.append(status_df7)
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#8
with st.container():
    st.write('---')
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

    with col1:
        a1_m1_df8 = 41; a1_m2_df8 = 94; a1_s_df8 = 5
        a1_p1_df8 = list(np.arange(a1_m1_df8, a1_m2_df8, (a1_m2_df8 - a1_m1_df8)/a1_s_df8)) + [a1_m2_df8]
        for i in range(0, len(a1_p1_df8) - 1):
            a1_p1_df8[i] = str(int(a1_p1_df8[i])) + ' ≤ x < ' + str(int(a1_p1_df8[i + 1]))
        a1_p2_df8 = list(np.arange(a1_m1_df8, a1_m2_df8, (a1_m2_df8 - a1_m1_df8)/a1_s_df8))
        for i in range(0, len(a1_p2_df8)):
            a1_p2_df8[i] = round(a1_p2_df8[i] + ((a1_m2_df8 - a1_m1_df8)/a1_s_df8)/2, 1)
        
        a1_df8 = st.radio(
            '**ECU temp** (°C)', options = [a1_p1_df8[0], a1_p1_df8[1], a1_p1_df8[2], a1_p1_df8[3], a1_p1_df8[4]], key = 'a1_df8'
        )
        engine_ecu_temperature_df8 = 0
        if a1_df8 == a1_p1_df8[0]:
            engine_ecu_temperature_df8 = a1_p2_df8[0]
        elif a1_df8 == a1_p1_df8[1]:
            engine_ecu_temperature_df8 = a1_p2_df8[1]
        elif a1_df8 == a1_p1_df8[2]:
            engine_ecu_temperature_df8 = a1_p2_df8[2]
        elif a1_df8 == a1_p1_df8[3]:
            engine_ecu_temperature_df8 = a1_p2_df8[3]
        elif a1_df8 == a1_p1_df8[4]:
            engine_ecu_temperature_df8 = a1_p2_df8[4]
    
    with col2:
        a2_m1_df8 = 3; a2_m2_df8 = 37; a2_s_df8 = 5
        a2_p1_df8 = list(np.arange(a2_m1_df8, a2_m2_df8, (a2_m2_df8 - a2_m1_df8)/a2_s_df8)) + [a2_m2_df8]
        for i in range(0, len(a2_p1_df8) - 1):
            a2_p1_df8[i] = str(int(a2_p1_df8[i])) + ' ≤ x < ' + str(int(a2_p1_df8[i + 1]))
        a2_p2_df8 = list(np.arange(a2_m1_df8, a2_m2_df8, (a2_m2_df8 - a2_m1_df8)/a2_s_df8))
        for i in range(0, len(a2_p2_df8)):
            a2_p2_df8[i] = round(a2_p2_df8[i] + ((a2_m2_df8 - a2_m1_df8)/a2_s_df8)/2, 1)
        
        a2_df8 = st.radio(
            '**fuel rate** (L/h)', options = [a2_p1_df8[0], a2_p1_df8[1], a2_p1_df8[2], a2_p1_df8[3], a2_p1_df8[4]], key = 'a2_df8'
        )
        engine_fuel_rate_df8 = 0
        if a2_df8 == a2_p1_df8[0]:
            engine_fuel_rate_df8 = a2_p2_df8[0]
        elif a2_df8 == a2_p1_df8[1]:
            engine_fuel_rate_df8 = a2_p2_df8[1]
        elif a2_df8 == a2_p1_df8[2]:
            engine_fuel_rate_df8 = a2_p2_df8[2]
        elif a2_df8 == a2_p1_df8[3]:
            engine_fuel_rate_df8 = a2_p2_df8[3]
        elif a2_df8 == a2_p1_df8[4]:
            engine_fuel_rate_df8 = a2_p2_df8[4]
    
    with col3:
        a3_m1_df8 = 4; a3_m2_df8 = 80; a3_s_df8 = 5
        a3_p1_df8 = list(np.arange(a3_m1_df8, a3_m2_df8, (a3_m2_df8 - a3_m1_df8)/a3_s_df8)) + [a3_m2_df8]
        for i in range(0, len(a3_p1_df8) - 1):
            a3_p1_df8[i] = str(int(a3_p1_df8[i])) + ' ≤ x < ' + str(int(a3_p1_df8[i + 1]))
        a3_p2_df8 = list(np.arange(a3_m1_df8, a3_m2_df8, (a3_m2_df8 - a3_m1_df8)/a3_s_df8))
        for i in range(0, len(a3_p2_df8)):
            a3_p2_df8[i] = round(a3_p2_df8[i] + ((a3_m2_df8 - a3_m1_df8)/a3_s_df8)/2, 1)
        
        a3_df8 = st.radio(
            '**IMP** (bar)', options = [a3_p1_df8[0], a3_p1_df8[1], a3_p1_df8[2], a3_p1_df8[3], a3_p1_df8[4]], key = 'a3_df8'
        )
        engine_intake_manifold_1_pressure_df8 = 0
        if a3_df8 == a3_p1_df8[0]:
            engine_intake_manifold_1_pressure_df8 = a3_p2_df8[0]
        elif a3_df8 == a3_p1_df8[1]:
            engine_intake_manifold_1_pressure_df8 = a3_p2_df8[1]
        elif a3_df8 == a3_p1_df8[2]:
            engine_intake_manifold_1_pressure_df8 = a3_p2_df8[2]
        elif a3_df8 == a3_p1_df8[3]:
            engine_intake_manifold_1_pressure_df8 = a3_p2_df8[3]
        elif a3_df8 == a3_p1_df8[4]:
            engine_intake_manifold_1_pressure_df8 = a3_p2_df8[4]
            
    with col4:
        a4_m1_df8 = 192; a4_m2_df8 = 348; a4_s_df8 = 5
        a4_p1_df8 = list(np.arange(a4_m1_df8, a4_m2_df8, (a4_m2_df8 - a4_m1_df8)/a4_s_df8)) + [a4_m2_df8]
        for i in range(0, len(a4_p1_df8) - 1):
            a4_p1_df8[i] = str(int(a4_p1_df8[i])) + ' ≤ x < ' + str(int(a4_p1_df8[i + 1]))
        a4_p2_df8 = list(np.arange(a4_m1_df8, a4_m2_df8, (a4_m2_df8 - a4_m1_df8)/a4_s_df8))
        for i in range(0, len(a4_p2_df8)):
            a4_p2_df8[i] = round(a4_p2_df8[i] + ((a4_m2_df8 - a4_m1_df8)/a4_s_df8)/2, 1)
        
        a4_df8 = st.radio(
            '**oil pressure** (bar)', options = [a4_p1_df8[0], a4_p1_df8[1], a4_p1_df8[2], a4_p1_df8[3], a4_p1_df8[4]], key = 'a4_df8'
        )
        engine_oil_pressure_df8 = 0
        if a4_df8 == a4_p1_df8[0]:
            engine_oil_pressure_df8 = a4_p2_df8[0]
        elif a4_df8 == a4_p1_df8[1]:
            engine_oil_pressure_df8 = a4_p2_df8[1]
        elif a4_df8 == a4_p1_df8[2]:
            engine_oil_pressure_df8 = a4_p2_df8[2]
        elif a4_df8 == a4_p1_df8[3]:
            engine_oil_pressure_df8 = a4_p2_df8[3]
        elif a4_df8 == a4_p1_df8[4]:
            engine_oil_pressure_df8 = a4_p2_df8[4]
    
    with col5:
        a5_m1_df8 = 83; a5_m2_df8 = 101; a5_s_df8 = 5
        a5_p1_df8 = list(np.arange(a5_m1_df8, a5_m2_df8, (a5_m2_df8 - a5_m1_df8)/a5_s_df8)) + [a5_m2_df8]
        for i in range(0, len(a5_p1_df8) - 1):
            a5_p1_df8[i] = str(int(a5_p1_df8[i])) + ' ≤ x < ' + str(int(a5_p1_df8[i + 1]))
        a5_p2_df8 = list(np.arange(a5_m1_df8, a5_m2_df8, (a5_m2_df8 - a5_m1_df8)/a5_s_df8))
        for i in range(0, len(a5_p2_df8)):
            a5_p2_df8[i] = round(a5_p2_df8[i] + ((a5_m2_df8 - a5_m1_df8)/a5_s_df8)/2, 1)
        
        a5_df8 = st.radio(
            '**oil temperature** (°C)', options = [a5_p1_df8[0], a5_p1_df8[1], a5_p1_df8[2], a5_p1_df8[3], a5_p1_df8[4]], key = 'a5_df8'
        )
        engine_oil_temperature_1_df8 = 0
        if a5_df8 == a5_p1_df8[0]:
            engine_oil_temperature_1_df8 = a5_p2_df8[0]
        elif a5_df8 == a5_p1_df8[1]:
            engine_oil_temperature_1_df8 = a5_p2_df8[1]
        elif a5_df8 == a5_p1_df8[2]:
            engine_oil_temperature_1_df8 = a5_p2_df8[2]
        elif a5_df8 == a5_p1_df8[3]:
            engine_oil_temperature_1_df8 = a5_p2_df8[3]
        elif a5_df8 == a5_p1_df8[4]:
            engine_oil_temperature_1_df8 = a5_p2_df8[4]
    
    with col6:
        a6_m1_df8 = 26.1; a6_m2_df8 = 28; a6_s_df8 = 5
        a6_p1_df8 = list(np.arange(a6_m1_df8, a6_m2_df8, (a6_m2_df8 - a6_m1_df8)/a6_s_df8)) + [a6_m2_df8]
        for i in range(0, len(a6_p1_df8) - 1):
            a6_p1_df8[i] = str(int(a6_p1_df8[i])) + ' ≤ x < ' + str(int(a6_p1_df8[i + 1]))
        a6_p2_df8 = list(np.arange(a6_m1_df8, a6_m2_df8, (a6_m2_df8 - a6_m1_df8)/a6_s_df8))
        for i in range(0, len(a6_p2_df8)):
            a6_p2_df8[i] = round(a6_p2_df8[i] + ((a6_m2_df8 - a6_m1_df8)/a6_s_df8)/2, 1)
        
        a6_df8 = st.radio(
            '**battery potential** (V)', options = [a6_p1_df8[0], a6_p1_df8[1], a6_p1_df8[2], a6_p1_df8[3], a6_p1_df8[4]], key = 'a6_df8'
        )
        keyswitch_battery_potential_df8 = 0
        if a6_df8 == a6_p1_df8[0]:
            keyswitch_battery_potential_df8 = a6_p2_df8[0]
        elif a6_df8 == a6_p1_df8[1]:
            keyswitch_battery_potential_df8 = a6_p2_df8[1]
        elif a6_df8 == a6_p1_df8[2]:
            keyswitch_battery_potential_df8 = a6_p2_df8[2]
        elif a6_df8 == a6_p1_df8[3]:
            keyswitch_battery_potential_df8 = a6_p2_df8[3]
        elif a6_df8 == a6_p1_df8[4]:
            keyswitch_battery_potential_df8 = a6_p2_df8[4]
    
    with col7:
        DF8 = pd.DataFrame({
            'keyswitch battery potential':[keyswitch_battery_potential_df8],
            'engine oil temperature 1':[engine_oil_temperature_1_df8],
            'engine oil pressure':[engine_oil_pressure_df8],
            'engine intake manifold 1 pressure':[engine_intake_manifold_1_pressure_df8],
            'engine fuel rate':[engine_fuel_rate_df8],
            'engine ecu temperature':[engine_ecu_temperature_df8]
        })
        
        result_df8 = model.predict(DF8)
        status_df8 = list(encoder_class.inverse_transform(np.array(int(result_df8)).ravel()))[0]
        
        st.subheader(':green[period 8]')
        st.write('Status: **:orange[' + status_df8 + ']**')
        status_ratio.append(status_df8)
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#9
with st.container():
    st.write('---')
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

    with col1:
        a1_m1_df9 = 41; a1_m2_df9 = 94; a1_s_df9 = 5
        a1_p1_df9 = list(np.arange(a1_m1_df9, a1_m2_df9, (a1_m2_df9 - a1_m1_df9)/a1_s_df9)) + [a1_m2_df9]
        for i in range(0, len(a1_p1_df9) - 1):
            a1_p1_df9[i] = str(int(a1_p1_df9[i])) + ' ≤ x < ' + str(int(a1_p1_df9[i + 1]))
        a1_p2_df9 = list(np.arange(a1_m1_df9, a1_m2_df9, (a1_m2_df9 - a1_m1_df9)/a1_s_df9))
        for i in range(0, len(a1_p2_df9)):
            a1_p2_df9[i] = round(a1_p2_df9[i] + ((a1_m2_df9 - a1_m1_df9)/a1_s_df9)/2, 1)
        
        a1_df9 = st.radio(
            '**ECU temp** (°C)', options = [a1_p1_df9[0], a1_p1_df9[1], a1_p1_df9[2], a1_p1_df9[3], a1_p1_df9[4]], key = 'a1_df9'
        )
        engine_ecu_temperature_df9 = 0
        if a1_df9 == a1_p1_df9[0]:
            engine_ecu_temperature_df9 = a1_p2_df9[0]
        elif a1_df9 == a1_p1_df9[1]:
            engine_ecu_temperature_df9 = a1_p2_df9[1]
        elif a1_df9 == a1_p1_df9[2]:
            engine_ecu_temperature_df9 = a1_p2_df9[2]
        elif a1_df9 == a1_p1_df9[3]:
            engine_ecu_temperature_df9 = a1_p2_df9[3]
        elif a1_df9 == a1_p1_df9[4]:
            engine_ecu_temperature_df9 = a1_p2_df9[4]
    
    with col2:
        a2_m1_df9 = 3; a2_m2_df9 = 37; a2_s_df9 = 5
        a2_p1_df9 = list(np.arange(a2_m1_df9, a2_m2_df9, (a2_m2_df9 - a2_m1_df9)/a2_s_df9)) + [a2_m2_df9]
        for i in range(0, len(a2_p1_df9) - 1):
            a2_p1_df9[i] = str(int(a2_p1_df9[i])) + ' ≤ x < ' + str(int(a2_p1_df9[i + 1]))
        a2_p2_df9 = list(np.arange(a2_m1_df9, a2_m2_df9, (a2_m2_df9 - a2_m1_df9)/a2_s_df9))
        for i in range(0, len(a2_p2_df9)):
            a2_p2_df9[i] = round(a2_p2_df9[i] + ((a2_m2_df9 - a2_m1_df9)/a2_s_df9)/2, 1)
        
        a2_df9 = st.radio(
            '**fuel rate** (L/h)', options = [a2_p1_df9[0], a2_p1_df9[1], a2_p1_df9[2], a2_p1_df9[3], a2_p1_df9[4]], key = 'a2_df9'
        )
        engine_fuel_rate_df9 = 0
        if a2_df9 == a2_p1_df9[0]:
            engine_fuel_rate_df9 = a2_p2_df9[0]
        elif a2_df9 == a2_p1_df9[1]:
            engine_fuel_rate_df9 = a2_p2_df9[1]
        elif a2_df9 == a2_p1_df9[2]:
            engine_fuel_rate_df9 = a2_p2_df9[2]
        elif a2_df9 == a2_p1_df9[3]:
            engine_fuel_rate_df9 = a2_p2_df9[3]
        elif a2_df9 == a2_p1_df9[4]:
            engine_fuel_rate_df9 = a2_p2_df9[4]
    
    with col3:
        a3_m1_df9 = 4; a3_m2_df9 = 80; a3_s_df9 = 5
        a3_p1_df9 = list(np.arange(a3_m1_df9, a3_m2_df9, (a3_m2_df9 - a3_m1_df9)/a3_s_df9)) + [a3_m2_df9]
        for i in range(0, len(a3_p1_df9) - 1):
            a3_p1_df9[i] = str(int(a3_p1_df9[i])) + ' ≤ x < ' + str(int(a3_p1_df9[i + 1]))
        a3_p2_df9 = list(np.arange(a3_m1_df9, a3_m2_df9, (a3_m2_df9 - a3_m1_df9)/a3_s_df9))
        for i in range(0, len(a3_p2_df9)):
            a3_p2_df9[i] = round(a3_p2_df9[i] + ((a3_m2_df9 - a3_m1_df9)/a3_s_df9)/2, 1)
        
        a3_df9 = st.radio(
            '**IMP** (bar)', options = [a3_p1_df9[0], a3_p1_df9[1], a3_p1_df9[2], a3_p1_df9[3], a3_p1_df9[4]], key = 'a3_df9'
        )
        engine_intake_manifold_1_pressure_df9 = 0
        if a3_df9 == a3_p1_df9[0]:
            engine_intake_manifold_1_pressure_df9 = a3_p2_df9[0]
        elif a3_df9 == a3_p1_df9[1]:
            engine_intake_manifold_1_pressure_df9 = a3_p2_df9[1]
        elif a3_df9 == a3_p1_df9[2]:
            engine_intake_manifold_1_pressure_df9 = a3_p2_df9[2]
        elif a3_df9 == a3_p1_df9[3]:
            engine_intake_manifold_1_pressure_df9 = a3_p2_df9[3]
        elif a3_df9 == a3_p1_df9[4]:
            engine_intake_manifold_1_pressure_df9 = a3_p2_df9[4]
            
    with col4:
        a4_m1_df9 = 192; a4_m2_df9 = 348; a4_s_df9 = 5
        a4_p1_df9 = list(np.arange(a4_m1_df9, a4_m2_df9, (a4_m2_df9 - a4_m1_df9)/a4_s_df9)) + [a4_m2_df9]
        for i in range(0, len(a4_p1_df9) - 1):
            a4_p1_df9[i] = str(int(a4_p1_df9[i])) + ' ≤ x < ' + str(int(a4_p1_df9[i + 1]))
        a4_p2_df9 = list(np.arange(a4_m1_df9, a4_m2_df9, (a4_m2_df9 - a4_m1_df9)/a4_s_df9))
        for i in range(0, len(a4_p2_df9)):
            a4_p2_df9[i] = round(a4_p2_df9[i] + ((a4_m2_df9 - a4_m1_df9)/a4_s_df9)/2, 1)
        
        a4_df9 = st.radio(
            '**oil pressure** (bar)', options = [a4_p1_df9[0], a4_p1_df9[1], a4_p1_df9[2], a4_p1_df9[3], a4_p1_df9[4]], key = 'a4_df9'
        )
        engine_oil_pressure_df9 = 0
        if a4_df9 == a4_p1_df9[0]:
            engine_oil_pressure_df9 = a4_p2_df9[0]
        elif a4_df9 == a4_p1_df9[1]:
            engine_oil_pressure_df9 = a4_p2_df9[1]
        elif a4_df9 == a4_p1_df9[2]:
            engine_oil_pressure_df9 = a4_p2_df9[2]
        elif a4_df9 == a4_p1_df9[3]:
            engine_oil_pressure_df9 = a4_p2_df9[3]
        elif a4_df9 == a4_p1_df9[4]:
            engine_oil_pressure_df9 = a4_p2_df9[4]
    
    with col5:
        a5_m1_df9 = 83; a5_m2_df9 = 101; a5_s_df9 = 5
        a5_p1_df9 = list(np.arange(a5_m1_df9, a5_m2_df9, (a5_m2_df9 - a5_m1_df9)/a5_s_df9)) + [a5_m2_df9]
        for i in range(0, len(a5_p1_df9) - 1):
            a5_p1_df9[i] = str(int(a5_p1_df9[i])) + ' ≤ x < ' + str(int(a5_p1_df9[i + 1]))
        a5_p2_df9 = list(np.arange(a5_m1_df9, a5_m2_df9, (a5_m2_df9 - a5_m1_df9)/a5_s_df9))
        for i in range(0, len(a5_p2_df9)):
            a5_p2_df9[i] = round(a5_p2_df9[i] + ((a5_m2_df9 - a5_m1_df9)/a5_s_df9)/2, 1)
        
        a5_df9 = st.radio(
            '**oil temperature** (°C)', options = [a5_p1_df9[0], a5_p1_df9[1], a5_p1_df9[2], a5_p1_df9[3], a5_p1_df9[4]], key = 'a5_df9'
        )
        engine_oil_temperature_1_df9 = 0
        if a5_df9 == a5_p1_df9[0]:
            engine_oil_temperature_1_df9 = a5_p2_df9[0]
        elif a5_df9 == a5_p1_df9[1]:
            engine_oil_temperature_1_df9 = a5_p2_df9[1]
        elif a5_df9 == a5_p1_df9[2]:
            engine_oil_temperature_1_df9 = a5_p2_df9[2]
        elif a5_df9 == a5_p1_df9[3]:
            engine_oil_temperature_1_df9 = a5_p2_df9[3]
        elif a5_df9 == a5_p1_df9[4]:
            engine_oil_temperature_1_df9 = a5_p2_df9[4]
    
    with col6:
        a6_m1_df9 = 26.1; a6_m2_df9 = 28; a6_s_df9 = 5
        a6_p1_df9 = list(np.arange(a6_m1_df9, a6_m2_df9, (a6_m2_df9 - a6_m1_df9)/a6_s_df9)) + [a6_m2_df9]
        for i in range(0, len(a6_p1_df9) - 1):
            a6_p1_df9[i] = str(int(a6_p1_df9[i])) + ' ≤ x < ' + str(int(a6_p1_df9[i + 1]))
        a6_p2_df9 = list(np.arange(a6_m1_df9, a6_m2_df9, (a6_m2_df9 - a6_m1_df9)/a6_s_df9))
        for i in range(0, len(a6_p2_df9)):
            a6_p2_df9[i] = round(a6_p2_df9[i] + ((a6_m2_df9 - a6_m1_df9)/a6_s_df9)/2, 1)
        
        a6_df9 = st.radio(
            '**battery potential** (V)', options = [a6_p1_df9[0], a6_p1_df9[1], a6_p1_df9[2], a6_p1_df9[3], a6_p1_df9[4]], key = 'a6_df9'
        )
        keyswitch_battery_potential_df9 = 0
        if a6_df9 == a6_p1_df9[0]:
            keyswitch_battery_potential_df9 = a6_p2_df9[0]
        elif a6_df9 == a6_p1_df9[1]:
            keyswitch_battery_potential_df9 = a6_p2_df9[1]
        elif a6_df9 == a6_p1_df9[2]:
            keyswitch_battery_potential_df9 = a6_p2_df9[2]
        elif a6_df9 == a6_p1_df9[3]:
            keyswitch_battery_potential_df9 = a6_p2_df9[3]
        elif a6_df9 == a6_p1_df9[4]:
            keyswitch_battery_potential_df9 = a6_p2_df9[4]
    
    with col7:
        DF9 = pd.DataFrame({
            'keyswitch battery potential':[keyswitch_battery_potential_df9],
            'engine oil temperature 1':[engine_oil_temperature_1_df9],
            'engine oil pressure':[engine_oil_pressure_df9],
            'engine intake manifold 1 pressure':[engine_intake_manifold_1_pressure_df9],
            'engine fuel rate':[engine_fuel_rate_df9],
            'engine ecu temperature':[engine_ecu_temperature_df9]
        })
        
        result_df9 = model.predict(DF9)
        status_df9 = list(encoder_class.inverse_transform(np.array(int(result_df9)).ravel()))[0]
        
        st.subheader(':green[period 9]')
        st.write('Status: **:orange[' + status_df9 + ']**')
        status_ratio.append(status_df9)
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#10
with st.container():
    st.write('---')
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

    with col1:
        a1_m1_df10 = 41; a1_m2_df10 = 94; a1_s_df10 = 5
        a1_p1_df10 = list(np.arange(a1_m1_df10, a1_m2_df10, (a1_m2_df10 - a1_m1_df10)/a1_s_df10)) + [a1_m2_df10]
        for i in range(0, len(a1_p1_df10) - 1):
            a1_p1_df10[i] = str(int(a1_p1_df10[i])) + ' ≤ x < ' + str(int(a1_p1_df10[i + 1]))
        a1_p2_df10 = list(np.arange(a1_m1_df10, a1_m2_df10, (a1_m2_df10 - a1_m1_df10)/a1_s_df10))
        for i in range(0, len(a1_p2_df10)):
            a1_p2_df10[i] = round(a1_p2_df10[i] + ((a1_m2_df10 - a1_m1_df10)/a1_s_df10)/2, 1)
        
        a1_df10 = st.radio(
            '**ECU temp** (°C)', options = [a1_p1_df10[0], a1_p1_df10[1], a1_p1_df10[2], a1_p1_df10[3], a1_p1_df10[4]], key = 'a1_df10'
        )
        engine_ecu_temperature_df10 = 0
        if a1_df10 == a1_p1_df10[0]:
            engine_ecu_temperature_df10 = a1_p2_df10[0]
        elif a1_df10 == a1_p1_df10[1]:
            engine_ecu_temperature_df10 = a1_p2_df10[1]
        elif a1_df10 == a1_p1_df10[2]:
            engine_ecu_temperature_df10 = a1_p2_df10[2]
        elif a1_df10 == a1_p1_df10[3]:
            engine_ecu_temperature_df10 = a1_p2_df10[3]
        elif a1_df10 == a1_p1_df10[4]:
            engine_ecu_temperature_df10 = a1_p2_df10[4]
    
    with col2:
        a2_m1_df10 = 3; a2_m2_df10 = 37; a2_s_df10 = 5
        a2_p1_df10 = list(np.arange(a2_m1_df10, a2_m2_df10, (a2_m2_df10 - a2_m1_df10)/a2_s_df10)) + [a2_m2_df10]
        for i in range(0, len(a2_p1_df10) - 1):
            a2_p1_df10[i] = str(int(a2_p1_df10[i])) + ' ≤ x < ' + str(int(a2_p1_df10[i + 1]))
        a2_p2_df10 = list(np.arange(a2_m1_df10, a2_m2_df10, (a2_m2_df10 - a2_m1_df10)/a2_s_df10))
        for i in range(0, len(a2_p2_df10)):
            a2_p2_df10[i] = round(a2_p2_df10[i] + ((a2_m2_df10 - a2_m1_df10)/a2_s_df10)/2, 1)
        
        a2_df10 = st.radio(
            '**fuel rate** (L/h)', options = [a2_p1_df10[0], a2_p1_df10[1], a2_p1_df10[2], a2_p1_df10[3], a2_p1_df10[4]], key = 'a2_df10'
        )
        engine_fuel_rate_df10 = 0
        if a2_df10 == a2_p1_df10[0]:
            engine_fuel_rate_df10 = a2_p2_df10[0]
        elif a2_df10 == a2_p1_df10[1]:
            engine_fuel_rate_df10 = a2_p2_df10[1]
        elif a2_df10 == a2_p1_df10[2]:
            engine_fuel_rate_df10 = a2_p2_df10[2]
        elif a2_df10 == a2_p1_df10[3]:
            engine_fuel_rate_df10 = a2_p2_df10[3]
        elif a2_df10 == a2_p1_df10[4]:
            engine_fuel_rate_df10 = a2_p2_df10[4]
    
    with col3:
        a3_m1_df10 = 4; a3_m2_df10 = 80; a3_s_df10 = 5
        a3_p1_df10 = list(np.arange(a3_m1_df10, a3_m2_df10, (a3_m2_df10 - a3_m1_df10)/a3_s_df10)) + [a3_m2_df10]
        for i in range(0, len(a3_p1_df10) - 1):
            a3_p1_df10[i] = str(int(a3_p1_df10[i])) + ' ≤ x < ' + str(int(a3_p1_df10[i + 1]))
        a3_p2_df10 = list(np.arange(a3_m1_df10, a3_m2_df10, (a3_m2_df10 - a3_m1_df10)/a3_s_df10))
        for i in range(0, len(a3_p2_df10)):
            a3_p2_df10[i] = round(a3_p2_df10[i] + ((a3_m2_df10 - a3_m1_df10)/a3_s_df10)/2, 1)
        
        a3_df10 = st.radio(
            '**IMP** (bar)', options = [a3_p1_df10[0], a3_p1_df10[1], a3_p1_df10[2], a3_p1_df10[3], a3_p1_df10[4]], key = 'a3_df10'
        )
        engine_intake_manifold_1_pressure_df10 = 0
        if a3_df10 == a3_p1_df10[0]:
            engine_intake_manifold_1_pressure_df10 = a3_p2_df10[0]
        elif a3_df10 == a3_p1_df10[1]:
            engine_intake_manifold_1_pressure_df10 = a3_p2_df10[1]
        elif a3_df10 == a3_p1_df10[2]:
            engine_intake_manifold_1_pressure_df10 = a3_p2_df10[2]
        elif a3_df10 == a3_p1_df10[3]:
            engine_intake_manifold_1_pressure_df10 = a3_p2_df10[3]
        elif a3_df10 == a3_p1_df10[4]:
            engine_intake_manifold_1_pressure_df10 = a3_p2_df10[4]
            
    with col4:
        a4_m1_df10 = 192; a4_m2_df10 = 348; a4_s_df10 = 5
        a4_p1_df10 = list(np.arange(a4_m1_df10, a4_m2_df10, (a4_m2_df10 - a4_m1_df10)/a4_s_df10)) + [a4_m2_df10]
        for i in range(0, len(a4_p1_df10) - 1):
            a4_p1_df10[i] = str(int(a4_p1_df10[i])) + ' ≤ x < ' + str(int(a4_p1_df10[i + 1]))
        a4_p2_df10 = list(np.arange(a4_m1_df10, a4_m2_df10, (a4_m2_df10 - a4_m1_df10)/a4_s_df10))
        for i in range(0, len(a4_p2_df10)):
            a4_p2_df10[i] = round(a4_p2_df10[i] + ((a4_m2_df10 - a4_m1_df10)/a4_s_df10)/2, 1)
        
        a4_df10 = st.radio(
            '**oil pressure** (bar)', options = [a4_p1_df10[0], a4_p1_df10[1], a4_p1_df10[2], a4_p1_df10[3], a4_p1_df10[4]], key = 'a4_df10'
        )
        engine_oil_pressure_df10 = 0
        if a4_df10 == a4_p1_df10[0]:
            engine_oil_pressure_df10 = a4_p2_df10[0]
        elif a4_df10 == a4_p1_df10[1]:
            engine_oil_pressure_df10 = a4_p2_df10[1]
        elif a4_df10 == a4_p1_df10[2]:
            engine_oil_pressure_df10 = a4_p2_df10[2]
        elif a4_df10 == a4_p1_df10[3]:
            engine_oil_pressure_df10 = a4_p2_df10[3]
        elif a4_df10 == a4_p1_df10[4]:
            engine_oil_pressure_df10 = a4_p2_df10[4]
    
    with col5:
        a5_m1_df10 = 83; a5_m2_df10 = 101; a5_s_df10 = 5
        a5_p1_df10 = list(np.arange(a5_m1_df10, a5_m2_df10, (a5_m2_df10 - a5_m1_df10)/a5_s_df10)) + [a5_m2_df10]
        for i in range(0, len(a5_p1_df10) - 1):
            a5_p1_df10[i] = str(int(a5_p1_df10[i])) + ' ≤ x < ' + str(int(a5_p1_df10[i + 1]))
        a5_p2_df10 = list(np.arange(a5_m1_df10, a5_m2_df10, (a5_m2_df10 - a5_m1_df10)/a5_s_df10))
        for i in range(0, len(a5_p2_df10)):
            a5_p2_df10[i] = round(a5_p2_df10[i] + ((a5_m2_df10 - a5_m1_df10)/a5_s_df10)/2, 1)
        
        a5_df10 = st.radio(
            '**oil temperature** (°C)', options = [a5_p1_df10[0], a5_p1_df10[1], a5_p1_df10[2], a5_p1_df10[3], a5_p1_df10[4]], key = 'a5_df10'
        )
        engine_oil_temperature_1_df10 = 0
        if a5_df10 == a5_p1_df10[0]:
            engine_oil_temperature_1_df10 = a5_p2_df10[0]
        elif a5_df10 == a5_p1_df10[1]:
            engine_oil_temperature_1_df10 = a5_p2_df10[1]
        elif a5_df10 == a5_p1_df10[2]:
            engine_oil_temperature_1_df10 = a5_p2_df10[2]
        elif a5_df10 == a5_p1_df10[3]:
            engine_oil_temperature_1_df10 = a5_p2_df10[3]
        elif a5_df10 == a5_p1_df10[4]:
            engine_oil_temperature_1_df10 = a5_p2_df10[4]
    
    with col6:
        a6_m1_df10 = 26.1; a6_m2_df10 = 28; a6_s_df10 = 5
        a6_p1_df10 = list(np.arange(a6_m1_df10, a6_m2_df10, (a6_m2_df10 - a6_m1_df10)/a6_s_df10)) + [a6_m2_df10]
        for i in range(0, len(a6_p1_df10) - 1):
            a6_p1_df10[i] = str(int(a6_p1_df10[i])) + ' ≤ x < ' + str(int(a6_p1_df10[i + 1]))
        a6_p2_df10 = list(np.arange(a6_m1_df10, a6_m2_df10, (a6_m2_df10 - a6_m1_df10)/a6_s_df10))
        for i in range(0, len(a6_p2_df10)):
            a6_p2_df10[i] = round(a6_p2_df10[i] + ((a6_m2_df10 - a6_m1_df10)/a6_s_df10)/2, 1)
        
        a6_df10 = st.radio(
            '**battery potential** (V)', options = [a6_p1_df10[0], a6_p1_df10[1], a6_p1_df10[2], a6_p1_df10[3], a6_p1_df10[4]], key = 'a6_df10'
        )
        keyswitch_battery_potential_df10 = 0
        if a6_df10 == a6_p1_df10[0]:
            keyswitch_battery_potential_df10 = a6_p2_df10[0]
        elif a6_df10 == a6_p1_df10[1]:
            keyswitch_battery_potential_df10 = a6_p2_df10[1]
        elif a6_df10 == a6_p1_df10[2]:
            keyswitch_battery_potential_df10 = a6_p2_df10[2]
        elif a6_df10 == a6_p1_df10[3]:
            keyswitch_battery_potential_df10 = a6_p2_df10[3]
        elif a6_df10 == a6_p1_df10[4]:
            keyswitch_battery_potential_df10 = a6_p2_df10[4]
    
    with col7:
        DF10 = pd.DataFrame({
            'keyswitch battery potential':[keyswitch_battery_potential_df10],
            'engine oil temperature 1':[engine_oil_temperature_1_df10],
            'engine oil pressure':[engine_oil_pressure_df10],
            'engine intake manifold 1 pressure':[engine_intake_manifold_1_pressure_df10],
            'engine fuel rate':[engine_fuel_rate_df10],
            'engine ecu temperature':[engine_ecu_temperature_df10]
        })
        
        result_df10 = model.predict(DF10)
        status_df10 = list(encoder_class.inverse_transform(np.array(int(result_df10)).ravel()))[0]
        
        st.subheader(':green[period 10]')
        st.write('Status: **:orange[' + status_df10 + ']**')
        status_ratio.append(status_df10)
#----------------------------------------------------------------#


#----------------------------------------------------------------#
with st.container():
    st.write('---')
    col1, col2, col3, col4, col5 = st.columns([3, 1, 2, 1, 3])
    
    with col1:
        st.subheader(':blue[Status ratio]')
        
        stA1 = str('||||' * status_ratio.count('A'))
        stA2 = str(status_ratio.count('A') * 10) + '%'
        st.subheader(':violet[A]: ' + ':green[' + stA1 + ']  ' + stA2)
        
        st.text('')
        st.text('')
        
        stB1 = str('||||' * status_ratio.count('B'))
        stB2 = str(status_ratio.count('B') * 10) + '%'
        st.subheader(':violet[B]: ' + ':orange[' + stB1 + ']  ' + stB2)
        
        st.text('')
        st.text('')
        
        stC1 = str('||||' * status_ratio.count('C'))
        stC2 = str(status_ratio.count('C') * 10) + '%'
        st.subheader(':violet[C]: ' + ':red[' + stC1 + ']  ' + stC2)
    
    with col3:
        st.subheader(':blue[Adjustable cutoff]')
        A_cutoff = st.slider(
            '**Min status A**', min_value = 30, max_value = 100,
            value = 70, step = 10, key = 'min_cutoff_A'
        )
        if A_cutoff == 100:
            st.write('')
            st.write('**Max status B = 0**')
            st.write('')
            st.write('')
            st.write('**Max status C = 0**')
            B_cutoff = 0
            C_cutoff = 0
        if A_cutoff <= 90:
            B_cutoff = st.slider(
                '**Max status B**', min_value = 0, max_value = 100 - A_cutoff,
                value = 0, step = 10, key = 'max_cutoff_B'
            )
            if A_cutoff + B_cutoff >= 100:
                st.write('')
                st.write('**Max status C = 0**')
                C_cutoff = 0
            if A_cutoff + B_cutoff < 100:
                C_cutoff = st.slider(
                    '**Max status C**', min_value = 0, max_value = 100 - A_cutoff - B_cutoff,
                    value = 0, step = 10, key = 'max_cutoff_C'
                )
    
    with col5:
        st.subheader(':blue[Recommendation]')
        
        if status_ratio.count('A') * 10 >= A_cutoff:
            if status_ratio.count('B') * 10 <= B_cutoff:
                if status_ratio.count('C') * 10 <= C_cutoff:
                    st.text('')
                    st.text('')
                    st.subheader(':green[no maintenance required]')
                else:
                    st.text('')
                    st.text('')
                    st.subheader(':red[urgent maintenance required]')
            else:
                if status_ratio.count('C') * 10 <= C_cutoff:
                    st.text('')
                    st.text('')
                    st.subheader(':orange[maintenance required]')
                else:
                    st.text('')
                    st.text('')
                    st.subheader(':red[urgent maintenance required]')
        else:
            if status_ratio.count('B') * 10 <= B_cutoff:
                if status_ratio.count('C') * 10 <= C_cutoff:
                    st.text('')
                    st.text('')
                    st.subheader(':orange[maintenance required]')
                else:
                    st.text('')
                    st.text('')
                    st.subheader(':red[urgent maintenance required]')
            else:
                if status_ratio.count('C') * 10 <= C_cutoff:
                    st.text('')
                    st.text('')
                    st.subheader(':orange[maintenance required]')
                else:
                    st.text('')
                    st.text('')
                    st.subheader(':red[urgent maintenance required]')
             
        
        
    
#----------------------------------------------------------------#

st.write('---')

