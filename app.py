import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Weather Prediction", layout="wide")

def load_data():
    df = pd.read_csv('dataset/for_trained_weatherAUS.csv')
    df_c = pd.read_csv('dataset/weather_cleaned.csv')
    return df, df_c

def load_model():
    model = joblib.load('model/rain_prediction_model.pkl')
    scaler = joblib.load('model/scaler_weather.pkl')
    packet = joblib.load('model/rain_prediction_model_accuracy.pkl')
    return model, scaler, packet

df, df_c = load_data()
model, scaler, packet = load_model()

model = packet['model_obj']
acc_score = packet['accuracy']

month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
df['MonthName'] = df['Month'].map(month_map)
list_kota = sorted(df_c['Location'].unique())

presentage = df_c["RainTomorrow"].map({'Yes':'Rain','No':'Bright'})
prob = presentage.value_counts()

st.sidebar.info(f"Algorithm: Random Forest - Model Accuracy: {acc_score * 100:.2f}%")
st.sidebar.markdown("---")
st.sidebar.title("Navigation Wheater")
menu = st.sidebar.radio("Menu", ["Dashboard", "Prediction"])

if menu == "Dashboard":
    st.title("Weather Dashboard")
    st.title("Summary And Probability of Rain in Australia")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total City", df_c['Location'].nunique())
    c2.metric("Total Rows", f"{len(df):,}")
    c3.metric("Total Columns", len(df.columns))
    st.markdown("---")

    st.subheader("Probability of Rain in Australia")
    st.plotly_chart(px.pie(prob, values=prob.values, names=prob.index,title='Probability of Rain',height=500,))
    st.write("Insight")
    st.info("Historical data shows Australia is a dry continent. Rain only occurs on ~22%")
    st.markdown("---")

    st.subheader("Correlation Between Columns")
    cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure3pm', 'RainToday', 'RainTomorrow']
    corr = df[cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto='.2f',height=600, color_continuous_scale='YlGnBu'), use_container_width=True)
    st.markdown("---")

    st.subheader("Most Influential Column")
    imp = pd.DataFrame({'feature': model.feature_names_in_, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    st.plotly_chart(px.bar(imp.head(10), x='importance', y='feature', orientation='h'), use_container_width=True)
    st.write("Humidity at 3 PM is the most important thing about its atmosphere ehich condition in the air is hot from water vaporation")
    st.markdown("---")

    st.subheader("Analysis by City")
    kota = st.selectbox("Select", list_kota)
    df_kota = df[df['Location'] == list_kota.index(kota)]
    st.markdown("---")
    tab1, tab2 = st.tabs(['Rainfall','Temperature'])
    with tab1:
        col_a, col_b = st.columns(2)
        avg_r = df_kota['Rainfall'].mean()
        col_a.info(f"Avarage Rain in {kota}: {avg_r:.2f} mm")
        rainy_month = df_kota.groupby('MonthName')['Rainfall'].mean().sort_values(ascending=False).index[0]
        col_b.success(f"Wettest Month: {rainy_month}")
        month_rain = df_kota.groupby('MonthName')['Rainfall'].mean().reset_index()
        st.plotly_chart(px.line(
            month_rain, 
            x='MonthName', 
            y='Rainfall', 
            title=f"Monthly Rain Trend - {kota}"
            ), use_container_width=True)
    with tab2:
        col1, col2 = st.columns(2)
        avg_t = df_kota['MaxTemp'].mean()
        col1.info(f"Avarage Temp in {kota}: {avg_t:.2f} C")
        hot_month = df_kota.groupby('MonthName')['MaxTemp'].max().sort_values(ascending=False).index[0]
        col2.success(f"Hottest Temp in {kota}: {hot_month}")
        month_temp = df_kota.groupby('MonthName')['MaxTemp'].mean().reset_index()
        st.plotly_chart(px.line(
            month_temp, 
            x='MonthName', 
            y='MaxTemp', 
            title=f"Monthly Temp Trend - {kota}"
            ), use_container_width=True)


else:
    st.title("Rain Prediction for Tomorrow")
    st.info("Fill in the weather data below to predict rain tomorrow")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        pk_kota = st.selectbox("City", list_kota)
        min_t = st.number_input("Min Temp", value=15.0)
        max_t = st.number_input("Max Temp", value=25.0)
    with c2:
        h9 = st.slider("Humidity 9am", 0, 100, 60)
        h3 = st.slider("Humidity 3pm", 0, 100, 40)
        p3 = st.number_input("Pressure 3pm", value=1015.0)
    with c3:
        rain_now = st.number_input("Rainfall Today", value=0.0)
        wind = st.number_input("Wind Gust Speed", value=30.0)
        # today = st.radio("Raining Today?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        bln = st.selectbox("Month", range(1, 13))

    if st.button("Prediction", use_container_width=True):
        loc_id = list_kota.index(pk_kota)
        
        input_raw = pd.DataFrame([[
            loc_id, min_t, max_t, np.log1p(rain_now), np.log1p(wind), h9, h3, p3, bln
        ]], columns=['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure3pm', 'Month'])

        for c in model.feature_names_in_:
            if c not in input_raw.columns: input_raw[c] = 0
        
        input_final = input_raw[model.feature_names_in_]
        input_sc = scaler.transform(input_final)
        
        prob = model.predict_proba(input_sc)[0][1]
        
        if prob >= 0.35:
            st.error(f"Result: Rain {prob * 100:.1f}%")
        else:
            st.success(f"Result: Bright {prob * 100:.1f}%")

