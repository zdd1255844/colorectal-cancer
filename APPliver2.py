import pandas as pd
import pickle
import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np

age_options={
    1:'<60(1)',
    2:'≥60(2)'
}

sex_options={
    1:'Male(1)',
    2:'Female(2)'
}

histologictype_options={
    1:'Adenocarcinoma(1)',
    2:'Mucinous adenocarcinoma(2)',
    3:'Carcinoid tumor (3)',
    4:'Signet ring cell carcinoma(4)'
    }

maritalstatus_options={
    1:'Married(1)',
    2:'Single(2)',
    3:'Unknown(3)'
}


grade_options={
    1:'Well differentiated(1)',
    2:'Moderately differentiated(2)',
    3:'Poorly differentiated(3)',
    4:'Undifferentiated(4)'
}



primarysite_options={
    2:'Cecum(2)',
    3:'Colon(3)',
    4:'Rectosigmoid junction(4)',
    5:'Rectum(5)'
}


tumorsize_options={
    1:'<5(1)',
    2:'≥5 (2)'
}

CEA_options={ 
 1:'Negative(1)',
 2:'Positive(2)'
 }


Tstage_options={ 
 1:'T0-T1(1)',
 2:'T2(2)',
 3:'T3(3)',
 4:'T4(4)'
 }


Nstage_options={ 
 1:'N0(1)',
 2:'N1(2)',
 3:'N2(3)'
 }

tumordeposits_options={
    1:'No(1)',
    2:'Yes(2)'
    }


feature_names = [ "age", "sex", "histologictype","maritalstatus", "grade","primarysite",  "tumorsize","CEA","Tstage","Nstage","tumordeposits"]

st.header(" Liver metastasis of colorectal cancer predictor app")

age=st.selectbox("age:", options=list(age_options.keys()), format_func=lambda x: age_options[x])
sex = st.selectbox("sex ( 1=Male, 2=Female):", options=[1, 2], format_func=lambda x: 'Female (2)' if x == 2 else 'Male (1)')
histologictype=st.selectbox("histologic type:", options=list(histologictype_options.keys()), format_func=lambda x: histologictype_options[x])
maritalstauts=st.selectbox("marital status:", options=list(maritalstatus_options.keys()), format_func=lambda x: maritalstatus_options[x])
grade=st.selectbox("grade:", options=list(grade_options.keys()), format_func=lambda x: grade_options[x])
primarysite=st.selectbox("primary site:", options=list(primarysite_options.keys()), format_func=lambda x: primarysite_options[x])
tumorsize=st.selectbox("tumor size:", options=list(tumorsize_options.keys()), format_func=lambda x: tumorsize_options[x])
CEA=st.selectbox("CEA:", options=list(CEA_options.keys()), format_func=lambda x: CEA_options[x])
Tstage=st.selectbox("T stage:", options=list(Tstage_options.keys()), format_func=lambda x: Tstage_options[x])
Nstage=st.selectbox("N stage:", options=list(Nstage_options.keys()), format_func=lambda x: Nstage_options[x])
tumordeposits=st.selectbox("tumor deposits:", options=list(tumordeposits_options.keys()), format_func=lambda x: tumordeposits_options[x])

feature_values = [age, sex, histologictype, maritalstauts,grade, primarysite,tumorsize,CEA,Tstage,Nstage,tumordeposits]   

features = np.array([feature_values])

if st.button("Submit"):
    clf = open("xgboostliver.pkl","rb")
    s=pickle.load(clf)
    predicted_class = s.predict(features)[0]
    predicted_proba = s.predict_proba(features)[0]
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    aaa=100
    probability2=aaa-probability
    if predicted_class == 1:
        advice = (
            f"The model predicts that your probability of having liver metastasis is {probability:.1f}%"
            )
    else:
        advice = (
            f"The model predicts that your probability of having liver metastasis is {probability2:.1f}%" 
            )
    st.write(advice)

    explainer = shap.Explainer(s)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")


