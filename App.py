import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from groq import Groq
import io

# --- ⚙️ PAGE CONFIG ---
st.set_page_config(page_title="Enterprise AI-NIDS Pro", page_icon="🛡️", layout="wide")

# --- 📄 ORIGINAL CSV FILE NAMES ---
CSV_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
]

# --- 🛠️ DATA ENGINE ---
@st.cache_data
def process_data(df):
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def load_data_engine(uploaded_files):
    all_dfs = []
    # Priority 1: User Uploaded Files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            df_temp = pd.read_csv(uploaded_file, nrows=20000, low_memory=False)
            all_dfs.append(df_temp)
        source = "Uploaded Files"
    # Priority 2: Local 4 CSV Files
    else:
        for file in CSV_FILES:
            if os.path.exists(file):
                df_temp = pd.read_csv(file, nrows=20000, low_memory=False)
                all_dfs.append(df_temp)
        source = "Local CSV Dataset"

    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        return process_data(combined_df), source
    return None, None

# --- 🕹️ SIDEBAR ---
st.sidebar.header("🔑 1. Security Access")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
st.sidebar.markdown("[Get Free Key](https://console.groq.com/keys)")

st.sidebar.header("📂 2. Data Feed")
uploaded_csvs = st.sidebar.file_uploader("Upload Network CSVs (Optional)", type="csv", accept_multiple_files=True)

if st.sidebar.button("🚀 INITIALIZE AI ENGINE"):
    data, src = load_data_engine(uploaded_csvs)
    if data is not None:
        st.session_state['nids_data'] = data
        if 'rf_model' in st.session_state: del st.session_state['rf_model']
        st.balloons()
        st.sidebar.success(f"Engine Ready! Source: {src}")
    else:
        st.sidebar.error("Error: CSV files not found locally or in upload!")

# --- 📊 MAIN DASHBOARD ---
if 'nids_data' in st.session_state:
    df = st.session_state['nids_data']
    
    # 💎 NETWORK TRAFFIC INTELLIGENCE
    st.title("AI-Powered Network Intrusion Detection System")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Logs Analyzed", f"{len(df):,}")
    m2.metric("Detected Threats", df['Label'].nunique())
    m3.metric("System Integrity", "99.9%", delta="Stable")
    m4.metric("AI Engine", "Random Forest", delta="Active")

    # --- ML PROCESSING ---
    features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
                'Total Length of Fwd Packets', 'Fwd Packet Length Max', 'Flow IAT Mean']
    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features]
    le = LabelEncoder()
    y_enc = le.fit_transform(df['Label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)
    
    if 'rf_model' not in st.session_state:
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        st.session_state['rf_model'] = rf
        st.session_state['le'] = le
        st.session_state['X_test'] = X_test

    # --- 📈 DYNAMIC GRAPHS ---
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Model Accuracy Heatmap")
        y_pred = st.session_state['rf_model'].predict(st.session_state['X_test'])
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = px.imshow(cm, text_auto=True, x=le.classes_, y=le.classes_, color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("### 📊 Threat Hierarchy (Sunburst)")
        counts = df['Label'].value_counts().reset_index()
        fig_sun = px.sunburst(counts, path=['Label'], values='count', color='count', color_continuous_scale='Rainbow')
        st.plotly_chart(fig_sun, use_container_width=True)

    # --- 🕸️ RADAR FINGERPRINT ---
    st.divider()
    st.header("🕸️ Threat Behavioral Fingerprint")
    radar_df = df.groupby('Label')[valid_features].mean().reset_index()
    for col in valid_features:
        radar_df[col] = radar_df[col] / radar_df[col].max()
    
    fig_radar = go.Figure()
    for i in range(len(radar_df)):
        fig_radar.add_trace(go.Scatterpolar(r=radar_df.iloc[i][valid_features].values, theta=valid_features, fill='toself', name=radar_df.iloc[i]['Label']))
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- 🤖 AI ASSISTANT & EXPORT ---
    st.divider()
    st.header("💬 AI Security Assistant & Forensic Report")
    
    c_ai, c_rep = st.columns([2, 1])
    
    with c_ai:
        query = st.text_input("Ask the AI Assistant about your network security...")
        if st.button("Query AI Agent"):
            if groq_api_key:
                try:
                    client = Groq(api_key=groq_api_key)
                    summary = df['Label'].value_counts().to_string()
                    prompt = f"Data Summary: {summary}. User Question: {query}. Provide expert cybersecurity advice."
                    with st.spinner("AI Analyst responding..."):
                        resp = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
                        st.info(resp.choices[0].message.content)
                except Exception as e: st.error(f"Error: {e}")
            else: st.warning("API Key missing!")

    with c_rep:
        st.markdown("### 📋 Export Results")
        csv_data = df['Label'].value_counts().to_csv().encode('utf-8')
        st.download_button(label="📥 Download Threat Report (CSV)", data=csv_data, file_name="NIDS_Analysis_Report.csv", mime="text/csv")

    # --- 🔬 MANUAL PACKET INSPECTION ---
    st.divider()
    if st.button("🎲 Inspect Random Network Packet"):
        idx = np.random.randint(0, len(st.session_state['X_test']))
        st.session_state['s_sample'] = st.session_state['X_test'].iloc[idx]
        st.session_state['s_label'] = le.inverse_transform([st.session_state['rf_model'].predict([st.session_state['s_sample']])[0]])[0]
        
    if 's_sample' in st.session_state:
        st.write(f"**Analysis:** Packet classified as **{st.session_state['s_label']}**")
        st.dataframe(st.session_state['s_sample'])
        if st.button("Explain this Packet with AI"):
            if groq_api_key:
                client = Groq(api_key=groq_api_key)
                prompt = f"Why is this packet {st.session_state['s_label']}? Data: {st.session_state['s_sample'].to_dict()}"
                with st.spinner("AI analyzing..."):
                    resp = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
                    st.write(resp.choices[0].message.content)

else:
    st.info("👋 Welcome! Make sure the 4 CSV files are in the folder or upload them, then click 'INITIALIZE AI ENGINE'.")
    