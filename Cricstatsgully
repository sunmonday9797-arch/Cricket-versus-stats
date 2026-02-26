import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Cricket Analytics", page_icon="🏏", layout="wide")
st.title("🏏 Cricket Stats Dashboard")

# --- LOAD DATA ---
FILE = "cricket_summary_data.csv"
@st.cache_data 
def load_data():
    if os.path.exists(FILE):
        return pd.read_csv(FILE)
    return pd.DataFrame()

df = load_data()

# --- MAIN DASHBOARD AREA ---
tab1, tab2 = st.tabs(["📊 Data & Controls", "🤖 AI Analyst"])

with tab1:
    st.markdown("### 🔍 Search Matches")
    col1, col2, col3 = st.columns(3)
    p1 = col1.text_input("Player 1:")
    p2 = col2.text_input("Player 2 (VS):")
    date_filter = col3.text_input("Date (Y-M-D):")

    st.markdown("### ⚙️ View Options")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    
    view_mode = ctrl1.radio("Mode:", ["All", "Batting", "Bowling"], horizontal=True)
    is_combined = ctrl2.checkbox("Combined View")
    is_recent = ctrl3.checkbox("Recent Form")
    
    recent_n = 3
    if is_recent:
        recent_n = ctrl3.number_input("How many recent matches?", min_value=1, max_value=100, value=3, step=1)

    # --- FILTER & CALCULATION LOGIC ---
    filtered_df = df.copy()

    if not filtered_df.empty:
        # Date Filter
        if date_filter:
            filtered_df = filtered_df[filtered_df['Date'].astype(str).str.contains(date_filter, case=False, na=False)]

        # Player 1 & Mode Filter
        if p1:
            m1 = filtered_df['Batsman'].str.contains(p1, case=False, na=False)
            v1 = filtered_df['Bowler'].str.contains(p1, case=False, na=False)
            if view_mode == "Batting": filtered_df = filtered_df[m1]
            elif view_mode == "Bowling": filtered_df = filtered_df[v1]
            else: filtered_df = filtered_df[m1 | v1]
        
        # Player 2 & Mode Filter
        if p2:
            m2 = filtered_df['Batsman'].str.contains(p2, case=False, na=False)
            v2 = filtered_df['Bowler'].str.contains(p2, case=False, na=False)
            if view_mode == "Batting": filtered_df = filtered_df[v2] 
            elif view_mode == "Bowling": filtered_df = filtered_df[m2] 
            else: filtered_df = filtered_df[m2 | v2]

        # Recent Form Filter
        if is_recent:
            filtered_df = filtered_df.groupby(['Batsman', 'Bowler']).tail(recent_n)

        # Base formatting to ensure math works
        for col in ['Runs', 'Balls', 'Wickets']:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)

        # COMBINED OR SINGLE VIEW LOGIC
        if is_combined:
            filtered_df['Inns'] = 1 
            agg_dict = {'Inns': 'sum', 'Runs': 'sum', 'Balls': 'sum', 'Wickets': 'sum'}
            if 'Boundaries' in filtered_df.columns:
                filtered_df['Boundaries'] = pd.to_numeric(filtered_df['Boundaries'], errors='coerce').fillna(0)
                agg_dict['Boundaries'] = 'sum'
            filtered_df = filtered_df.groupby(['Batsman', 'Bowler']).agg(agg_dict).reset_index()
        else:
            # If not combined, explicitly add an Inns column so it's always visible
            if 'Inns' not in filtered_df.columns:
                filtered_df.insert(3, 'Inns', 1)

        # ADVANCED METRICS 
        filtered_df['BatSR'] = np.where(filtered_df['Balls'] > 0, (filtered_df['Runs'] / filtered_df['Balls'] * 100).round(1), 0)
        filtered_df['BatAvg'] = np.where(filtered_df['Wickets'] > 0, (filtered_df['Runs'] / filtered_df['Wickets']).round(1), "-")
        filtered_df['BwlSR'] = np.where(filtered_df['Wickets'] > 0, (filtered_df['Balls'] / filtered_df['Wickets']).round(1), "-")
        filtered_df['BwlAvg'] = np.where(filtered_df['Wickets'] > 0, (filtered_df['Runs'] / filtered_df['Wickets']).round(1), "-")
        filtered_df['Econ'] = np.where(filtered_df['Balls'] > 0, (filtered_df['Runs'] / (filtered_df['Balls'] / 6)).round(2), 0)

    # --- DISPLAY TABLE ---
    st.markdown("### 📋 Match Data")
    
    if filtered_df.empty:
        st.warning("No data found. Check your filters or ensure 'cricket_summary_data.csv' is uploaded.")
    else:
        # THE ROBUST COLUMN CUSTOMIZER (EYE BUTTON)
        available_columns = filtered_df.columns.tolist()
        
        with st.expander("👁️ Customize Table Columns"):
            selected_columns = st.multiselect(
                "Add or remove columns to change what the table shows:", 
                options=available_columns, 
                default=available_columns
            )
        
        # Display the main table using ONLY the columns the user selected
        if selected_columns:
            st.dataframe(filtered_df[selected_columns], use_container_width=True, hide_index=True)
        else:
            st.error("Please select at least one column to display the table.")

        if p1 and p2 and not is_combined:
            if st.button(f"📊 Generate Matchup Graph: {p1} vs {p2}"):
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(1, len(filtered_df) + 1), filtered_df['Runs'].values, marker='o', color='#3498db', linewidth=2)
                ax.set_title(f"Runs per Innings: {p1} vs {p2}", fontweight='bold')
                ax.set_xlabel("Match/Innings Number")
                ax.set_ylabel("Runs Scored")
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig) 

with tab2:
    st.subheader("🤖 Chat with your Data")
    
    st.markdown("**🧠 Choose your AI Brain:**")
    model_choice = st.selectbox(
        "Select Model", 
        ["gemini-2.5-flash (Fast & Great for basic stats)", "gemini-2.5-pro (Smarter & Great for deep analysis)"],
        label_visibility="collapsed"
    )
    selected_model_name = model_choice.split(" ")[0]

    API_KEY = st.text_input("Enter Gemini API Key", type="password")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I've analyzed your cricket data. Ask me anything!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("E.g., Who has the highest strike rate?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        if not API_KEY:
            st.error("Please enter your API Key above.")
        else:
            with st.spinner(f"{selected_model_name} is thinking..."):
                try:
                    genai.configure(api_key=API_KEY)
                    model = genai.GenerativeModel(selected_model_name)
                    
                    context = filtered_df.head(50).to_csv(index=False) 
                    ai_prompt = f"Data context:\n{context}\n\nUser Question: {prompt}\nAnswer concisely and accurately based ONLY on the data provided."
                    
                    response = model.generate_content(ai_prompt)
                    ans = response.text
                    
                    st.chat_message("assistant").write(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"Error: {e}")
