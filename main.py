import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from fpdf import FPDF
# --- PAGE SETUP ---
st.set_page_config(page_title="Cricket Analytics", page_icon="🏏", layout="wide")
st.title("🏏 Cricket Stats Dashboard")

# --- WELCOME MESSAGE ---
if 'welcome_shown' not in st.session_state:
    st.success("🏏 This website is purely for members of the Gully Cricket Gang, developed by gang leader Rahul Raj with pure love for this game. Enjoy!")
    st.balloons()
    st.session_state.welcome_shown = True

# --- INITIALIZE SESSION STATE ---
for key in ['p1_input', 'p2_input', 'date_input', 'match_input']:
    if key not in st.session_state:
        st.session_state[key] = ""

def clear_search():
    st.session_state.p1_input = ""
    st.session_state.p2_input = ""
    st.session_state.date_input = ""
    st.session_state.match_input = ""

# --- LOAD DATA ---
FILE = "cricket_summary_data.csv"
@st.cache_data 
def load_data():
    if os.path.exists(FILE):
        return pd.read_csv(FILE)
    return pd.DataFrame(columns=["Match No", "Date", "Batsman", "Bowler", "Runs", "Balls", "Wickets"])

df = load_data()

# --- SMART STATS CALCULATION ENGINE ---
@st.cache_data
def calculate_smart_rankings(df_all):
    df_calc = df_all.copy()
    for col in ['Runs', 'Balls', 'Wickets']:
        if col in df_calc.columns:
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce').fillna(0)
            
    # Base Profiles
    bat_profile = df_calc.groupby('Batsman').agg({'Runs': 'sum', 'Balls': 'sum', 'Wickets': 'sum'}).reset_index()
    bat_profile['BatAvg'] = np.where(bat_profile['Wickets'] > 0, bat_profile['Runs'] / bat_profile['Wickets'], bat_profile['Runs'])
    bat_profile['BatSR'] = np.where(bat_profile['Balls'] > 0, (bat_profile['Runs'] / bat_profile['Balls']) * 100, 0)
    
    bwl_profile = df_calc.groupby('Bowler').agg({'Runs': 'sum', 'Balls': 'sum', 'Wickets': 'sum'}).reset_index()
    bwl_profile['BwlAvg'] = np.where(bwl_profile['Wickets'] > 0, bwl_profile['Runs'] / bwl_profile['Wickets'], bwl_profile['Runs'])
    bwl_profile['Econ'] = np.where(bwl_profile['Balls'] > 0, (bwl_profile['Runs'] / bwl_profile['Balls']) * 6, 0)
    
    # League Averages
    l_bat_avg = bat_profile['BatAvg'].mean() if not bat_profile.empty else 1
    l_sr = bat_profile['BatSR'].mean() if not bat_profile.empty else 1
    l_bwl_avg = bwl_profile['BwlAvg'].mean() if not bwl_profile.empty else 1
    l_econ = bwl_profile['Econ'].mean() if not bwl_profile.empty else 1
    
    # Compute OQI
    bat_profile['OQI'] = (bat_profile['BatAvg'] / l_bat_avg) * (bat_profile['BatSR'] / l_sr)
    bwl_profile['OQI'] = np.where((bwl_profile['BwlAvg'] > 0) & (bwl_profile['Econ'] > 0), 
                                   (l_bwl_avg / bwl_profile['BwlAvg']) * (l_econ / bwl_profile['Econ']), 1)
    
    oqi_bat = dict(zip(bat_profile['Batsman'], bat_profile['OQI']))
    oqi_bwl = dict(zip(bwl_profile['Bowler'], bwl_profile['OQI']))
    
    # Weighted Events
    df_calc['Batsman_OQI'] = df_calc['Batsman'].map(oqi_bat).fillna(1.0)
    df_calc['Bowler_OQI'] = df_calc['Bowler'].map(oqi_bwl).fillna(1.0)
    df_calc['Weighted_Runs'] = df_calc['Runs'] * df_calc['Bowler_OQI']
    df_calc['Weighted_Wickets'] = df_calc['Wickets'] * df_calc['Batsman_OQI']
    
    # SBR Rankings
    sbr_df = df_calc.groupby('Batsman').agg(Total_Balls=('Balls', 'sum'), Weighted_Runs=('Weighted_Runs', 'sum')).reset_index()
    sbr_df = sbr_df.merge(bat_profile[['Batsman', 'BatAvg']], on='Batsman')
    sbr_df['SBR'] = np.where(sbr_df['Total_Balls'] > 0, (sbr_df['Weighted_Runs'] / sbr_df['Total_Balls']) * np.sqrt(sbr_df['BatAvg']), 0)
    sbr_df = sbr_df.round({'SBR': 2, 'Weighted_Runs': 2, 'BatAvg': 2})
    sbr_df = sbr_df[['Batsman', 'SBR', 'Weighted_Runs', 'BatAvg']].sort_values(by='SBR', ascending=False).reset_index(drop=True)
    sbr_df.index += 1 

    # SBO Rankings (Fixed Innings Calculation)
    if 'Match No' in df_calc.columns and df_calc['Match No'].notna().any():
        inns_df = df_calc[df_calc['Balls'] > 0].groupby('Bowler')['Match No'].nunique().reset_index()
        inns_df.rename(columns={'Match No': 'Innings'}, inplace=True)
    elif 'Date' in df_calc.columns and df_calc['Date'].notna().any():
        inns_df = df_calc[df_calc['Balls'] > 0].groupby('Bowler')['Date'].nunique().reset_index()
        inns_df.rename(columns={'Date': 'Innings'}, inplace=True)
    else:
        inns_df = df_calc[df_calc['Balls'] > 0].groupby('Bowler').size().reset_index(name='Innings')
        
    sbo_df = df_calc.groupby('Bowler').agg(Weighted_Wickets=('Weighted_Wickets', 'sum')).reset_index()
    sbo_df = sbo_df.merge(inns_df, on='Bowler', how='left').fillna(1)
    sbo_df = sbo_df.merge(bwl_profile[['Bowler', 'Econ']], on='Bowler')
    
    sbo_df['SBO'] = np.where((sbo_df['Innings'] > 0) & (sbo_df['Econ'] > 0),
                             (sbo_df['Weighted_Wickets'] / sbo_df['Innings']) * (l_econ / sbo_df['Econ']), 0)
    
    sbo_df = sbo_df.round({'SBO': 2, 'Weighted_Wickets': 2, 'Econ': 2})
    # Added Innings and Econ to output for absolute transparency
    sbo_df = sbo_df[['Bowler', 'SBO', 'Weighted_Wickets', 'Innings', 'Econ']].sort_values(by='SBO', ascending=False).reset_index(drop=True)
    sbo_df.index += 1

    # MVP Rankings
    mvp_df = pd.merge(sbr_df.rename(columns={'Batsman': 'Player'}), sbo_df.rename(columns={'Bowler': 'Player'}), on='Player', how='outer').fillna(0)
    mvp_df['Total_Smart_Points'] = (mvp_df['SBR'] + mvp_df['SBO']).round(2)
    mvp_df = mvp_df[['Player', 'Total_Smart_Points', 'SBR', 'Weighted_Runs', 'SBO', 'Weighted_Wickets']]
    mvp_df = mvp_df.sort_values(by='Total_Smart_Points', ascending=False).reset_index(drop=True)
    mvp_df.index += 1
    
    return sbr_df, sbo_df, mvp_df

# --- SMART RATINGS HEADER ---
show_smart = st.toggle("🧠 Enable Smart Stats Leaderboards", value=False)
if show_smart and not df.empty:
    st.markdown("### 🏆 Smart Stats Rankings")
    sbr_board, sbo_board, mvp_board = calculate_smart_rankings(df)
    stat_mode = st.radio("Select Leaderboard:", ["Most Valuable Player (MVP)", "Smart Batting Stats (SBR)", "Smart Bowling Stats (SBO)"], horizontal=True)
    
    if stat_mode == "Smart Batting Stats (SBR)":
        st.dataframe(sbr_board, use_container_width=True)
    elif stat_mode == "Smart Bowling Stats (SBO)":
        st.dataframe(sbo_board, use_container_width=True)
    else:
        st.dataframe(mvp_board, use_container_width=True)
    st.divider()

# --- MAIN DASHBOARD AREA ---
tab1, tab2 = st.tabs(["📊 Data & Controls", "🤖 AI Analyst"])

with tab1:
    st.markdown("### 🔍 Search Matches")
    st.button("🧹 Clear All Searches", on_click=clear_search)
    
    col1, col2, col3, col4 = st.columns(4)
    p1 = col1.text_input("Player 1:", key="p1_input")
    p2 = col2.text_input("Player 2 (VS):", key="p2_input")
    date_filter = col3.text_input("Date (Y-M-D):", key="date_input")
    match_filter = col4.text_input("Match No:", key="match_input")

    st.markdown("### ⚙️ View Options")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    
    view_mode = ctrl1.radio("Mode:", ["All", "Batting", "Bowling"], horizontal=True)
    is_combined = ctrl2.checkbox("Combined View")
    is_recent = ctrl3.checkbox("Recent Form")
    
    recent_n = 3
    if is_recent:
        recent_n = ctrl3.number_input("How many recent matches?", min_value=1, max_value=100, value=3, step=1)

    filtered_df = df.copy()

    if not filtered_df.empty:
        if match_filter and 'Match No' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Match No'].astype(str).str.contains(match_filter, case=False, na=False)]
        if date_filter and 'Date' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Date'].astype(str).str.contains(date_filter, case=False, na=False)]

        if p1:
            m1 = filtered_df['Batsman'].str.contains(p1, case=False, na=False)
            v1 = filtered_df['Bowler'].str.contains(p1, case=False, na=False)
            if view_mode == "Batting": filtered_df = filtered_df[m1]
            elif view_mode == "Bowling": filtered_df = filtered_df[v1]
            else: filtered_df = filtered_df[m1 | v1]
        
        if p2:
            m2 = filtered_df['Batsman'].str.contains(p2, case=False, na=False)
            v2 = filtered_df['Bowler'].str.contains(p2, case=False, na=False)
            if view_mode == "Batting": filtered_df = filtered_df[v2] 
            elif view_mode == "Bowling": filtered_df = filtered_df[m2] 
            else: filtered_df = filtered_df[m2 | v2]

        if is_recent:
            filtered_df = filtered_df.groupby(['Batsman', 'Bowler']).tail(recent_n)

        for col in ['Runs', 'Balls', 'Wickets']:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)

        # FIXED LOGIC: Execute calculations based on Combined vs Match-by-Match
        if is_combined:
            if (p1 and not p2) or (p2 and not p1):
                target_p = p1 if p1 else p2
                filtered_df['is_bat'] = filtered_df['Batsman'].str.contains(target_p, case=False, na=False)
                filtered_df = filtered_df.sort_values(by=['is_bat'], ascending=False).drop(columns=['is_bat'])
                
            agg_dict = {'Runs': 'sum', 'Balls': 'sum', 'Wickets': 'sum'}
            filtered_df = filtered_df.groupby(['Batsman', 'Bowler']).agg(agg_dict).reset_index()
            
            # Recalculate absolute stats on the grouped totals
            filtered_df['BatSR'] = np.where(filtered_df['Balls'] > 0, (filtered_df['Runs'] / filtered_df['Balls'] * 100).round(1), 0)
            filtered_df['BatAvg'] = np.where(filtered_df['Wickets'] > 0, (filtered_df['Runs'] / filtered_df['Wickets']).round(1), np.nan)
            filtered_df['BwlSR'] = np.where(filtered_df['Wickets'] > 0, (filtered_df['Balls'] / filtered_df['Wickets']).round(1), np.nan)
            filtered_df['BwlAvg'] = np.where(filtered_df['Wickets'] > 0, (filtered_df['Runs'] / filtered_df['Wickets']).round(1), np.nan)
            filtered_df['Econ'] = np.where(filtered_df['Balls'] > 0, (filtered_df['Runs'] / (filtered_df['Balls'] / 6)).round(2), 0)

        else:
            # PROGRESSIVE STATS LOGIC
            sort_col = 'Match No' if 'Match No' in filtered_df.columns else 'Date'
            if sort_col in filtered_df.columns:
                filtered_df = filtered_df.sort_values(sort_col)
                
            filtered_df['Cum_Runs'] = filtered_df.groupby(['Batsman', 'Bowler'])['Runs'].cumsum()
            filtered_df['Cum_Balls'] = filtered_df.groupby(['Batsman', 'Bowler'])['Balls'].cumsum()
            filtered_df['Cum_Wickets'] = filtered_df.groupby(['Batsman', 'Bowler'])['Wickets'].cumsum()

            # Calculate progressive metrics
            filtered_df['BatSR'] = np.where(filtered_df['Cum_Balls'] > 0, (filtered_df['Cum_Runs'] / filtered_df['Cum_Balls'] * 100).round(1), 0)
            filtered_df['BatAvg'] = np.where(filtered_df['Cum_Wickets'] > 0, (filtered_df['Cum_Runs'] / filtered_df['Cum_Wickets']).round(1), np.nan)
            filtered_df['BwlSR'] = np.where(filtered_df['Cum_Wickets'] > 0, (filtered_df['Cum_Balls'] / filtered_df['Cum_Wickets']).round(1), np.nan)
            filtered_df['BwlAvg'] = np.where(filtered_df['Cum_Wickets'] > 0, (filtered_df['Cum_Runs'] / filtered_df['Cum_Wickets']).round(1), np.nan)
            filtered_df['Econ'] = np.where(filtered_df['Cum_Balls'] > 0, (filtered_df['Cum_Runs'] / (filtered_df['Cum_Balls'] / 6)).round(2), 0)

    st.markdown("### 📋 Match Data")
    
    if filtered_df.empty:
        st.warning("No data found. Check your filters.")
    else:
        fixed_cols = ['Batsman', 'Bowler']
        hide_from_picker = fixed_cols + ['Cum_Runs', 'Cum_Balls', 'Cum_Wickets'] 
        available_columns = [col for col in filtered_df.columns if col not in hide_from_picker]
        
        with st.expander("👁️ Customize Table Columns"):
            selected_columns = st.multiselect(
                "Add or remove stats (Batsman and Bowler are permanent):", 
                options=available_columns, 
                default=available_columns
            )
        
        final_display_cols = fixed_cols + selected_columns
        display_df = filtered_df[final_display_cols].set_index(fixed_cols)
        st.dataframe(display_df, use_container_width=True)

        def generate_pdf(df_to_print):
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()
            pdf.set_font("Arial", size=8)
            df_to_print = df_to_print.reset_index()
            columns = list(df_to_print.columns)
            col_width = 280 / len(columns) if len(columns) > 0 else 20
            
            pdf.set_font("Arial", 'B', 8)
            for col in columns:
                pdf.cell(col_width, 8, txt=str(col), border=1, align='C')
            pdf.ln()
            
            pdf.set_font("Arial", size=8)
            for i in range(len(df_to_print)):
                for col in columns:
                    val = str(df_to_print[col].iloc[i])
                    pdf.cell(col_width, 8, txt=val, border=1, align='C')
                pdf.ln()
            return pdf.output(dest='S').encode('latin-1')

        pdf_bytes = generate_pdf(display_df)
        st.download_button(
            label="📄 Download Clean PDF Report",
            data=pdf_bytes,
            file_name="cricket_stats_report.pdf",
            mime="application/pdf"
        )

        st.divider()

        st.markdown("### 📈 Progressive Analytics Dashboard")
        if st.button("Generate Matchup Graphs"):
            if not p1 or not p2:
                st.error("⚠️ Error: You must enter both Player 1 and Player 2 names.")
            elif view_mode == "All":
                st.error("⚠️ Error: Select either 'Batting' or 'Bowling' mode options.")
            elif is_combined:
                 st.error("⚠️ Trends analysis cannot be generated in 'Combined View'.")
            else:
                st.success(f"Plotting Progressive Timeline: {p1} vs {p2} ({view_mode} Perspective)")
                chart_data = filtered_df.reset_index(drop=True)
                
                colA, colB = st.columns(2)
                
                with colA:
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    if view_mode == "Batting":
                        ax1.plot(chart_data.index + 1, chart_data['BatSR'], marker='o', color='#e74c3c', linewidth=2)
                        ax1.set_title(f"Progressive Batting Strike Rate", fontweight='bold')
                        ax1.set_ylabel("Strike Rate")
                    else:
                        ax1.plot(chart_data.index + 1, chart_data['BwlSR'], marker='o', color='#e74c3c', linewidth=2)
                        ax1.set_title(f"Progressive Bowling Strike Rate", fontweight='bold')
                        ax1.set_ylabel("Strike Rate")
                    
                    ax1.set_xlabel("Innings Timeline")
                    ax1.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig1)
                        
                with colB:
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    if view_mode == "Batting":
                        ax2.plot(chart_data.index + 1, chart_data['BatAvg'], marker='s', color='#2980b9', linewidth=2)
                        ax2.set_title(f"Progressive Batting Average", fontweight='bold')
                        ax2.set_ylabel("Average")
                    else:
                        ax2.plot(chart_data.index + 1, chart_data['BwlAvg'], marker='s', color='#2980b9', linewidth=2)
                        ax2.set_title(f"Progressive Bowling Average", fontweight='bold')
                        ax2.set_ylabel("Average")
                        
                    ax2.set_xlabel("Innings Timeline")
                    ax2.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig2)

# --- AI TAB COMPONENTS ---
with tab2:
    st.subheader("🤖 Chat with your Data")
    model_choice = st.selectbox("Select Model", ["gemini-2.5-flash (Fast & Great for basic stats)", "gemini-2.5-pro (Smarter & Great for deep analysis)"], label_visibility="collapsed")
    selected_model_name = model_choice.split(" ")[0]

    API_KEY = st.text_input("Enter Gemini API Key", type="password")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I've analyzed your cricket data. Ask me anything!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("E.g., Who has the highest progressive strike rate?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        if not API_KEY:
            st.error("Please enter your API Key above.")
        else:
            with st.spinner(f"{selected_model_name} is thinking..."):
                try:
                    genai.configure(api_key=API_KEY)
                    model = genai.GenerativeModel(selected_model_name)
                    context = filtered_df.to_csv(index=False) 
                    ai_prompt = f"Data context:\n{context}\n\nUser Question: {prompt}\nAnswer concisely and accurately based ONLY on the data provided."
                    response = model.generate_content(ai_prompt)
                    ans = response.text
                    st.chat_message("assistant").write(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"Error: {e}")
