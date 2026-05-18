import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from fpdf import FPDF

# --- PAGE SETUP ---
st.set_page_config(page_title="Cricket Analytics", page_icon="🏏", layout="wide")
st.title("🏏 Cricket Stats Dashboard")

# --- INITIALIZE SESSION STATE FOR CLEAR BUTTON ---
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

# --- SMART STATS CALCULATION FUNCTIONS ---
def calculate_smart_baselines(df_all):
    df_calc = df_all.copy()
    for col in ['Runs', 'Balls', 'Wickets']:
        if col in df_calc.columns:
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce').fillna(0)
            
    # Calculate player career metrics across full dataset
    bat_profile = df_calc.groupby('Batsman').agg({'Runs': 'sum', 'Balls': 'sum', 'Wickets': 'sum'}).reset_index()
    bat_profile['BatAvg'] = np.where(bat_profile['Wickets'] > 0, bat_profile['Runs'] / bat_profile['Wickets'], bat_profile['Runs'])
    bat_profile['BatSR'] = np.where(bat_profile['Balls'] > 0, (bat_profile['Runs'] / bat_profile['Balls']) * 100, 0)
    
    bwl_profile = df_calc.groupby('Bowler').agg({'Runs': 'sum', 'Balls': 'sum', 'Wickets': 'sum'}).reset_index()
    bwl_profile['BwlAvg'] = np.where(bwl_profile['Wickets'] > 0, bwl_profile['Runs'] / bwl_profile['Wickets'], bwl_profile['Runs'])
    bwl_profile['Econ'] = np.where(bwl_profile['Balls'] > 0, (bwl_profile['Runs'] / bwl_profile['Balls']) * 6, 0)
    
    # Global league averages
    l_bat_avg = bat_profile['BatAvg'].mean() if not bat_profile.empty else 1
    l_sr = bat_profile['BatSR'].mean() if not bat_profile.empty else 1
    l_bwl_avg = bwl_profile['BwlAvg'].mean() if not bwl_profile.empty else 1
    l_econ = bwl_profile['Econ'].mean() if not bwl_profile.empty else 1
    
    # Compute OQI profiles
    bat_profile['OQI'] = (bat_profile['BatAvg'] / l_bat_avg) * (bat_profile['BatSR'] / l_sr)
    bwl_profile['OQI'] = np.where((bwl_profile['BwlAvg'] > 0) & (bwl_profile['Econ'] > 0), 
                                   (l_bwl_avg / bwl_profile['BwlAvg']) * (l_econ / bwl_profile['Econ']), 1)
    
    return (
        dict(zip(bat_profile['Batsman'], bat_profile['OQI'])),
        dict(zip(bwl_profile['Bowler'], bwl_profile['OQI'])),
        dict(zip(bat_profile['Batsman'], bat_profile['BatAvg'])),
        dict(zip(bwl_profile['Bowler'], bwl_profile['Econ'])),
        l_econ
    )

def get_player_smart_rating(name_query, df_all, oqi_bat, oqi_bwl, bat_avg, bwl_econ, l_econ):
    m_batsman = [b for b in oqi_bat.keys() if name_query.lower() in b.lower()]
    m_bowler = [b for b in oqi_bwl.keys() if name_query.lower() in b.lower()]
    
    sbr, sbo = None, None
    
    if m_batsman:
        act_name = m_batsman[0]
        p_df = df_all[df_all['Batsman'] == act_name]
        total_balls = p_df['Balls'].sum()
        if total_balls > 0:
            weighted_runs = sum(row['Runs'] * oqi_bwl.get(row['Bowler'], 1.0) for _, row in p_df.iterrows())
            sbr = (weighted_runs / total_balls) * np.sqrt(bat_avg.get(act_name, 0))
            sbr = round(sbr, 2)
            
    if m_bowler:
        act_name = m_bowler[0]
        p_df = df_all[df_all['Bowler'] == act_name]
        inns = p_df['Match No'].nunique() if 'Match No' in p_df.columns else len(p_df)
        inns = max(inns, 1)
        weighted_wkts = sum(row['Wickets'] * oqi_bat.get(row['Batsman'], 1.0) for _, row in p_df.iterrows())
        p_econ = bwl_econ.get(act_name, 0)
        if p_econ > 0:
            sbo = (weighted_wkts / inns) * (l_econ / p_econ)
            sbo = round(sbo, 2)
            
    return sbr, sbo

# --- SMART RATINGS HEADER COMPONENT ---
show_smart = st.toggle("🧠 Enable Smart Analytics Mode (SBR & SBO Engine)", value=False)
if show_smart and not df.empty:
    oqi_bat, oqi_bwl, bat_avg, bwl_econ, l_econ = calculate_smart_baselines(df)
    st.markdown("### 📊 Live Smart Metrics")
    sc1, sc2 = st.columns(2)
    
    p1_curr = st.session_state.p1_input
    p2_curr = st.session_state.p2_input
    
    with sc1:
        if p1_curr:
            sbr1, sbo1 = get_player_smart_rating(p1_curr, df, oqi_bat, oqi_bwl, bat_avg, bwl_econ, l_econ)
            st.markdown(f"**{p1_curr}**")
            if sbr1 is not None: st.metric("Smart Batting Rating (SBR)", sbr1)
            if sbo1 is not None: st.metric("Smart Bowling Rating (SBO)", sbo1)
        else:
            st.caption("Enter Player 1 below to generate smart ratings.")
            
    with sc2:
        if p2_curr:
            sbr2, sbo2 = get_player_smart_rating(p2_curr, df, oqi_bat, oqi_bwl, bat_avg, bwl_econ, l_econ)
            st.markdown(f"**{p2_curr}**")
            if sbr2 is not None: st.metric("Smart Batting Rating (SBR)", sbr2)
            if sbo2 is not None: st.metric("Smart Bowling Rating (SBO)", sbo2)
        else:
            st.caption("Enter Player 2 below to generate smart ratings.")
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

        sort_col = 'Match No' if 'Match No' in filtered_df.columns else 'Date'
        if sort_col in filtered_df.columns:
            filtered_df = filtered_df.sort_values(sort_col)

        # PROGRESSIVE STATS CALCULATIONS
        filtered_df['Cum_Runs'] = filtered_df.groupby(['Batsman', 'Bowler'])['Runs'].cumsum()
        filtered_df['Cum_Balls'] = filtered_df.groupby(['Batsman', 'Bowler'])['Balls'].cumsum()
        filtered_df['Cum_Wickets'] = filtered_df.groupby(['Batsman', 'Bowler'])['Wickets'].cumsum()

        filtered_df['BatSR'] = np.where(filtered_df['Cum_Balls'] > 0, (filtered_df['Cum_Runs'] / filtered_df['Cum_Balls'] * 100).round(1), 0)
        filtered_df['BatAvg'] = np.where(filtered_df['Cum_Wickets'] > 0, (filtered_df['Cum_Runs'] / filtered_df['Cum_Wickets']).round(1), np.nan)
        filtered_df['BwlSR'] = np.where(filtered_df['Cum_Wickets'] > 0, (filtered_df['Cum_Balls'] / filtered_df['Cum_Wickets']).round(1), np.nan)
        filtered_df['BwlAvg'] = np.where(filtered_df['Cum_Wickets'] > 0, (filtered_df['Cum_Runs'] / filtered_df['Cum_Wickets']).round(1), np.nan)
        filtered_df['Econ'] = np.where(filtered_df['Cum_Balls'] > 0, (filtered_df['Cum_Runs'] / (filtered_df['Cum_Balls'] / 6)).round(2), 0)

        if is_combined:
            filtered_df['Inns'] = 1 
            # Strategic positional sorting if one player is target filtered
            if (p1 and not p2) or (p2 and not p1):
                target_p = p1 if p1 else p2
                filtered_df['is_bat'] = filtered_df['Batsman'].str.contains(target_p, case=False, na=False)
                filtered_df = filtered_df.sort_values(by=['is_bat'], ascending=False).drop(columns=['is_bat'])
                
            agg_dict = {'Inns': 'sum', 'Runs': 'sum', 'Balls': 'sum', 'Wickets': 'sum'}
            filtered_df = filtered_df.groupby(['Batsman', 'Bowler']).agg(agg_dict).reset_index()
        else:
            if 'Inns' not in filtered_df.columns:
                filtered_df.insert(4, 'Inns', 1)

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

        # PDF Report Builder Function
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

        # VALIDATED PROGRESSIVE ANALYSIS GRAPHS
        st.markdown("### 📈 Progressive Analytics Dashboard")
        if st.button("Generate Matchup Graphs"):
            if not p1 or not p2:
                st.error("⚠️ Error: You must enter both Player 1 and Player 2 names to generate metrics trends graphs.")
            elif view_mode == "All":
                st.error("⚠️ Error: You must isolate a single role view. Select either 'Batting' or 'Bowling' mode options.")
            elif is_combined:
                 st.error("⚠️ Trends analysis cannot be generated in 'Combined View'. Uncheck it to see match-by-match timelines.")
            else:
                st.success(f"Plotting Progressive Timeline: {p1} vs {p2} ({view_mode} Perspective)")
                chart_data = filtered_df.reset_index(drop=True)
                
                colA, colB = st.columns(2)
                with colA:
                    if view_mode == "Batting":
                        st.markdown("**📈 Progressive Batting Strike Rate Trend**")
                        st.line_chart(chart_data['BatSR'])
                    else:
                        st.markdown("**📈 Progressive Bowling Strike Rate Trend**")
                        st.line_chart(chart_data['BwlSR'])
                        
                with colB:
                    if view_mode == "Batting":
                        st.markdown("**📉 Progressive Batting Average Trend**")
                        st.line_chart(chart_data['BatAvg'])
                    else:
                        st.markdown("**📉 Progressive Bowling Average Trend**")
                        st.line_chart(chart_data['BwlAvg'])

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
