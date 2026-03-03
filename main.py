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
    # Return dummy structure if no file exists to prevent hard crashes
    return pd.DataFrame(columns=["Match No", "Date", "Batsman", "Bowler", "Runs", "Balls", "Wickets"])

df = load_data()

# --- MAIN DASHBOARD AREA ---
tab1, tab2 = st.tabs(["📊 Data & Controls", "🤖 AI Analyst"])

with tab1:
    st.markdown("### 🔍 Search Matches")
    
    # 4. Clear button implementation
    st.button("🧹 Clear All Searches", on_click=clear_search)
    
    col1, col2, col3, col4 = st.columns(4)
    p1 = col1.text_input("Player 1:", key="p1_input")
    p2 = col2.text_input("Player 2 (VS):", key="p2_input")
    date_filter = col3.text_input("Date (Y-M-D):", key="date_input")
    
    # 6. Added Match No search
    match_filter = col4.text_input("Match No:", key="match_input")

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
        # Match No Filter
        if match_filter and 'Match No' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Match No'].astype(str).str.contains(match_filter, case=False, na=False)]

        # Date Filter
        if date_filter and 'Date' in filtered_df.columns:
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

        # Base formatting
        for col in ['Runs', 'Balls', 'Wickets']:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)

        # 1. PROGRESSIVE STATS LOGIC (Cumulative summing)
        # Ensure data is sorted by Match No or Date so cumulative stats make chronological sense
        sort_col = 'Match No' if 'Match No' in filtered_df.columns else 'Date'
        if sort_col in filtered_df.columns:
            filtered_df = filtered_df.sort_values(sort_col)

        filtered_df['Cum_Runs'] = filtered_df.groupby(['Batsman', 'Bowler'])['Runs'].cumsum()
        filtered_df['Cum_Balls'] = filtered_df.groupby(['Batsman', 'Bowler'])['Balls'].cumsum()
        filtered_df['Cum_Wickets'] = filtered_df.groupby(['Batsman', 'Bowler'])['Wickets'].cumsum()

        # ADVANCED PROGRESSIVE METRICS 
        filtered_df['BatSR'] = np.where(filtered_df['Cum_Balls'] > 0, (filtered_df['Cum_Runs'] / filtered_df['Cum_Balls'] * 100).round(1), 0)
        filtered_df['BatAvg'] = np.where(filtered_df['Cum_Wickets'] > 0, (filtered_df['Cum_Runs'] / filtered_df['Cum_Wickets']).round(1), np.nan)
        filtered_df['BwlSR'] = np.where(filtered_df['Cum_Wickets'] > 0, (filtered_df['Cum_Balls'] / filtered_df['Cum_Wickets']).round(1), np.nan)
        filtered_df['BwlAvg'] = np.where(filtered_df['Cum_Wickets'] > 0, (filtered_df['Cum_Runs'] / filtered_df['Cum_Wickets']).round(1), np.nan)
        filtered_df['Econ'] = np.where(filtered_df['Cum_Balls'] > 0, (filtered_df['Cum_Runs'] / (filtered_df['Cum_Balls'] / 6)).round(2), 0)

        # COMBINED VIEW LOGIC
        if is_combined:
            filtered_df['Inns'] = 1 
            # 5. If combined and ONE player searched, sort by Batsman role first
            if p1 and not p2:
                filtered_df['is_bat'] = filtered_df['Batsman'].str.contains(p1, case=False, na=False)
                filtered_df = filtered_df.sort_values(by=['is_bat'], ascending=False).drop(columns=['is_bat'])
            elif p2 and not p1:
                filtered_df['is_bat'] = filtered_df['Batsman'].str.contains(p2, case=False, na=False)
                filtered_df = filtered_df.sort_values(by=['is_bat'], ascending=False).drop(columns=['is_bat'])
                
            agg_dict = {'Inns': 'sum', 'Runs': 'sum', 'Balls': 'sum', 'Wickets': 'sum'}
            filtered_df = filtered_df.groupby(['Batsman', 'Bowler']).agg(agg_dict).reset_index()
        else:
            if 'Inns' not in filtered_df.columns:
                filtered_df.insert(4, 'Inns', 1)

    # --- DISPLAY TABLE ---
    st.markdown("### 📋 Match Data")
    
    if filtered_df.empty:
        st.warning("No data found. Check your filters or ensure 'cricket_summary_data.csv' is properly formatted.")
    else:
        # 7. Make Batsman & Bowler permanent and sticky
        fixed_cols = ['Batsman', 'Bowler']
        # Filter out columns we don't want the user to toggle off, or temporary calc columns
        hide_from_picker = fixed_cols + ['Cum_Runs', 'Cum_Balls', 'Cum_Wickets'] 
        available_columns = [col for col in filtered_df.columns if col not in hide_from_picker]
        
        with st.expander("👁️ Customize Table Columns"):
            selected_columns = st.multiselect(
                "Add or remove stats (Batsman and Bowler are permanent):", 
                options=available_columns, 
                default=available_columns
            )
        
        final_display_cols = fixed_cols + selected_columns

        # Display dataframe. Setting the index to Batsman and Bowler automatically "freezes" them on horizontal scroll!
        display_df = filtered_df[final_display_cols].set_index(fixed_cols)
        st.dataframe(display_df, use_container_width=True)

        # 3. PDF EXPORT LOGIC
        def generate_pdf(df_to_print):
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()
            pdf.set_font("Arial", size=8)
            
            # Reset index so Batsman and Bowler appear in the PDF
            df_to_print = df_to_print.reset_index()
            columns = list(df_to_print.columns)
            
            # Calculate dynamic column width based on A4 landscape width (~280mm usable)
            col_width = 280 / len(columns) if len(columns) > 0 else 20
            
            # Headers
            pdf.set_font("Arial", 'B', 8)
            for col in columns:
                pdf.cell(col_width, 8, txt=str(col), border=1, align='C')
            pdf.ln()
            
            # Rows
            pdf.set_font("Arial", size=8)
            for i in range(len(df_to_print)):
                for col in columns:
                    val = str(df_to_print[col].iloc[i])
                    pdf.cell(col_width, 8, txt=val, border=1, align='C')
                pdf.ln()
                
            return pdf.output(dest='S').encode('latin-1')

        pdf_bytes = generate_pdf(display_df)
        st.download_button(
            label="📄 Download Table as PDF",
            data=pdf_bytes,
            file_name="cricket_stats_report.pdf",
            mime="application/pdf"
        )

        st.divider()

        # 2. MATCHUP GRAPHS WITH VALIDATION
        st.markdown("### 📈 Deep Dive Analytics")
        if st.button("Generate Matchup Graphs"):
            if not p1 or not p2:
                st.error("⚠️ Validation Error: You must enter BOTH Player 1 and Player 2 to generate matchup graphs.")
            elif view_mode == "All":
                st.error("⚠️ Validation Error: You must select either 'Batting' or 'Bowling' in the Mode options (cannot be 'All').")
            elif is_combined:
                 st.error("⚠️ Please uncheck 'Combined View' to see progressive graphs over time.")
            else:
                st.success(f"Showing Progressive Stats for {p1} vs {p2} ({view_mode} Perspective)")
                
                chart_data = filtered_df.reset_index(drop=True)
                
                colA, colB = st.columns(2)
                with colA:
                    if view_mode == "Batting":
                        st.markdown("**📈 Progressive Batting Strike Rate**")
                        st.line_chart(chart_data['BatSR'])
                    else:
                        st.markdown("**📈 Progressive Bowling Strike Rate**")
                        st.line_chart(chart_data['BwlSR'])
                        
                with colB:
                    if view_mode == "Batting":
                        st.markdown("**📉 Progressive Batting Average**")
                        st.line_chart(chart_data['BatAvg'])
                    else:
                        st.markdown("**📉 Progressive Bowling Average**")
                        st.line_chart(chart_data['BwlAvg'])

# --- AI TAB REMAINS UNCHANGED ---
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
