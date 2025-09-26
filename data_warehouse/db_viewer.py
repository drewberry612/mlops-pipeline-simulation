import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

st.title("Database Viewer")

# Connect to the database
conn = sqlite3.connect("warehouse.db")

# Read both tables
df_images = pd.read_sql_query("SELECT * FROM images;", conn)
df_training = pd.read_sql_query("SELECT * FROM training_runs;", conn)

conn.close()

# Create two columns
col1, col2 = st.columns(2)

# Display each table in its own column

with col1:
    st.subheader("Images Table")
    st.dataframe(df_images, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Training Runs Table")
    st.dataframe(df_training, use_container_width=True, hide_index=True)
