# streamlit_app.py

import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# Create a connection object.
conn = st.experimental_connection("gsheets", type=GSheetsConnection)

df = conn.read(worksheet="WARNING_TEST", usecols=[0,1], ttl=1)

# # Print results.
# for row in df.itertuples():
#     st.write(f"{row.name} has a :{row.pet}:")

st.dataframe(df)