
import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
from PIL import Image

# st.set_page_config(page_title="Demo Dashboard", layout="wide")

# # ---- READ EXCEL ----
# # 'st.cache_resource'  --> is good for ML conns and DB conns, and 'st.cache_data' is good for dataframes, excel etc
# @st.cache_data
# def get_data_from_excel():
#     df = pd.read_excel(
#         # io="ADEVETUREWORKS_ETL.xlsx",
#         io="supermarkt_sales.xlsx",

#         engine="openpyxl",
#         sheet_name="Sheet1",
#         skiprows=0,
#         usecols=[0, 1, 2, 3, 4],
#         # usecols="A:E",
#         nrows=140,
#     )
#     # # Add 'hour' column to dataframe
#     # df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
#     # return df

# df = get_data_from_excel()

# # ---- SIDEBAR ----
# st.sidebar.header("Please Filter Data Here:")
# db = st.sidebar.multiselect(
#     "Select the Database:",
#     options=df["DatabaseName"].unique(),
#     default=df["DatabaseName"].unique()
# )

# tbl = st.sidebar.multiselect(
#     "Select the Table:",
#     options=df["table_name"].unique(),
#     default=df["table_name"].unique(),
# )

# st.dataframe(df)  # Same as st.write(df)

# df_selection = df.query(
#     "DatabaseName == @db & table_name == @tbl"
# )

st.set_page_config(page_title='Demo Results')
st.header('Demo Results 2021')
st.subheader('Was the tutorial helpful?')

### --- LOAD DATAFRAME
excel_file = 'ADEVETUREWORKS_ETL.xlsx'
sheet_name = 'SCHEMA_T'

df = pd.read_excel(excel_file,
                   sheet_name=sheet_name,
                   usecols='A:E',
                   header=1)

# df_participants = pd.read_excel(excel_file,
#                                 sheet_name= sheet_name,
#                                 usecols='F:G',
#                                 header=3)
# df_participants.dropna(inplace=True)

# --- STREAMLIT SELECTION
v_DatabaseName = df['DatabaseName'].unique().tolist()
# ages = df['Age'].unique().tolist()

# age_selection = st.slider('Age:',
#                         min_value= min(ages),
#                         max_value= max(ages),
#                         value=(min(ages),max(ages)))

department_selection = st.multiselect('DatabaseName:',
                                    v_DatabaseName,
                                    default=v_DatabaseName)

# --- FILTER DATAFRAME BASED ON SELECTION
# mask = (df['Age'].between(*age_selection)) & (df['Department'].isin(department_selection))
mask = (df['Department'].isin(department_selection))

number_of_result = df[mask].shape[0]
st.markdown(f'*Available Results: {number_of_result}*')

# # --- GROUP DATAFRAME AFTER SELECTION
# df_grouped = df[mask].groupby(by=['Rating']).count()[['Age']]
# df_grouped = df_grouped.rename(columns={'Age': 'Votes'})
# df_grouped = df_grouped.reset_index()

# # --- PLOT BAR CHART
# bar_chart = px.bar(df_grouped,
#                    x='Rating',
#                    y='Votes',
#                    text='Votes',
#                    color_discrete_sequence = ['#F63366']*len(df_grouped),
#                    template= 'plotly_white')
# st.plotly_chart(bar_chart)

# # --- DISPLAY IMAGE & DATAFRAME
# col1, col2 = st.beta_columns(2)
# image = Image.open('images/Demo.jpg')
# print(image)
# col1.image(image,
#         caption='Designed by slidesgo / Freepik',
#         use_column_width=True)
# col2.dataframe(df[mask])

# # --- PLOT PIE CHART
# pie_chart = px.pie(df_participants,
#                 title='Total No. of Participants',
#                 values='Participants',
#                 names='Departments')

# st.plotly_chart(pie_chart)