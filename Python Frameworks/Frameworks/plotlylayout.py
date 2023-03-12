# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

#NOTE: This app does not have any dynamic user interactivity since there are no app callbacks

# Necessary imports
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

# The beginning of the app 
app = Dash(__name__)

# The DataFrame with the data entries we want to display using the dashboard
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

# Generate a barplot with the corresponding values
fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

API_KEY = 'pk.eyJ1IjoiZ3JlZW5ib2kxMDUiLCJhIjoiY2xlMGo4aGZ0MDlyNTN2cWtlYnAyeTduNCJ9.22dna4Wm_VmFU1OKBXCFgQ'
airbnb = pd.read_csv('https://raw.githubusercontent.com/greenboi105/Python-Frameworks-and-Projects/main/Data%20Analysis/Data/AB_NYC_2019.csv?token=GHSAT0AAAAAAB6Y3543GIUWRAEG6IN5TZU2Y7NNVXA')
sub_6=airbnb[airbnb.price < 500]
px.set_mapbox_access_token(API_KEY)
fig2 = px.scatter_mapbox(sub_6, lon='longitude', lat='latitude', color='price')


# The structure of the app is a tree
app.layout = html.Div(children=[
		
		# An initial message with the statement "Hello Dash"
    html.H1(children='Hello Dash'),
		
		# To be displayed below the initial H1 heading
    html.Div(children='''
        Dash: A web application framework for your data.
    '''),
		
		# Display the Graph Objects
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    dcc.Graph(
        id='example-graph2',
        figure=fig2
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)