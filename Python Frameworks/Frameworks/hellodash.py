"""
Dash apps are composed of two parts. 

The first part is the "layout", which describes what the app looks like. 

The second part describes the interactivity of the app.
"""

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")


# 1. The layout is composed of a tree of "components" such as html.Div and dcc.Graph.
app.layout = html.Div(children=[

    # 2. The Dash HTML Components module (dash.html) has a component for every HTML tag. 
    # 3. Not all components are pure HTML. The Dash Core Components module dash.dcc contains higher-level components that are interactive and are generated with JS, HTML, and CSS using the React.js library.
    html.H1(children='Hello Dash'),
    
    # 5. The children property is special. 
    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
