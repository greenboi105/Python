import pandas as pd
import plotly.express as px

"""
You're importing the following elements from dash:

Dash helps you initialize your applications.

html, also called Dash HTML Components, lets you access HTML tags.

dcc, short for Dash Core Components, allows you to create interactive components like graphs, dropdowns, or date ranges.
"""
from dash import Dash, Input, Output, dcc, html

data = (
    pd.read_csv("/Users/greenboi/ComputerScience/Python/Python Frameworks and Projects/Frameworks/Avocado Analytics/avocado.csv")
    .assign(Date=lambda data: pd.to_datetime(data["Date"], format="%Y-%m-%d"))
    .sort_values(by="Date")
)

regions = data["region"].sort_values().unique()
avocado_types = data["type"].sort_values().unique()

"""
Here we set the styling and title qualities of the app.
"""

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Avocado Analytics: Understand Your Avocados!"

"""
Here the layout of the app is defined.

In this particular case we have an initial header section with a particular styling, followed by three core components that filter the options, followed by two displayed graphs.
"""

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="🥑", className="header-emoji"),
                html.H1(
                    children="Avocado Analytics", className="header-title"
                ),
                html.P(
                    children=(
                        "Analyze the behavior of avocado prices and the number of avocados sold in the US between 2015 and 2018"
                    ),
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Region", className="menu-title"),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[
                                {"label": region, "value": region}
                                for region in regions
                            ],
                            value="Albany",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Type", className="menu-title"),
                        dcc.Dropdown(
                            id="type-filter",
                            options=[
                                {
                                    "label": avocado_type.title(),
                                    "value": avocado_type,
                                }
                                for avocado_type in avocado_types
                            ],
                            value="organic",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range", className="menu-title"
                        ),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=data["Date"].min().date(),
                            max_date_allowed=data["Date"].max().date(),
                            start_date=data["Date"].min().date(),
                            end_date=data["Date"].max().date(),
                        ),
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="price-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="volume-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),
    ]
)

"""
Callback function returns two figures, one for the prices of avocados and another for the volume sold for a given geographic region.
"""
@app.callback(
    Output("price-chart", "figure"),
    Output("volume-chart", "figure"),
    Input("region-filter", "value"),
    Input("type-filter", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_charts(region, avocado_type, start_date, end_date):

    # Generate a DataFrame with the valid data from the given filters
    filtered_data = data.query("region == @region and type == @avocado_type and Date >= @start_date and Date <= @end_date")

    # Generate an updated price figure
    price_chart_figure = px.line(filtered_data, x='Date', y='AveragePrice', title="Average Price of Avocados")

    # Generate an updated volume figure using the new data
    volume_chart_figure = px.line(filtered_data, x="Date", y="Total Volume", title="Avocados Sold")

    # Return the updated figures for the given callback
    return price_chart_figure, volume_chart_figure

if __name__ == "__main__":
    app.run_server(debug=True)

    