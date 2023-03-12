from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

"""
In this example, the "value" property of the dcc.Slider is the input of the app, and the output of the app is the "figure" property of the dcc.Graph.

Whenever the value of the dcc.Slider changes, Dash calls the callback function update_figure with the new value. 

The function filters the dataframe with this new value, constructs a figure object, and returns it to the Dash application.

We use the Pandas library to load the dataframe at the start of the app. This dataframe df is in the global state of the app and can be read inside the callback functions.

Loading data into memory can be expensive. By loading querying data at the start of the app instead of inside the callback functions, we ensure that this operation is only done once - when the app server starts.

When a user visits the app or interacts with the app, that data df is already in memory. If possible, expensive initialization should be done in the global scope of the app instead of within the callback functions.

The callback does not modify the original data, it only creates copies of the dataframe by filtering using pandas. 
"""

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        df['year'].min(),
        df['year'].max(),
        step=None,
        value=df['year'].min(),
        marks={str(year): str(year) for year in df['year'].unique()},
        id='year-slider'
    )
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):

    filtered_df = df[df.year == selected_year]

    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                     size="pop", color="continent", hover_name="country",
                     log_x=True, size_max=55)

    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

