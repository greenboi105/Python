"""
Here we have an example where a dcc.Slider updates a dcc.Graph.
"""

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

import pandas as pd

# We use the Pandas library to load the dataframe at the start of the app.
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

app = Dash(__name__)
app.title = 'GDP Per Capita Across Time'

"""
Here, the "value" property of the dcc.Slider is the input of the app, and the output of the app is the "figure" property of the dcc.Graph.

Whenever the value of the dcc.Slider changes, Dash calls the callback function update_figure with the new value.

The function filters the dataframe with this new value, constructs a figure object, and returns it to the Dash application.

This application contains a graph which displays a figure for the GDP per captia for a given selected year, along with a slider to modify the selected year.

The Slider takes parameters for the earliest year in the DataFrame, the latest year in the DataFrame, the input of the app, the marks for the unique years and a component id.
"""

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

"""
The callback does not modify the original data, it only creates copies of the dataframe by filtering using pandas.

Callbacks should never modify variables outside of their scope. If the callbacks modify global state, then one user's session might affect the next user's session and when the app is deployed on multiple processes or threads, those modifications will not be shared.
"""

@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):

    # Generate a filtered DataFrame using the given year
    filtered_df = df[df.year == selected_year]

    # Generate a figure from the associated DataFrame
    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp", size="pop", color="continent", hover_name="country", log_x=True, size_max=55)

    # Update the figure using transition
    fig.update_layout(transition_duration=500)
    
    # Return the updated figure
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
