"""
Basic Dash Callbacks

app.layout describes what the app looks like and is a hierarchical tree of components. 

The Dash HTML Components module provides classes for all of the HTML tags, and the keyword arguments describe the HTML attributes like style, className, and id. 

The Dash Core Components module generates higher-level components like controls and graphs.

Callback functions are functions that are automatically called by Dash whenever an input component's property changes, in order to update some property in another component.
"""

from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text')
    ]),
    html.Br(),
    html.Div(id='my-output'),

])

"""
The inputs and outputs of our application are described as the arguments of the decorator app.callback

In Dash, the inputs and outputs of our application are simply the properties of a particular component. 

Whenever an input property changes, the function that the callback decorator wraps will get called automatically.

Dash provides this callback function with the new value of the input property as its argument, and Dash updates the property of the output component with whatever was returned by the function.

The component_id and component_property keywords are optional. They are included in this example for clarity but will be omitted in the rest of the documentation for the sake of brevity and readability.

Whenever a cell changes (the input), all the cells that depend on that cell (the outputs) will get updated automatically.

This is called "Reactive Programming" because the outputs react to changes in the inputs automatically.

With Dash's interactivity, we can dynamically update any of the arguments set using callbacks.
"""

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    return f'Output: {input_value}'


if __name__ == '__main__':
    app.run_server(debug=True)