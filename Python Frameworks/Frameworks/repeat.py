from dash import Dash, dcc, html, Input, Output 

app = Dash(__name__)

app.layout = html.Div([
    html.H6("Enter text to display the effects of callbacks."),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text')
    ]),
    html.Br(),
    html.Div(id='my-output'),
])

"""
The inputs and outputs of the application are described as arguments of the @app.callback decorator.

In Dash, the inputs and outputs of our application are simply the properties of a particular component. 

In this example, our input is the "value" property of the component that has the ID "my-input". 

Our output is the "children" property of the component with the ID "my-output".

Whenever an input property changes, the function that the callback decorator wraps will get called automatically. 

Dash provides this callback function with the new value of the input property as its argument, and Dash updates the property of the output component with whatever was returned by the function.
"""

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    return f'Output: {input_value}'

if __name__ == '__main__':
    app.run_server(debug=True)
