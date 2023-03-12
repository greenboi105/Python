from dash import dcc, html, Dash
from dash.dependencies import Input, Output
import requests

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Iframe(src = 'https://www.flightradar24.com')
    ]),

    html.Div([
    html.Pre(
        id='counter_text',
        children='Active flights worldwide:'
    ),
    dcc.Interval(
        id='interval-component',
        interval=6000, # 6000 milliseconds = 6 seconds
        n_intervals=0
    )])
])

@app.callback(Output('counter_text', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_layout(n):
    url = "https://data-live.flightradar24.com/zones/fcgi/feed.js?faa=1\
           &mlat=1&flarm=1&adsb=1&gnd=1&air=1&vehicles=1&estimated=1&stats=1"
    # A fake header is necessary to access the site:
    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = res.json()
    counter = 0
    for element in data["stats"]["total"]:
        counter += data["stats"]["total"][element]
    return 'Active flights worldwide: {}'.format(counter)

if __name__ == '__main__':
    app.run_server(debug=True)
