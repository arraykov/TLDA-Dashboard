import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import sqlite3
import dash_ag_grid as dag
import numpy as np
import statsmodels.api as sm
from dash import dcc, html, callback, Output, Input, State
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from datetime import    datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Assuming the path to the SQLite database file
db_file_path = 'Database.db'  # Replace with the actual path to your SQLite database

# Connect to the SQLite database
conn = sqlite3.connect(db_file_path)

# Load the tickers from the funds.csv file
funds_df = pd.read_csv('Mapping.csv')  # Replace with the actual path to your funds.csv file

# Creating an empty DataFrame to store the results
all_data = pd.DataFrame()

for index, row in funds_df.iterrows():
    ticker, nav_symbol = row['fund_ticker'], row['nav_symbol']
    try:
        # Queries to select data from the price, NAV, and fundamentals tables
        query_price = f"SELECT * FROM {ticker} ORDER BY date DESC LIMIT 2"
        query_nav = f"SELECT * FROM {nav_symbol} ORDER BY date DESC LIMIT 2"
        query_fundamentals = f"SELECT * FROM {ticker}_fundamentals"
        
        # Execute the queries and load the data into DataFrames
        price_df = pd.read_sql(query_price, conn)
        nav_df = pd.read_sql(query_nav, conn)
        fundamentals_df = pd.read_sql(query_fundamentals, conn)

        # Convert 'Close' columns in price_df and nav_df to float
        price_df['Close'] = pd.to_numeric(price_df['Close'], errors='coerce')
        nav_df['Close'] = pd.to_numeric(nav_df['Close'], errors='coerce')

        if price_df.empty or nav_df.empty or fundamentals_df.empty or len(nav_df) < 2:
            print(f"Not enough data for ticker: {ticker}")
            continue

        # Extracting and calculating necessary values
        nav = nav_df.iloc[0]['Close']
        nav_t = nav_df.iloc[1]['Close']

        # Safe check for nav_t to avoid division by zero or null
        if pd.notnull(nav) and pd.notnull(nav_t) and nav_t != 0:
            nav_delta = round(nav - nav_t, 2)
            nav_percent_delta = round(nav_delta / nav_t * 100, 2)
        else:
            nav_delta = None  # or some default value
            nav_percent_delta = None  # or some default value

        price = price_df.iloc[0]['Close']
        price_delta = round(price - price_df.iloc[1]['Close'], 2)

        # Extracting other values from the fundamentals table
        type_ = fundamentals_df.iloc[0]['Category']
        curr_disc = fundamentals_df.iloc[0]['Current Premium/Discount']
        three_month_z = fundamentals_df.iloc[0]['3 Month Z-Score']
        six_month_z = fundamentals_df.iloc[0]['6 Month Z-Score']
        one_year_z = fundamentals_df.iloc[0]['1 Year Z-Score']
        nav_avg = fundamentals_df.iloc[0]['52 Wk Avg Premium/Discount']
        nav_low = fundamentals_df.iloc[0]['52 Wk Low Premium/Discount']
        nav_high = fundamentals_df.iloc[0]['52 Wk High Premium/Discount']
        curr_yield = fundamentals_df.iloc[0]['Distribution Rate']
        amount = fundamentals_df.iloc[0]['Distribution Amount']
        frequency = fundamentals_df.iloc[0]['Distribution Frequency']
        fiscal_year = fundamentals_df.iloc[0]['Fiscal Year End']

        # Creating a DataFrame for the selected ticker
        data = pd.DataFrame({
            'Ticker': ticker,
            'Nav Ticker': nav_symbol,
            'Category': type_,
            '3m Z': three_month_z,
            '6m Z': six_month_z,
            '1y Z': one_year_z,
            'Price': price,
            'PriceΔ': price_delta,
            'NAV': nav,
            'NAV%Δ': round(nav_delta / nav_t * 100, 2),
            'NAVΔ': nav_delta,
            'Premium/Discount': curr_disc,
            '52W NAV Avg': nav_avg,
            '52W NAV Low': nav_low,
            '52W NAV High': nav_high,
            'Current Yield': curr_yield, 
            'Distribution Amount': amount,
            'Distribution Frequency': frequency,
            'Fiscal Year End': fiscal_year,
        }, index=[0])

        all_data = pd.concat([all_data, data], ignore_index=True)
    except Exception as e:
        print(f"An error occurred while processing the ticker {ticker}: {e}")
        continue

# Close the database connection
conn.close()

all_data['Current Yield'] = all_data['Current Yield'].astype(str).str.replace('%', '')
all_data['Current Yield'] = pd.to_numeric(all_data['Current Yield'], errors='coerce')

columns_to_convert = ["3m Z", "6m Z", "1y Z", "PriceΔ", "NAV%Δ", "NAVΔ", "Premium/Discount", "Current Yield"]

for column in columns_to_convert:
    all_data[column] = pd.to_numeric(all_data[column], errors='coerce')

dollar_columns = ["Price", "NAV"]
for column in dollar_columns:
    all_data[column] = all_data[column].apply(lambda x: f'${x:.2f}' if pd.notnull(x) else x)

styleConditionsForZScores = {
    "styleConditions": [
        {"condition": "params.value < 0", "style": {"color": "#FC766A"}},
        {"condition": "params.value > 0", "style": {"color": "#CCF381"}}
    ]
}

filterParams = {
    "buttons": ['apply', 'reset'],
    "closeOnApply": True,
}

columnDefs = [
            {"field": "Ticker", "cellRenderer": "StockLink", "filter": "agTextColumnFilter", "filterParams": filterParams, "floatingFilter": True},        
            {
                "headerName": "Z-Scores",
                "children": [
                    {"field": "3m Z", "cellStyle": styleConditionsForZScores, "filter": "agNumberColumnFilter", "filterParams": filterParams},
                    {"field": "6m Z", "cellStyle": styleConditionsForZScores, "filter": "agNumberColumnFilter", "filterParams": filterParams},
                    {"field": "1y Z", "cellStyle": styleConditionsForZScores, "filter": "agNumberColumnFilter", "filterParams": filterParams},
                ]
            },

            {"field": "Premium/Discount", "filter": "agNumberColumnFilter", "filterParams": filterParams},
            {"field": "Price", "filter": "agNumberColumnFilter", "filterParams": filterParams},
            {"field": "PriceΔ", "cellStyle": styleConditionsForZScores, "filter": "agNumberColumnFilter", "filterParams": filterParams},
            {
                "headerName": "NAV",
                "children": [
                    {"field": "NAV", "columnGroupShow": "open", "filter": "agNumberColumnFilter", "filterParams": filterParams},
                    {"field": "NAV%Δ", "cellStyle": styleConditionsForZScores, "filter": "agNumberColumnFilter", "filterParams": filterParams},
                    {"field": "NAVΔ", "cellStyle": styleConditionsForZScores, "filter": "agNumberColumnFilter", "filterParams": filterParams},
                    {"field": "52W NAV Avg", "columnGroupShow": "open", "filter": "agNumberColumnFilter", "filterParams": filterParams},
                    {"field": "52W NAV Low", "columnGroupShow": "open", "filter": "agNumberColumnFilter", "filterParams": filterParams},
                    {"field": "52W NAV High", "columnGroupShow": "open", "filter": "agNumberColumnFilter", "filterParams": filterParams},
                ]
            },
            {
                "headerName": "Fundamentals",
                "children": [
                    {"field": "Category", "suppressSizeToFit": True, "width": 250, "filter": "agTextColumnFilter", "filterParams": filterParams, "floatingFilter": True},
                    {"field": "Current Yield", "filter": "agNumberColumnFilter", "filterParams": filterParams},
                    {"field": "Distribution Amount", "columnGroupShow": "open","filter": "agNumberColumnFilter", "filterParams": filterParams},
                    {"field": "Distribution Frequency", "columnGroupShow": "open", "filter": "agTextColumnFilter", "filterParams": filterParams},
                    {"field": "Fiscal Year End", "columnGroupShow": "open", "filter": "agTextColumnFilter", "filterParams": filterParams},
                ]
            },
]

defaultColDef = {
    "resizable": True,
    "initialWidth": 200,
    "wrapHeaderText": True,
    "autoHeaderHeight": True,
    "sortable": True,
    "filter": "agMultiColumnFilter",
    "filterParams": filterParams,
    "enableRowGroup": False,
    "enableValue": False,
    "enablePivot": False,
}

external_stylesheets = ['custom.css', dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

def generate_dashboard(height='190px'):
    table = dag.AgGrid(
        id='table',
        columnDefs=columnDefs,
        rowData=all_data.to_dict('records'),
        style={"height": height, "width": "100%"},
        columnSize="responsiveSizeToFit",
        defaultColDef=defaultColDef,
        dashGridOptions={
            "autopaginationAutofPageSize": True,
            "animateRows": True, 
            "rowSelection": "single",
            "enableCellTextSelection": True,
            "ensureDomOrder":True,
        },
    )
    return table

def create_styled_empty_plot(title, yaxis_title):
    # Create an empty figure with styling
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='#041619',  # Background color for the plot area
        paper_bgcolor='#041619',  # Background color for the whole figure
        font=dict(color='#FBEAEB'),  # Text color for all text in the figure
        xaxis={'visible': False},  # Hide x-axis
        yaxis={'visible': False, 'title': yaxis_title},  # Hide y-axis but set title
        title=title,  # Set plot title
        margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
    )
    return fig

app.layout = dbc.Container([
    # Toggle control for showing/hiding plots with labels
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("Compact View", style={'float': 'left', 'margin-top': '5px','color': '#FBEAEB'}),
                daq.ToggleSwitch(
                    id='toggle-plots',
                    value=False,
                    style={'display': 'inline-block', 'margin': '0 10px'}
                ),
                html.Span("Arbitrage View", style={'float': 'right', 'margin-top': '5px','color': '#FBEAEB'})
            ], style={'textAlign': 'center', 'padding': '10px'})
        ], width={"size": 6, "offset": 3}),
    ]),

    # First row with AG Grid table
    dbc.Row([
        dbc.Col([
            html.Div(id='dashboard-table-container', children=[generate_dashboard()])
        ]),
    ]),

    # Second row with scatter plot, correlation table, and line charts
    dbc.Row([
        # Column for scatter plot and correlation table
        dbc.Col([
            # Scatter plot
            dcc.Graph(id='scatter-plot', figure=create_styled_empty_plot('', 'ETF Daily Returns')),
            
            # Div wrapping the Correlation table
            html.Div(
                id='correlation-table-container',  # New ID for the Div
                children=[
                    dash_table.DataTable(
                        id='correlation-table',

                columns=[
                    {"name": "", "id": "etf"}, 
                    {"name": "Correlation", "id": "correlation"}, 
                    {"name": "Beta", "id": "beta"}, 
                    {"name": "R Squared", "id": "r_squared"}
                ],
                style_table={'height': '350px', 'overflowY': 'auto'},
                style_data={
                    'border': '1px solid rgba(251, 234, 235, 0.5)'
                },
                sort_action="native",
                style_cell={
                    'textAlign': 'center', 
                    'backgroundColor': '#041619', 
                    'color': '#FBEAEB',
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'etf'}, 'width': '10%'},
                    {'if': {'column_id': 'correlation'}, 'width': '30%'},
                    {'if': {'column_id': 'beta'}, 'width': '30%'},
                    {'if': {'column_id': 'r_squared'}, 'width': '30%'}
                ],
                style_header={
                    'backgroundColor': '#041619', 
                    'color': '#FBEAEB', 
                    'fontWeight': 'bold',
                },
                row_selectable='single',  
                )
            ]
        )
    ], width=6),

        # Column for line charts
        dbc.Col([
            dcc.Graph(id='line-chart', figure=create_styled_empty_plot('', '')),
            dcc.Graph(id='premium-discount-chart', figure=create_styled_empty_plot('Premium/Discount Chart', 'Premium/Discount (%)')),
        ], width=6),
    ]),

    html.Div(id="selection-output")
], fluid=True)

@app.callback(
    [Output('scatter-plot', 'style'), 
     Output('premium-discount-chart', 'style'),
     Output('correlation-table-container', 'style'), 
     Output('line-chart', 'style')],
    [Input('toggle-plots', 'value')]
)

def toggle_plot_visibility(is_visible):
    if is_visible:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

@app.callback(
    Output('dashboard-table-container', 'children'),
    [Input('toggle-plots', 'value')]
)

def update_table_layout(is_visible):
    if is_visible:
        return generate_dashboard(height='190px')
    else:
        return generate_dashboard(height='90vh')

def fetch_and_plot_z_score(conn, ticker):
    z_score_query = f"SELECT Date, `1Y Z` FROM {ticker}_Statistics ORDER BY Date"
    z_score_data = pd.read_sql(z_score_query, conn)

    fig = go.Figure(data=[
        go.Scatter(x=z_score_data['Date'], y=z_score_data['1Y Z'], mode='lines', line_color='#FBEAEB')
    ])
    fig.update_layout(
        plot_bgcolor='#041619',
        paper_bgcolor='#041619',
        font=dict(color='#FBEAEB'),
        yaxis_title='1-Year Z-Score',
        xaxis=dict(
            zeroline=True,
            zerolinecolor='rgba(255, 255, 255, 0.5)',
            zerolinewidth=2,
            gridcolor='rgba(128, 128, 128, 0.25)',  # Gray gridlines with 50% transparency
        ),
        yaxis=dict(
            zeroline=True,
            zerolinecolor='rgba(255, 255, 255, 0.5)',
            zerolinewidth=2,
            gridcolor='rgba(128, 128, 128, 0.25)',  # Gray gridlines with 50% transparency
        ),
    )

    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector_bgcolor='#041619',
        rangeselector_activecolor='#082c33',
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    return fig

def fetch_and_plot_premium_discount(conn, ticker):
    prem_disc_query = f"SELECT Date, `Prem/Disc` FROM {ticker}_Statistics ORDER BY Date"
    prem_disc_data = pd.read_sql(prem_disc_query, conn)

    fig = go.Figure(data=[
        go.Scatter(x=prem_disc_data['Date'], y=prem_disc_data['Prem/Disc'], mode='lines', line_color='#FBEAEB')
    ])
    fig.update_layout(
        plot_bgcolor='#041619',
        paper_bgcolor='#041619',
        font=dict(color='#FBEAEB'),
        yaxis_title='Premium/Discount (%)',
        xaxis=dict(
            zeroline=True,
            zerolinecolor='rgba(255, 255, 255, 0.5)',
            zerolinewidth=2,
            gridcolor='rgba(128, 128, 128, 0.25)',  # Gray gridlines with 50% transparency
        ),
        yaxis=dict(
            zeroline=True,
            zerolinecolor='rgba(255, 255, 255, 0.5)',
            zerolinewidth=2,
            gridcolor='rgba(128, 128, 128, 0.25)',  # Gray gridlines with 50% transparency
        ),
    )

    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector_bgcolor='#041619',
        rangeselector_activecolor='#082c33',
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig

@app.callback(
    [Output("line-chart", "figure"), Output("premium-discount-chart", "figure")],
    [Input("table", "selectedRows")]
)

def update_line_chart(selected_rows):
    if not selected_rows:
        # Return empty styled plots if no rows are selected
        return create_styled_empty_plot('', '1-Year Z-Score'), create_styled_empty_plot('', 'Premium/Discount (%)')

    selected_ticker = selected_rows[0]['Ticker']
    conn = sqlite3.connect(db_file_path)

    try:
        # Fetch and plot the data for the selected ticker
        z_score_figure = fetch_and_plot_z_score(conn, selected_ticker)
        premium_discount_figure = fetch_and_plot_premium_discount(conn, selected_ticker)

        # Apply the desired styling to these plots as well
        for fig in [z_score_figure, premium_discount_figure]:
            fig.update_layout(
                plot_bgcolor='#041619',
                paper_bgcolor='#041619',
                font=dict(color='#FBEAEB')
                # Add any other styling elements you need
            )
    finally:
        conn.close()

    return z_score_figure, premium_discount_figure

def calculate_returns(df, price_col):
    """ Calculate percentage change (returns) for the given price column. """
    df['Return'] = df[price_col].pct_change()
    df = df.dropna()  # Drop rows with NaN values
    return df

def calculate_top_10_correlations(cef_nav_data, conn):
    etf_data = pd.read_sql("SELECT Date, Ticker, Close AS Price FROM ETFs", conn)
    etf_data['ETF_Return'] = etf_data.groupby('Ticker')['Price'].pct_change()
    etf_data = etf_data.dropna()

    cef_nav_data['NAV_Return'] = cef_nav_data['NAV'].pct_change()
    cef_nav_data = cef_nav_data.dropna()

    # Convert 'Date' column to datetime
    etf_data['Date'] = pd.to_datetime(etf_data['Date'])
    cef_nav_data['Date'] = pd.to_datetime(cef_nav_data['Date'])
    
    # Ensure that 'Price' and 'NAV' columns are numeric
    etf_data['Price'] = pd.to_numeric(etf_data['Price'], errors='coerce')
    cef_nav_data['NAV'] = pd.to_numeric(cef_nav_data['NAV'], errors='coerce')

    # Filter to last year's data
    one_year_ago = datetime.now() - timedelta(days=252)
    etf_data = etf_data[etf_data['Date'] > one_year_ago]
    cef_nav_data = cef_nav_data[cef_nav_data['Date'] > one_year_ago]

    merged = pd.merge(cef_nav_data, etf_data, on='Date', how='inner')

    correlations = []
    for ticker, group in merged.groupby('Ticker'):
        X = group['ETF_Return'].values.reshape(-1, 1)  # ETF daily returns
        y = group['NAV_Return'].values  # CEF NAV daily returns
        model = sm.OLS(y, sm.add_constant(X)).fit()
        correlation = round(np.corrcoef(X.flatten(), y)[0, 1], 2)
        beta = round(model.params[1], 2)
        r_squared = round(model.rsquared, 2)
        correlations.append({'etf': ticker, 'correlation': correlation, 'beta': beta, 'r_squared': r_squared})

    top_10 = sorted(correlations, key=lambda x: x['correlation'], reverse=True)[:10]
    return top_10

@app.callback(
    [Output("scatter-plot", "figure"), Output("correlation-table", "data")],
    [Input("table", "selectedRows"), Input("correlation-table", "selected_rows")],
    [State("correlation-table", "data")]
)

def update_scatter_plot_from_etf_selection(selected_cef_row, selected_etf_row, etf_table_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "table":
        if not selected_cef_row:
            raise PreventUpdate
        selected_ticker = selected_cef_row[0]['Ticker']
        etf_ticker = 'SPY'  # Default ETF or use dynamic selection
    elif trigger_id == "correlation-table" and selected_etf_row:
        selected_ticker = selected_cef_row[0]['Ticker']
        etf_ticker = etf_table_data[selected_etf_row[0]]['etf']
    else:
        raise PreventUpdate

    # Connect to the database
    conn = sqlite3.connect(db_file_path)

    try:
        nav_symbol = funds_df[funds_df['fund_ticker'] == selected_ticker]['nav_symbol'].iloc[0]

        cef_nav_query = f"SELECT Date, Close AS NAV FROM {nav_symbol}"
        cef_nav_data = pd.read_sql(cef_nav_query, conn)
        cef_nav_data['NAV_Return'] = cef_nav_data['NAV'].pct_change()
        cef_nav_data = cef_nav_data.dropna()

        etf_price_query = f"SELECT Date, Close AS ETF_Price FROM ETFs WHERE Ticker = '{etf_ticker}'"
        etf_price_data = pd.read_sql(etf_price_query, conn)
        etf_price_data['ETF_Return'] = etf_price_data['ETF_Price'].pct_change()
        etf_price_data = etf_price_data.dropna()

        merged_data = pd.merge(cef_nav_data, etf_price_data, on='Date', how='inner')

        if len(merged_data) > 1:
            X = merged_data['ETF_Return'].values.reshape(-1, 1)
            y = merged_data['NAV_Return'].values

            if len(X) > 0 and len(y) > 0:
                model = sm.OLS(y, sm.add_constant(X)).fit()
                scatter_fig = go.Figure()
                scatter_fig.add_trace(go.Scatter(x=merged_data['ETF_Return'], y=merged_data['NAV_Return'], mode='markers', marker=dict(color='#FC766A')))
                scatter_fig.add_trace(go.Scatter(x=merged_data['ETF_Return'], y=model.predict(sm.add_constant(X)), mode='lines', line=dict(color='#FBEAEB', width=0.5, dash='dash')))

                scatter_fig.update_layout(
                    showlegend=False,
                    xaxis_title='ETF Daily Returns',
                    yaxis_title='CEF NAV Daily Returns',
                    plot_bgcolor='#041619',
                    paper_bgcolor='#041619',
                    font=dict(color='#FBEAEB'),
                    xaxis=dict(gridcolor='rgba(128, 128, 128, 0.25)', zeroline=True, zerolinecolor='rgba(255, 255, 255, 0.5)', zerolinewidth=2),
                    yaxis=dict(gridcolor='rgba(128, 128, 128, 0.25)', zeroline=True, zerolinecolor='rgba(255, 255, 255, 0.5)', zerolinewidth=2)
                )

                top_10_correlation_data = calculate_top_10_correlations(cef_nav_data, conn)
                return scatter_fig, top_10_correlation_data
            else:
                return create_styled_empty_plot("Not enough data for plot", "CEF NAV Daily Returns"), []
        else:
            return create_styled_empty_plot("Not enough data for plot", "CEF NAV Daily Returns"), []

    except Exception as e:
        print(f"An error occurred: {e}")
        return create_styled_empty_plot("", "CEF NAV Daily Returns"), []

    finally:
        conn.close()

if __name__ == '__main__':
    app.run_server(debug=True)
