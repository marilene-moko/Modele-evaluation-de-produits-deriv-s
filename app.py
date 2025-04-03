from dash import Dash, html, Input, Output, callback, dcc, State, dash_table, ctx
import os
from datetime import datetime, timedelta
import dash

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np

from components.menu import *
from components.analyse import Analyse
from tools_class.asset_pricing import Pricing
from tools_class.asset_management import Management
from tools_class.asset_tracking import IndexReplication
from tools_class.zero_coupons import Zero_Coupons
from tools_class.optional_exotic import Optional_Excotic

import plotly.express as px
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from plotly.subplots import make_subplots

np.random.seed(102)

# Initialisation du chemin permettant le lancement de l'application
# Définition du chemin de base pour l'application Dash en utilisant une variable d'environnement pour l'utilisateur

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

path = f"/"
app = Dash(__name__, requests_pathname_prefix=path, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME], suppress_callback_exceptions = True)

app.index_string = INDEX_CONFIG

pricing = Pricing()

# Initialisation des différentes sections de l'application via des objets personnalisés
analyse = Analyse()      

CONTENT_STYLE = {
        "margin-left": "5.7rem",
        "margin-right": "5.7rem",
        "padding": "2rem 1rem",
    }

sidebar = Menu(path).render()
content =  dbc.Spinner(html.Div(id="page-content", style=CONTENT_STYLE), spinner_style={"width": "3rem", "height": "3rem"})

app.layout = html.Div(
    [
        dcc.Location(id="url"), # Permet de gérer les URLs et la navigation au sein de l'application
        sidebar, # Ajout de la barre latérale
        content, # Contenu principal avec un Spinner
        html.Button(id='load-data-button', style={"display": "none"}), # Bouton caché pour déclencher le chargement des données
        dcc.Store(id='selected-item', data='', storage_type='session'),  # Stockage temporaire de données sélectionnées en session
        html.Div(id="hidden-div", style={"display": "none"}), # Division cachée pour stocker d'autres informations ou déclencher des callbacks
    ])


options_dict = {
    True: "Call",  # Quand le switch est activé, "Call"
    False: "Put"   # Quand le switch est désactivé, "Put"
}

options_dict_value = {
    True: "call",  # Quand le switch est activé, "Call"
    False: "put"   # Quand le switch est désactivé, "Put"
}

taux_zero_coupons = pd.read_excel("data/Courbe-zero-coupon-31-decembre-2024.xlsx", sheet_name="Données",header = 1)
taux_zero_coupons.columns = ["Maturity", "Rate", "Actual Coef"]

symbole_list = [
    "AAPL",  # Apple (Technologie)
    "EEM",   # iShares MSCI Emerging Markets ETF (Pays émergents)
    "TLT",   # US Treasury Bonds (Obligations souveraines)
    "HYG",   # High Yield Corporate Bonds
    "GLD",   # Gold (Or)
    "LDOS",  # Leidos Holdings Inc
    "LLY",   # Eli Lilly & Co
    "AMD",   # AMD
    "USO",   # United States Oil Fund (Pétrole)
    "VNQ",   # Vanguard Real Estate ETF (Immobilier)
    "BTC-USD", # Bitcoin (Cryptomonnaie)
]

stocks_dict = {
    "AC.PA": "Accor",
    "AI.PA": "Air Liquide",
    "AIR.PA": "Airbus",
    "MT.AS": "ArcelorMittal",
    "CS.PA": "AXA",
    "BNP.PA": "BNP Paribas",
    "EN.PA": "Bouygues",
    "BVI.PA": "Bureau Veritas",
    "CAP.PA": "Capgemini",
    "CA.PA": "Carrefour",
    "ACA.PA": "Crédit Agricole",
    "BN.PA": "Danone",
    "DSY.PA": "Dassault Systèmes",
    "EDEN.PA": "Edenred",
    "ENGI.PA": "Engie",
    "EL.PA": "EssilorLuxottica",
    "ERF.PA": "Eurofins Scientific",
    "RMS.PA": "Hermès",
    "KER.PA": "Kering",
    "LR.PA": "Legrand",
    "OR.PA": "L'Oréal",
    "MC.PA": "LVMH",
    "ML.PA": "Michelin",
    "ORA.PA": "Orange",
    "RI.PA": "Pernod Ricard",
    "PUB.PA": "Publicis",
    "RNO.PA": "Renault",
    "SAF.PA": "Safran",
    "SGO.PA": "Saint-Gobain",
    "SAN.PA": "Sanofi",
    "SU.PA": "Schneider Electric",
    "GLE.PA": "Société Générale",
    "STLA": "Stellantis",
    "STMPA.PA": "STMicroelectronics",
    "TEP.PA": "Teleperformance",
    "HO.PA": "Thales",
    "TTE.PA": "TotalEnergies",
    "UNBLF": "Unibail-Rodamco-Westfield",
    "VIE.PA": "Veolia",
    "DG.PA": "Vinci",
}

sector_mapping = {
        "AC.PA": "Hôtellerie et Loisirs",
        "ACA.PA": "Banques",
        "AI.PA": "Industrie",
        "AIR.PA": "Aéronautique",
        "BN.PA": "Consommation de base",
        "BNP.PA": "Banques",
        "BVI.PA": "Médias",
        "CA.PA": "Distribution",
        "CAP.PA": "Technologie",
        "CS.PA": "Assurance",
        "DG.PA": "Industrie et Construction",
        "DSY.PA": "Technologie",
        "EDEN.PA": "Technologie",
        "EL.PA": "Santé",
        "EN.PA": "Services publics",
        "ENGI.PA": "Énergie",
        "ERF.PA": "Santé",
        "GLE.PA": "Banques",
        "HO.PA": "Aéronautique",
        "KER.PA": "Luxe",
        "LR.PA": "Énergie",
        "MC.PA": "Luxe",
        "ML.PA": "Transport et Logistique",
        "MT.AS": "Métaux et Mines",
        "OR.PA": "Produits de beauté",
        "ORA.PA": "Télécommunications",
        "PUB.PA": "Médias et Publicité",
        "RI.PA": "Consommation discrétionnaire",
        "RMS.PA": "Luxe",
        "RNO.PA": "Automobile", 
        "SAF.PA": "Aéronautique",
        "SAN.PA": "Santé",
        "SGO.PA": "Matériaux de Construction",
        "STLA": "Automobile",
        "STMPA.PA": "Technologie",
        "SU.PA": "Services publics",
        "TEP.PA": "Énergie",
        "TTE.PA": "Énergie",
        "UNBLF": "Immobilier",
        "VIE.PA": "Services publics"
    }


symbole_list = list(set(symbole_list))

management = Management(symbole_list)

zero_coupons = Zero_Coupons()
optionel_exotic = Optional_Excotic()

replication = IndexReplication(
            index_ticker="^FCHI",
            component_tickers=[
                "AC.PA", "AI.PA", "AIR.PA", "MT.AS", "CS.PA", "BNP.PA", "EN.PA", "BVI.PA", 
                "CAP.PA", "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "EDEN.PA", "ENGI.PA", 
                "EL.PA", "ERF.PA", "RMS.PA", "KER.PA", "LR.PA", "OR.PA", "MC.PA", "ML.PA", 
                "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA", 
                "SU.PA", "GLE.PA", "STLA", "STMPA.PA", "TEP.PA", "HO.PA", "TTE.PA", 
                "UNBLF", "VIE.PA", "DG.PA"
            ],
            start_date="2010-01-01",
            end_date=datetime.today().strftime('%Y-%m-%d'),
        )

# Fetch data
replication.get_data()

def plot_3d_surface(pricing_data, column_name):
    """
    Crée un graphique de surface 3D à partir des données de volatilité implicite.

    Arguments :
    pricing_data (pd.DataFrame) : DataFrame contenant les colonnes 'K', 'T' et une colonne de valeurs (par exemple, 'IV').
    column_name (str) : Le nom de la colonne à tracer sur l'axe Z (par exemple, 'IV').

    Retourne :
    go.Figure : Un graphique Plotly de surface 3D.
    """
    # Vérification des colonnes nécessaires
    if not all(col in pricing_data.columns for col in ["K", "T", column_name]):
        raise ValueError(f"Les colonnes 'K', 'T' et '{column_name}' doivent être présentes dans le DataFrame.")
    
    # Extraction des données
    strikes = pricing_data["K"].values
    maturities = pricing_data["T"].values
    z_values = pricing_data[column_name].values

    # Création du graphique de surface 3D
    fig = go.Figure(data=[
        go.Mesh3d(
            x=strikes,
            y=maturities,
            z=z_values,
            colorbar_title=column_name,
            colorscale='Viridis',
            intensity=z_values,
            showscale=True,
            opacity=0.9
        )
    ])

    # Mise en page
    fig.update_layout(
        title=f"{column_name} Surface",
        scene=dict(
            xaxis_title="Strike Price (K)",
            yaxis_title="Maturity (T in years)",
            zaxis_title=column_name,
        ),
        height=600,
        width=800
    )

    return fig


def plot_3d_and_2d(pricing_data, column_name, greek_function, greek_name="Greek", option_type="call"):
    """
    Crée un graphique 3D à gauche et un graphique 2D à droite avec Plotly.
    
    Arguments :
    pricing_data (pd.DataFrame) : DataFrame contenant 'K', 'T', 'S', 'sigma' et une colonne de valeurs (par exemple 'IV').
    column_name (str) : Le nom de la colonne à tracer sur l'axe Z (par exemple 'IV') dans le graphique 3D.
    greek_function (callable) : Fonction pour calculer le Greek.
    greek_name (str) : Nom du Greek pour l'étiquetage dans le graphique 2D.
    option_type (str) : Type d'option, soit "call" soit "put".
    
    Retourne :
    go.Figure : Un graphique Plotly avec deux sous-graphiques.
    """
    # Vérification des colonnes nécessaires
    if not all(col in pricing_data.columns for col in ["K", "T", "S", "IV", column_name]):
        raise ValueError("Les colonnes 'K', 'T', 'S', 'IV', et '{column_name}' doivent être présentes dans le DataFrame.")
    
    # Extraction des données pour le 3D
    strikes = pricing_data["K"].values
    maturities = pricing_data["T"].values
    z_values = pricing_data[column_name].values

    # Création du graphique 3D (Surface)
    surface = go.Mesh3d(
        x=strikes,
        y=maturities,
        z=z_values,
        colorbar_title=column_name,
        colorscale='Viridis',
        intensity=z_values,
        showscale=True,
        opacity=0.9,
        name="3D Surface"
    )

    # Sélection des strikes et maturités pour le 2D
    unique_strikes = np.linspace(pricing_data['K'].values.min(), pricing_data['K'].values.max(), 15)
    selected_maturities = [0.0, 0.5, 1.0, 1.5]  # Maturités spécifiées
    
    # Création des courbes 2D (projection)
    greek_lines = []
    for T in selected_maturities:
        greek_values = []
        for K in unique_strikes:
            S_value = pricing.price
            S_, sigma_value = pricing.price_option_by_interpolation(K, T, S_value, option_type=option_type)
            #print(sigma_value)
                                
            greek_values.append(greek_function(K, T, S_value, sigma_value, option_type=option_type))
            
        # Ajout de la courbe pour cette maturité
        greek_lines.append(go.Scatter(
            x=unique_strikes,
            y=greek_values,
            mode="lines",  # Seulement lignes continues
            name=f"T={T:.2f}",  # Nom de la courbe
            line=dict(width=2)
        ))

    # Création du layout avec deux graphiques côte à côte
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],  # Taille relative des colonnes
        specs=[[{"type": "surface"}, {"type": "xy"}]],  # Définit le type de chaque colonne
        subplot_titles=("3D Surface", "2D Greeks")
    )

    # Ajout du 3D à gauche
    fig.add_trace(surface, row=1, col=1)

    # Ajout des lignes 2D à droite
    for line in greek_lines:
        fig.add_trace(line, row=1, col=2)

    # Mise en page globale
    fig.update_layout(
        title=f"3D Surface et 2D Greeks ({greek_name})",
        scene=dict(  # Configuration pour le graphique 3D
            xaxis_title="Strike Price (K)",
            yaxis_title="Maturity (T in years)",
            zaxis_title=column_name,
        ),
        xaxis2=dict(title="Strike Price (K)"),  # Configuration pour l'axe X du graphique 2D
        yaxis2=dict(title=f"Valeur de {greek_name}"),  # Configuration pour l'axe Y du graphique 2D
        height=700,
        width=1000,
        legend=dict(  # Configuration de la légende
            groupclick="toggleitem",
            x=1.15,  # Légende des courbes 2D à droite
            y=1,
            title=dict(text="Courbes 2D")
        )
    )

    return fig




# Callback pour mettre à jour le contenu de la page en fonction du chemin d'URL
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    # Affiche le rendu correspondant à l'URL, sinon retourne l'analyse par défaut
    if pathname == f"{path}":
        return analyse.render() 
    else:
        return analyse.render()             # Page par défaut (analyse) si le chemin n'est pas reconnu



############################ ANALYSE #################################

@app.callback(
    [Output("ticker-symbole", "value"), Output("risk-free", "value"), Output("symbole-portofio", "children"),
     Output('remove-ticker-dropdown', 'options'),],
    Input('load-data-button', 'n_clicks'),
)
def initialize_elements(_):
    bdg = [dbc.Badge(s, color=analyse.color_name[np.random.randint(0, 7)], className="border me-1") for s in symbole_list]
    return "AAPL", 0.03, bdg, symbole_list

@app.callback(
    Output("standalone-value", "children"),
    Input("standalone-switch", "value")
)
def update_option(value):

    return html.Div(f"{options_dict[value]}", style={"fontWeight": "bold", "color": "red"})

@app.callback(
    [Output('ticker-pricing-graph', 'figure'),
     Output('error-popup', 'is_open')],
    [Input('ticker-symbole', 'value'),
     Input('close-error-popup', 'n_clicks')],
)
def update_graph(ticker_, close_error_clicks):
    # Détection si la fermeture du popup est cliquée

    if ctx.triggered_id == "close-error-popup":
        return dash.no_update, False
    # Tentative de récupération des données pour le ticker
    try:
        dta_ = yf.download(ticker_, start='2010-01-01')
        if ticker_ is None or dta_.shape[0]<1:
            raise ValueError("Invalid ticker")
        
        if len(dta_.columns) == 5: 
            dta_.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        else:
            dta_.columns = ['Adj Close','Close', 'High', 'Low', 'Open', 'Volume']
           
        pricing.get_data(ticker_)
        pricing.ticker = ticker_
        
        pricing.price = float(dta_['Close'].values[-1]) 
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=dta_.index, y=dta_['Close'], mode='lines', name='Close'))
        
        fig.update_layout(
        title=f'{ticker_} Closing Price Evolution',
    )
        
        return fig, False
    
    except Exception:
        # Retourne un graphique par défaut (ligne droite) et affiche le popup
        default_data = pd.DataFrame({
            'Date': pd.date_range(start='2010-01-01', periods=10),
            'Close': [10] * 10
        }).set_index('Date')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=default_data.index, y=default_data['Close'], mode='lines', name='Close'))
        return fig, True



@app.callback(
    [Output("modal-xl", "is_open"), Output("volatility-graph", "figure"), Output("output-greeks", "children")],
    [Input("open-volatility", "n_clicks"), Input("risk-free", "value")],
    [State("modal-xl", "is_open"), State('ticker-symbole', 'value'), State("standalone-switch", "value")]
)
def compute_iv_button(n_clicks, risk, is_open, ticker, option_type):
    if n_clicks > 0:
        # Vérifier si la colonne "Volatilité" existe, sinon la calculer
        if "IV" not in pricing.data.columns or ticker != pricing.ticker:
            if risk == None:
                risk = 0.03
            pricing.compute_iv(risk)
            pricing.calculate_greeks(risk, option_type=options_dict_value[option_type])

        pricing.data = pricing.data.dropna(subset=['T', 'K', 'IV'])

        strikes = pricing.data["K"].values
        maturities = pricing.data["T"].values
        implied_vols = pricing.data["IV"].values

        # Création de la surface 3D
        fig = go.Figure(data=[go.Mesh3d(
            x=strikes,
            y=maturities,
            z=implied_vols,
            colorbar_title='Volatilité',
            colorscale='Viridis',
            intensity=implied_vols,
            showscale=True,
            opacity=0.9
        )])

        # Ajouter des titres et des étiquettes
        fig.update_layout(
            title="Implied Volatility Surface",
            scene=dict(
                xaxis_title="Strike Price (K)",
                yaxis_title="Maturity (T in years)",
                zaxis_title="Implied Volatility (IV)",
            ),
            height=600,
            width=800
        )

        graph = dcc.Tabs([
                dcc.Tab(label='Delta', children=[
                        dbc.Row(
                                [   
                                    dcc.Graph(figure=plot_3d_and_2d(pricing.data, "Delta", pricing.delta_greek, greek_name="Greek")),
                                ])
                    
                        ]),
                dcc.Tab(label='Gamma', children=[
                        dbc.Row(
                                [   
                                    dcc.Graph(figure=plot_3d_and_2d(pricing.data, "Gamma", pricing.gamma_greek, greek_name="Greek")),
                                ])
                    
                        ]),

                dcc.Tab(label='Vega', children=[
                        dbc.Row(
                                [
                                    dcc.Graph(figure=plot_3d_and_2d(pricing.data, "Vega", pricing.vega_greek, greek_name="Greek")),
                                ])
                    
                        ]),
                dcc.Tab(label='Theta', children=[
                        dbc.Row(
                                [
                                    dcc.Graph(figure=plot_3d_and_2d(pricing.data, "Theta", pricing.theta_greek, greek_name="Greek")),
                                ])
                    
                        ]),
                ])
        
        return True, fig, graph

    return is_open, dash.no_update, dash.no_update



@app.callback(
    [Output("output-div", "children"), Output("open-interpol", "disabled")],
    State("risk-free", "value"),
    State("standalone-switch", "value"),
    State("input-K", "value"),
    Input("input-T", "value"),
    prevent_initial_call=True
)
def interpole_iv(r, option_type, K, T):
    if K is None or T is None:
        return dbc.Alert("Please provide valid inputs for K and T.", color="danger")
    if "IV" not in pricing.data.columns:
            if r == None:
                r = 0.03
            pricing.compute_iv(r)
    
    S0 = pricing.price
    price, volatility= pricing.price_option_by_interpolation(K, T, S0, r=r, option_type=options_dict_value[option_type])
    
    greeks = pricing.compute_greeks(K, T, S0, volatility, r=r, option_type=options_dict_value[option_type]) 

    # Mise en page en deux colonnes
    greeks_child = dbc.Row([
        dbc.Col(html.Ul([
            html.Li(f"S0: {S0:.2f}", style={"marginBottom": "10px"}),
            html.Li(f"Price: {price:.2f}", style={"marginBottom": "10px"}),
            html.Li(f"IV: {volatility:.2f}", style={"marginBottom": "10px"}),
            html.Li(f"Delta: {greeks['delta']:.2f}", style={"marginBottom": "10px"}),
        ]), width=6),

        dbc.Col(html.Ul([
            html.Li(f"Gamma: {greeks['gamma']:.2f}", style={"marginBottom": "10px"}),
            html.Li(f"Vega: {greeks['vega']:.2f}", style={"marginBottom": "10px"}),
            html.Li(f"Theta: {greeks['theta']:.2f}", style={"marginBottom": "10px"})
        ]), width=6)
    ],)
    return greeks_child, False



@app.callback(
    [Output("modal-xl-interpol", "is_open"), Output("interpol-graph", "figure")],
    State("risk-free", "value"),
    State("standalone-switch", "value"),
    State("input-K", "value"),
    State("input-T", "value"),
    State("modal-xl-interpol", "is_open"),
    Input("open-interpol", "n_clicks"),
    prevent_initial_call=True
)
def interpole_show(r, option_type, K, T, is_open, n_clicks):
    if n_clicks > 0:
        if K is None or T is None:
            return dbc.Alert("Please provide valid inputs for K and T.", color="danger")
        if "IV" not in pricing.data.columns:
                if r == None:
                    r = 0.03
                pricing.compute_iv(r)
        
        S0 = pricing.price
        _, volatility= pricing.price_option_by_interpolation(K, T, S0, r=r, option_type=options_dict_value[option_type])
        
        points = np.round(pricing.data[['K', 'T']].values, 2)
        volatilities = np.round(pricing.data['IV'].values, 2)

        # Trouver les points qui encadrent la cible
        K_values = np.unique(points[:, 0])
        T_values = np.unique(points[:, 1])

        # Trouver les plus proches valeurs de K et T qui encadrent K_target et T_target
        K_left = K_values[K_values <= K].max()
        K_right = K_values[K_values >= K].min()
        T_bottom = T_values[T_values <= T].max()
        T_top = T_values[T_values >= T].min()

        # Points qui encadrent
        encasing_points = np.array([
            [K_left, T_bottom],
            [K_left, T_top],
            [K_right, T_bottom],
            [K_right, T_top],
        ])
        #encasing_vols = [volatilities[np.where((points == p).all(axis=1))[0][0]] for p in encasing_points]

        limited_points = None
        limited_volatilities = None
        max_points = 20
        if len(points) > max_points:
            random_indices = np.random.choice(len(points), size=max_points, replace=False)
            limited_points = points[random_indices]
            limited_volatilities = volatilities[random_indices]
        else:
            limited_points = points
            limited_volatilities = volatilities
            
        fig = go.Figure()

        # Existing points
        fig.add_trace(go.Scatter(
            x=limited_points[:, 0],
            y=limited_points[:, 1],
            mode='markers+text',
            marker=dict(size=10, color=limited_volatilities, colorscale='Viridis', showscale=True),
            text=[f'({x}, {y})' for x, y in points],
            textposition="top center",
            name="Existing Points"
        ))

        # Target point
        fig.add_trace(go.Scatter(
            x=[K],
            y=[T],
            mode='markers+text',
            marker=dict(size=15, color='red', line=dict(width=2, color='black')),
            text=[f'Target (vol={volatility:.3f})'],
            textposition="top center",
            name="Target Point"
        ))

        # Encasing points
        fig.add_trace(go.Scatter(
            x=encasing_points[:, 0],
            y=encasing_points[:, 1],
            mode='markers+text',
            marker=dict(size=12, color='orange', line=dict(width=2, color='black')),
            text=[f'({x}, {y})' for x, y in encasing_points],
            textposition="top center",
            name="Encasing Points"
        ))

        # Layout adjustments
        fig.update_layout(
            title="Volatility Interpolation and Encasing Points",
            xaxis_title="K (Strike)",
            yaxis_title="T (Time to Maturity)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            coloraxis_colorbar=dict(title="Volatility"),
            width=800,
            height=600,
            template="plotly_white"
        )

        return True, fig

    return is_open, dash.no_update



@app.callback(
    [Output("symbole-portofio", "children", allow_duplicate=True),
     Output('error-popup', 'is_open', allow_duplicate=True), Output('remove-ticker-dropdown', 'options', allow_duplicate=True)],
    [Input('add-ticker-management', 'value'),
     Input('close-error-popup', 'n_clicks')],
     prevent_initial_call=True,
)
def add_symbole(ticker, close_error_clicks):
    global symbole_list
    # Détection si la fermeture du popup est cliquée
    if ctx.triggered_id == "close-error-popup":
        return dash.no_update, False, symbole_list
    
    # Tentative de récupération des données pour le ticker
    try:
        dta = yf.download(ticker, start='2010-01-01')
        if ticker is None or dta.shape[0]<1:
            raise ValueError("Invalid ticker")
        
        symbole_list.append(ticker)
        symbole_list = list(set(symbole_list))
        bdg = [dbc.Badge(s, color=analyse.color_name[np.random.randint(0, 7)], className="border me-1") for s in symbole_list]
        return bdg, False, symbole_list
    
    except Exception:
        bdg = [dbc.Badge(s, color=analyse.color_name[np.random.randint(0, 7)], className="border me-1") for s in symbole_list]
        return bdg, True, symbole_list
    

@app.callback(
    [Output("symbole-portofio", "children", allow_duplicate=True),
     Output('error-popup', 'is_open', allow_duplicate=True), Output('remove-ticker-dropdown', 'options', allow_duplicate=True)],
    [Input('remove-ticker-dropdown', 'value')],
     prevent_initial_call=True,
)
def remove_symbole(ticker):
 
    global symbole_list

    if ticker is not None:
        symbole_list.remove(ticker)
        bdg = [dbc.Badge(s, color=analyse.color_name[np.random.randint(0, 7)], className="border me-1") for s in symbole_list]
        return bdg, False, symbole_list
    else :
        bdg = [dbc.Badge(s, color=analyse.color_name[np.random.randint(0, 7)], className="border me-1") for s in symbole_list]
        return bdg, False, symbole_list
    

@app.callback(
    [Output('portfolio-graph', 'figure'), Output('data-table', 'data'),
     Output('data-table', 'columns'), Output("modal-xl-corr", "is_open"),  Output("correlation-graph", "figure")],
    [Input('run-frontier', 'n_clicks'), Input("open-correlation", "n_clicks")], 
    [State('radio-type', 'value'), State('input-weight-inf', 'value'),State("input-weight-sup", "value"),
     State("date-picker", "date"), State("risk-free", "value"), State("modal-xl-corr", "is_open")]
)
def update_graph_portfolio(n_click, cor_n, type_freq, inf_w, sup_w, date_, r_, open):
    mu_, _, symb_list, corr_matrice = management.get_parameters(freq = type_freq, date_=date_)
    rf = 0.03
    if (n_click>0) and (r_ != None):
        rf = r_

    # Rendements cibles
    mu_targets_ = np.linspace(0.01, 0.2, 100)
    sml_volatilities = []
    sml_weights = []
    mu_targets = []

    # Calcul de la frontière efficiente
    for mu_target in mu_targets_:
        vol, weights = management.efficient_portfolio(mu_target, range_= (inf_w, sup_w))
        if vol is not None and weights is not None:
            sml_volatilities.append(vol)
            sml_weights.append(weights)
            mu_targets.append(mu_target)

    # Identifier le portefeuille tangent (maximisation du ratio de Sharpe)
    sharpe_ratios = (np.array(mu_targets) - rf) / np.array(sml_volatilities)
    market_index = np.argmax(sharpe_ratios)
    market_volatility = sml_volatilities[market_index]
    market_return = mu_targets[market_index]
    market_weights = sml_weights[market_index]

    # Pente de la CML
    cml_slope = (market_return - rf) / market_volatility

    # Étendre la CML
    extended_volatilities = np.linspace(0, max(sml_volatilities) * 1.1, 200)
    cml_y = rf + cml_slope * extended_volatilities

    fig = go.Figure()

    # Ajouter la frontière efficiente
    fig.add_trace(go.Scatter(
        x=sml_volatilities, 
        y=mu_targets,
        mode='lines',
        name='Efficient frontier',
        line=dict(color='blue')
    ))

    # Ajouter la CML étendue
    fig.add_trace(go.Scatter(
        x=extended_volatilities, 
        y=cml_y,
        mode='lines',
        name='CML (Capital Market Line)',
        line=dict(color='red', dash='dash')
    ))

    # Marquer le portefeuille du marché
    fig.add_trace(go.Scatter(
        x=[market_volatility], 
        y=[market_return],
        mode='markers',
        name='Market Portfolio',
        marker=dict(color='green', size=10)
    ))

    # Marquer le taux sans risque
    fig.add_trace(go.Scatter(
        x=[0], 
        y=[rf],
        mode='markers',
        name='Risk-free rate',
        marker=dict(color='black', size=10)
    ))

    # Configurer le graphique
    fig.update_layout(
        title=f'Extended CML with Market Portfolio -- Vol = {market_volatility:.2f} & Return = {market_return:.2f}',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Return',
        legend=dict(x=0.02, y=0.98),
        template='plotly_white'
    )

    df_market_weights = pd.DataFrame({
        'Asset': symb_list,
        'Return': np.round(mu_, 3),
        'Weight': np.round(market_weights, 5)
    })

    df_market_weights = df_market_weights.reindex(
        df_market_weights['Weight'].abs().sort_values(ascending=False).index
    )
    
    table_data = df_market_weights.to_dict('records')
    table_columns = [{'name': col, 'id': col} for col in df_market_weights.columns]

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr_matrice.to_numpy(),  # Les valeurs de corrélation
            x=corr_matrice.columns,  # Les noms des colonnes (actifs)
            y=corr_matrice.index,  # Les noms des lignes (actifs)
            colorscale="Viridis",  # Échelle de couleurs
            colorbar=dict(title="Correlation"),  # Titre de la barre de couleur
        )
        )

    # Ajout du titre et des ajustements
    fig_corr.update_layout(
        title="Asset Correlation Matrix",
        xaxis=dict(title="Assets"),
        yaxis=dict(title="Assets"),
        width=600,
        height=600,
    )
    _open = open
    if cor_n>0 : 
        _open = True
    return fig, table_data, table_columns, _open, fig_corr




################### TRACKING ERROR ####################

@app.callback(
    [Output("tracking-error-graph", "figure"),
     Output("annualized-returns-graph", "figure"),
     Output("optimized-weights-table", "data")],
    [Input("run-backtest", "n_clicks")],  
    [State("start-date-picker", "date"),
     State("end-date-picker", "date"),
     State("data-rebalancing", "value"),
     State("sector-standalone-switch", "value"),
     State("max-assets", "value")] 
)
def update_tracking_error(n_clicks, start_date, end_date, frequency,sector, max_assets):
    if not n_clicks or n_clicks <= 0:
        return dash.no_update, dash.no_update, []

    try:
        # Initialize IndexReplication instance
        replication.monthly = (frequency == "M")
        replication.get_sub_data(start_date, end_date)

        if max_assets == None:
            max_assets = 40
        # Run backtest
        tracking_df, annualized_portfolio_return, annualized_benchmark_return = replication.run_backtest(max_assets)
        # Convert `Tracking Error` to a DataFrame with clean floats
        tracking_df["Tracking Error"] = tracking_df["Tracking Error"].apply(
            lambda x: float(x.iloc[0]) if isinstance(x, pd.Series) else x
        )
        

        # Create Tracking Error Graph
        fig_te = go.Figure()
        fig_te.add_trace(go.Scatter(
            x=tracking_df["Period"], y=tracking_df["Tracking Error"],
            mode="lines+markers", name="Tracking Error"
        ))
        per = 'month'
        if frequency!='M':
            per = 'year'
        fig_te.update_layout(
            title=f"CAC 40 tracking error by {per}",
            xaxis_title="Year",
            yaxis_title="Tracking Error",
            template="plotly_white"
        )

        # Create Annualized Returns Graph
        fig_ar = go.Figure()
        fig_ar.add_trace(go.Scatter(
            x=annualized_portfolio_return.index, y=annualized_portfolio_return,
            mode="lines", name="Portfolio Returns"
        ))
        fig_ar.add_trace(go.Scatter(
            x=annualized_benchmark_return["^FCHI"].index,y=annualized_benchmark_return["^FCHI"].values,
            mode="lines",
            name="Benchmark Returns"
        ))
        fig_ar.update_layout(
            title="CAC 40 replication backtest",
            xaxis_title="Date",
            yaxis_title="Return",
            template="plotly_white"
        )

        # Optimized Weights Data
        if sector :
            sector_weights = {}
            for ticker, weight in replication.weights_history[-1].items():
                sector = sector_mapping.get(ticker, "Non attribué")
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

            # Conversion des poids par secteur en liste de dictionnaires
            weights_data = [{"Sector": sector, "Weight": round(weight * 100, 2)}
                                    for sector, weight in sector_weights.items()]
        else:
            weights_data = [{"Ticker": stocks_dict[ticker], "Sector" : sector_mapping[ticker] ,"Symbol": ticker, "Weight": round(weight * 100, 2)}
                for ticker, weight in replication.weights_history[-1].items()]

        
        return fig_te, fig_ar, weights_data
    except Exception as e:
        print(f"Error during callback execution: {e}")
        return dash.no_update, dash.no_update, [{"Ticker": "Error", "Weight": str(e)}]


################### OTHER RATE/PRICING ####################

def generate_interpolation_graph(maturities, zero_coupon_rates):

    tau_fine, fitted_rates = zero_coupons.interpoler_taux_by_nelson_siegel(maturities, zero_coupon_rates)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=maturities, y=zero_coupon_rates, mode='markers+lines', 
                                name="Observed Zero Coupon Rates", marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=tau_fine, y=fitted_rates, mode='lines',
                                name="Fitted Curve (Nelson-Siegel)", line=dict(color='orange',  dash='dash')))
    fig.update_layout(title="Zero Coupon Rate Curve Fitting (Nelson-Siegel)",
                        xaxis_title="Maturity (Years)",
                        yaxis_title="Zero Coupon Rate (%)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    return fig


@app.callback(
    [Output("zero-coupons-table", "data"),
     Output("zero-coupons-table", "columns"),
     Output("interpolate-graph", "figure")],
    Input('load-data-button', 'n_clicks')
)
def update_interpolation(n_clicks):
    # Load data
    maturities = taux_zero_coupons["Maturity"]
    zero_coupon_rates = taux_zero_coupons["Rate"]
    
    # Generate table

    table_data = taux_zero_coupons.copy()
    table_data[["Rate", "Actual Coef"]] = table_data[["Rate", "Actual Coef"]].round(3)
    table_data = table_data.to_dict("records")
    table_columns = [{"name": col, "id": col} for col in ["Maturity", "Rate", "Actual Coef"]]
    
    # Generate graph
    figure = generate_interpolation_graph(maturities, zero_coupon_rates)
    
    return table_data, table_columns, figure


@app.callback(
    Output("input-princing-div", "children"),
    Input("princing-method", "value")
)
def update_input_princing(method):
    hide = {"display": "none"}
    show = {}
    coupon = show
    principal = show
    delta = show
    amplitude_echeances = show
    maturity = show
    time = show
    bloc_1 = show
    bloc_2 = show
    bloc_3 = show
    if method == "flex-bond":
        delta = hide
        time = hide
        bloc_3 = hide
        
    elif method == "swap-rate":
        coupon = hide
        principal = hide
        amplitude_echeances = hide
        time = hide
        bloc_1 = hide
        bloc_3 = hide
        
    elif method == "fra":
        coupon = hide
        principal = hide
        amplitude_echeances = hide
        bloc_1 = hide
        bloc_3 = hide
    
    child = [dbc.Row([
                    dbc.Col([
                        dbc.Input(id="coupon", type="number", placeholder="Coupon", className="mb-3")
                    ], width=6, style=coupon),
                    dbc.Col([
                        dbc.Input(id="principal", debounce=True, type="number", placeholder="Main", className="mb-3")
                    ], width=6, style=principal)
                ], style=bloc_1),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Input(id="delta", type="number", placeholder="Delta", className="mb-3")
                    ], width=6, style=delta),
                    dbc.Col([
                        dbc.Input(id="amplitude_echeances", type="number", placeholder="Amplitude Deadlines", className="mb-3")
                    ], width=6, style=amplitude_echeances),
                    dbc.Col([
                        dbc.Input(id="maturity", debounce=True, type="number", placeholder="Maturity", className="mb-3")
                    ], width=6, style=maturity),
                ], style=bloc_2),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Input(id="time", type="number", placeholder="Time", className="mb-3")
                    ], width=12, style=time)
                ], style=bloc_3),
                ]
    return child


@app.callback(
    Output("output-princing-div", "children"),
    [State("princing-method", "value"), State("coupon", "value"),State("principal", "value"),
     State("amplitude_echeances", "value"), State("maturity", "value"), State("delta", "value"),
     ],
     Input("run-pricing", "n_clicks"),
    prevent_initial_call=True,
)
def update_output_princing(method, coupon, principal, amplitude_echeances, maturity, delta, n_clicks):
    
    taux_zero_coupons_value = taux_zero_coupons["Rate"]
    discount_factors = taux_zero_coupons["Actual Coef"]
    val = None
    if method == "flex-bond":
        if (coupon != None and principal != None and amplitude_echeances != None and maturity != None):
            val = zero_coupons.pricer_obligation_flexible(coupon, principal, amplitude_echeances ,maturity, taux_zero_coupons_value)
        return html.H5(f"Price : {val:.2f}")
    elif method == "swap-rate":
        if (delta != None and maturity != None):
            K = zero_coupons.calculate_fixed_swap_rate_flexible(taux_zero_coupons_value, discount_factors, delta, maturity)
            S = zero_coupons.swap_rate_flexible(taux_zero_coupons_value, discount_factors, delta, maturity)
        return html.Div([html.H5(f"K : {K:.2f}"), html.H5(f"S : {S:.2f}")]) 
    elif method == "fra":
        if (delta != None and maturity != None):
            val = zero_coupons.calculate_FRA(taux_zero_coupons_value ,discount_factors, maturity, delta, t=0)
        return html.H5(f"FRA Rate : {val:.2f}")
    else:
        return html.H5(f"Result : {val:.2f}")
    

def barrier_option_wrapper(S0, K, T, r, sigma, barrier, option_type, barrier_type, n_paths, n_steps, seed=102):
    return optionel_exotic.barrier_option(S0, K, T, r, sigma, n_paths, n_steps, barrier, option_type, barrier_type, seed=seed)
    

@app.callback(
    [Output("simulate-barrier-graph", "figure"),
     Output("output-barrier-div", "children")],
    [State("price_s0-bar", "value"),
     State("maturity-bar", "value"),
     State("risk-free-bar", "value"),
     State("volatility-bar", "value"),
     State("strike-bar", "value"),
     State("barrier-bar", "value"),
     State("barrier-type", "value"),
     State("option-type", "value")],
     Input("run-barrier", "n_clicks"),
)
def simulate_option_barrier(price_s0, maturity, risk_free, volatility, strike, barrier, barrier_type, option_type, n_clicks):
    if (price_s0 != None and maturity != None and risk_free != None and volatility != None and strike != None and barrier != None):
        barrier_price = optionel_exotic.barrier_option(price_s0, strike, maturity, risk_free, volatility, 50, 1000, barrier, option_type, barrier_type, seed=102)

        
        paths = optionel_exotic.simulate_black_scholes(price_s0, maturity, risk_free, volatility, n_paths=1000, n_steps=50, seed=102)
        
        delta_barrier = optionel_exotic.calculate_greeks_barrier(barrier_option_wrapper, price_s0, strike, maturity, risk_free, volatility, barrier, option_type, barrier_type, n_paths=1000, n_steps=50, greek="delta", seed=102)
        gamma_barrier = optionel_exotic.calculate_greeks_barrier(barrier_option_wrapper, price_s0, strike, maturity, risk_free, volatility, barrier, option_type, barrier_type, n_paths=1000, n_steps=50, greek="gamma", seed=102)
        vega_barrier = optionel_exotic.calculate_greeks_barrier(barrier_option_wrapper, price_s0, strike, maturity, risk_free, volatility, barrier, option_type, barrier_type, n_paths=1000, n_steps=50, greek="vega", seed=102)
        theta_barrier = optionel_exotic.calculate_greeks_barrier(barrier_option_wrapper, price_s0, strike, maturity, risk_free, volatility, barrier, option_type, barrier_type, n_paths=1000, n_steps=50, greek="theta", seed=102)
        rho_barrier = optionel_exotic.calculate_greeks_barrier(barrier_option_wrapper, price_s0, strike, maturity, risk_free, volatility, barrier, option_type, barrier_type, n_paths=1000, n_steps=50, greek="rho", seed=102)

       
        greeks_child = dbc.Row([
                dbc.Col(html.Ul([
                    html.Li(f"Price: {barrier_price:.2f}", style={"marginBottom": "10px"}),
                    html.Li(f"Delta: {delta_barrier:.2f}", style={"marginBottom": "10px"}),
                    html.Li(f"Gamma: {gamma_barrier:.2f}", style={"marginBottom": "10px"}),
                    html.Li(f"Vega: {vega_barrier:.2f}", style={"marginBottom": "10px"}),
                    html.Li(f"Theta: {theta_barrier:.2f}", style={"marginBottom": "10px"}),
                    html.Li(f"Rho: {rho_barrier:.2f}", style={"marginBottom": "10px"})
                ]), width=12)
            ],)
        
        fig = go.Figure()
        for i in range(5):
            fig.add_trace(go.Scatter(x=np.arange(paths.shape[0]), y=paths[:, i], mode='lines', name=f"Path {i+1}"))

        # Ajouter la ligne de barrière
        fig.add_trace(go.Scatter(x=np.arange(paths.shape[0]), y=[barrier] * paths.shape[0],
                                mode='lines', line=dict(color='red', dash='dash'),
                                name=f"Barrier={barrier}"))

        # Ajouter le titre et les labels
        fig.update_layout(
            title=f"Paths Barrier Option (Barrier={barrier}) - Price : {barrier_price:.2f}",
            xaxis_title="Time Steps",
            yaxis_title="Price",
            showlegend=True,
            template="plotly_white"
        )
        return fig, greeks_child
    else:
        return dash.no_update, dash.no_update



@app.callback(
    [Output("simulate-asian-graph", "figure"),
     Output("output-asian-div", "children")],
    [State("price_s0-asian", "value"),
     State("maturity-asian", "value"),
     State("risk-free-asian", "value"),
     State("volatility-asian", "value"),
     State("strike-asian", "value"),
     State("delta-asian", "value"),
     State("option-type-asian", "value")],
     Input("run-asian", "n_clicks"),
)
def simulate_option_asian(price_s0, maturity, risk_free, volatility, strike, delta, option_type, n_clicks):
    if (price_s0 != None and maturity != None and risk_free != None and volatility != None and strike != None and delta != None):
        
        asian_price, paths, _, asian_expectation = optionel_exotic.asian_option_monte_carlo(price_s0, strike, maturity, risk_free, volatility, n_paths=1000, n_steps=50, delta=delta, option_type=option_type, seed=102)


        delta_asian = optionel_exotic.calculate_greeks_asiatique(optionel_exotic.asian_option_monte_carlo,price_s0, strike, maturity, risk_free, volatility, delta, greek="delta", seed=102)
        gamma_asian = optionel_exotic.calculate_greeks_asiatique(optionel_exotic.asian_option_monte_carlo, price_s0, strike, maturity, risk_free, volatility, delta, greek="gamma", seed=102)
        vega_asian = optionel_exotic.calculate_greeks_asiatique(optionel_exotic.asian_option_monte_carlo,  price_s0, strike, maturity, risk_free, volatility, delta, greek="vega", seed=102)
        theta_asian = optionel_exotic.calculate_greeks_asiatique(optionel_exotic.asian_option_monte_carlo, price_s0, strike, maturity, risk_free, volatility, delta, greek="theta", seed=102)
        rho_asian = optionel_exotic.calculate_greeks_asiatique(optionel_exotic.asian_option_monte_carlo,  price_s0, strike, maturity, risk_free, volatility, delta, greek="rho", seed=102)

        greeks_child = dbc.Row([
                dbc.Col(html.Ul([
                    html.Li(f"Avg over [T-delta, T]: {asian_price:.2f}", style={"marginBottom": "10px"}),
                    html.Li(f"Expectation E[S_T - K]: {asian_expectation:.2f}", style={"marginBottom": "10px"}),
                    html.Li(f"Delta: {delta_asian:.2f}", style={"marginBottom": "10px"}),
                    html.Li(f"Gamma: {gamma_asian:.2f}", style={"marginBottom": "10px"}),
                    html.Li(f"Vega: {vega_asian:.2f}", style={"marginBottom": "10px"}),
                    html.Li(f"Theta: {theta_asian:.2f}", style={"marginBottom": "10px"}),
                    html.Li(f"Rho: {rho_asian:.2f}", style={"marginBottom": "10px"})
                ]), width=12)
            ],)
        
        fig = go.Figure()
        for i in range(5):
            fig.add_trace(go.Scatter(x=np.arange(paths.shape[0]), y=paths[:, i], mode='lines', name=f"Path {i+1}"))

        # Ajouter le titre et les labels
        fig.update_layout(
            title=f"Paths for Asian Option (Averaging over [T-delta, T] with delta={delta:.2f})",
            xaxis_title="Time Steps",
            yaxis_title="Price",
            showlegend=True,
            template="plotly_white"
        )
        return fig, greeks_child
    else:
        return dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run(debug=True)
    
