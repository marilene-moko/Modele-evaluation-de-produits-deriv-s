# Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize


class Management:
    def __init__(self, assets = []):
        self.assets = assets
        self.data , self.returns = self.get_data(assets), self.get_returns(assets, "2010-01-01")
        self.mu , self.sigma = [], None

    def get_data(self, assets):
        data = yf.download(assets, start="2010-01-01", group_by="ticker")
        self.data = data
        return data
    
    def get_returns(self, assets, date_="2010-01-01", freq="day"):
        # Initialisation d'un DataFrame pour les rendements
        filtered_data = self.data.loc[self.data.index > date_]

        if freq == "week":
            resampled_data = filtered_data.resample('W').last()  # Dernière valeur de chaque semaine
        elif freq == "month":
            resampled_data = filtered_data.resample('M').last()  # Dernière valeur de chaque mois
        else:  # Par défaut, "day"
            resampled_data = filtered_data

        returns = pd.DataFrame()

        # Calcul des log-rendements négatifs pour chaque actif
        for asset in assets:
            try:
                # Vérifie si la colonne 'Close' est disponible
                if 'Close' in resampled_data[asset]:
                    # Log-rendements négatifs
                    returns[asset] = np.log(resampled_data[asset]['Close'] / resampled_data[asset]['Close'].shift(1))
            except KeyError:
                print(f"Données manquantes pour {asset}")
        returns = returns.dropna()
        self.returns = returns
        
        return returns
    
    def get_parameters(self, freq="day", date_="2010-01-01"):
        self.get_returns(self.assets, date_, freq)
        valid_assets = []  # Liste pour les actifs valides
        self.mu = []  # Liste pour stocker les mu
        self.sigma = None  # Matrice de covariance

        # Facteurs d'annualisation
        annualization_factors = {
            "day": 252,  # Nombre moyen de jours de bourse par an
            "week": 52,  # Nombre de semaines dans une année
            "month": 12  # Nombre de mois dans une année
        }
        
        if freq not in annualization_factors:
            raise ValueError("Invalid frequency. Choose from 'day', 'week', or 'month'.")

        annual_factor = annualization_factors[freq]
        
        # Resampler les données selon la fréquence
        if freq == "week":
            resampled_data = self.data.resample('W').last()  # Dernière valeur de chaque semaine
        elif freq == "month":
            resampled_data = self.data.resample('M').last()  # Dernière valeur de chaque mois
        else:  # Par défaut, "day"
            resampled_data = self.data

        for asset in self.assets:
            try:
                # Calcul des rendements log
                first_value = float(resampled_data[asset]['Close'].dropna().iloc[0])
                last_value = float(resampled_data[asset]['Close'].dropna().iloc[-1])
                r_p = np.log(last_value / first_value)

                first_date = resampled_data[asset]['Close'].dropna().index[0]
                last_date = resampled_data[asset]['Close'].dropna().index[-1]

                # Calcul de la différence en années
                num_years = (last_date - first_date).days / 365.25

                # Vérifie si r_p > -1
                if r_p > -1:
                    valid_assets.append(asset)
                    mu_an = (r_p + 1)**(1/num_years) - 1  # Calcul du rendement annualisé
                    self.mu.append(mu_an)
                else:
                    print(f"The asset {asset} is excluded because r_p = {r_p} < -1")
            except KeyError:
                print(f"Missing data for {asset}")

        # Recalcul de sigma avec les actifs valides
        if valid_assets:
            filtered_returns = self.returns[valid_assets]
            
            self.sigma = filtered_returns.cov().to_numpy()
            self.sigma = annual_factor * self.sigma 

        return self.mu, self.sigma, valid_assets, filtered_returns.corr()
    
    def portfolio_variance(self, weights):
        return weights.T @ self.sigma @ weights
    
    def weight_sum_constraint(self, weights):
        return np.sum(weights) - 1

    def target_return_constraint(self, weights, mu_target):
        return weights.T @ self.mu - mu_target
    
    def efficient_portfolio(self, mu_target, range_=(-0.1, None)):
        # Initialisation des poids
        n_assets = len(self.mu)
        init_weights = np.ones(n_assets) / n_assets
        init_weights = init_weights
        # Contraintes
        constraints = [
            {'type': 'eq', 'fun': self.weight_sum_constraint},
            {'type': 'ineq', 'fun': lambda w: self.target_return_constraint(w, mu_target)}
        ]

        # Contraintes de positivité (pas de vente à découvert)
        bounds = [range_ for _ in range(n_assets)]

        result = minimize(self.portfolio_variance, init_weights, bounds= bounds, method='SLSQP', constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_volatility = np.sqrt(self.portfolio_variance(optimal_weights))
            return portfolio_volatility, optimal_weights
        else :
            return None,None