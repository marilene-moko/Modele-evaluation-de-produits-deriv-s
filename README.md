# Modele-evaluation-de-produits-deriv-s
Ce projet porte sur la gestion d’un projet de pricing et la mise en place d’une base de gestion d’actifs. Il couvre les grandes phases du processus de pricing des instruments financiers ainsi que l’optimisation d’un portefeuille d’actifs en utilisant des modèles financiers classiques.

1. **Extraction ou Simulation de Données de Marché**  
   - **Courbe des taux zéro-coupon** extraite du site de la **Banque de France** (dossier `data/`).
   - **Données de marché** pour les actions et la volatilité implicite simulées ou extraites de plateformes publiques comme **Yahoo Finance**.

2. **Pricing d'instruments de taux vanille**  
   - **Calcul du prix d'une obligation à coupons** à l'aide des méthodes classiques de valorisation.
   - **Pricing d'un Swap ou d'un Future** sans modèle complexe, basé sur des calculs simples utilisant la courbe de taux zéro-coupon.

3. **Implémentation et calibration d’un modèle equity (Black-Scholes)**  
   - **Modèle Black-Scholes** pour le pricing d'options sur actions.
   - **Calibration des paramètres** du modèle à partir des données de marché : volatilité implicite, taux d'intérêt, etc.

4. **Pricing d’un produit optionnel Equity avec les grecques**  
   - **Pricing d’options** européennes, américaines ou asiatiques en utilisant les grecques (Delta, Gamma, Vega, etc.).

5. **Optimisation de portefeuille**  
   - **Construction d’un portefeuille optimal** avec la théorie de **Markowitz** (frontière efficiente). Ce modèle permet de maximiser le rendement pour un niveau de risque donné ou minimiser le risque pour un rendement attendu donné.
   - **Réplication d’un indice** via une gestion passive, en optimisant les poids des actifs.


