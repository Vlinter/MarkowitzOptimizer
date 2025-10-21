# MarkowitzOptimizer

# 📘 Notice  – Application d’optimisation de portefeuille (Markowitz)

## 🎯 Objectif

Cette application permet :

* d’analyser les performances de plusieurs actifs (rendement, risque, corrélation),
* d’optimiser automatiquement la répartition d’un portefeuille,
* de visualiser les portefeuilles les plus efficaces,
* et de tester leurs performances dans le temps.

---

## ⚙️ 1. Import des données

* Charge un fichier **Excel (.xlsx)** contenant :

  * **Colonne A** : les dates
  * **Colonnes suivantes** : les prix de chaque actif
    *(ex : Gold, Silver, Copper… mais ça peu ausstrès bien être des equity, bond ou autres)*
* L’app détecte automatiquement la fréquence (jour / semaine / mois).

---

## 📊 2. Optimisation

Choisis la **méthode de covariance** :

* **Échantillon** : méthode simple.
* **Ledoit–Wolf** : méthode plus robuste (recommandée et mise par défaut ici).

L’application calcule pour chaque actif :

* le **rendement annualisé**,
* la **volatilité**,
* et le **ratio de Sharpe**.

Puis elle construit trois portefeuilles :

* **Max Sharpe** → le plus efficient
* **Min Variance** → le plus stable
* **Max Return** → le plus offensif

---

## 📈 3. Graphiques

* **Nuage de portefeuilles** : montre tous les portefeuilles simulés et la **frontière efficiente** (meilleur rendement pour un niveau de risque donné).
* **Camemberts** : affichent la répartition des 3 portefeuilles optimaux --> c'est ce qui nous interesse au final ici.

---

## 🔗 4. Corrélations

* Affiche la **matrice de corrélation** entre actifs.
* Montre les **3 corrélations les plus fortes et les plus faibles**.
* Permet aussi d’analyser la **corrélation entre groupes d’actifs**.

---

## 🔁 5. Backtest

Permet de simuler les performances d’un portefeuille dans le temps.

Options :

* Choisir les poids manuellement ou utiliser ceux du **Max Sharpe** (par défaut).
* Activer ou non le **rebalancement** automatique.
* Choisir la fréquence (mensuelle, trimestrielle, etc.).

Résultats :

* Rendement total et annualisé
* Volatilité
* Sharpe ratio
* Drawdown (plus forte baisse)

---

## En résumé

| Étape | Ce que tu fais           | Ce que tu obtiens                        |
| :---- | :----------------------- | :--------------------------------------- |
| 1     | Upload ton fichier Excel | Lecture automatique des données          |
| 2     | Onglet “Optimisation”    | Poids optimaux et Sharpe max             |
| 3     | Onglet “Graphiques”      | Frontière efficiente et allocation opti  |
| 4     | Onglet “Corrélation”     | Relations entre actifs                   |
| 5     | Onglet “Backtest”        | Performance historique du portefeuille   |


