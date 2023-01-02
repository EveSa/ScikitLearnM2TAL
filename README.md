# Devoir Scikit Learn

Le script contenu dans ce dossier permet d'entraîner un modèle de classification automatique binaire d'avis. Les données utilisées pour l'entraînement sont issues de IMDB.

## Pré-requis

Installer le fichier requirements.txt avec la commande `pip install -r requirements.txt`

## Utilisation 

Le modèle à entraîner peut être choisi en modifiant la variable `name` parmi les choix suivants :
- 'Arbre de decision'
- 'SVM'
- 'Regression Logistique'
- 'Random Forest'
- 'Naive Bayes'

Lancer le script avec la commande `python3 DevoirScikitlearn.py`

Le rapport de classification doit s'afficher directement dans le terminal.

## Résultats des modèles
| modele | precision | rappel | f-mesure |
|--------|-----------|--------|----------|
| Arbre de décision | 0.69| 0.69 | 0.69 |
| SVM | 0.87 | 0.87 | 0.87 |
| Regression Logitique | 0.85 | 0.85 | 0.85 |
| Random Forest | 0.76 | 0.76 | 0.76 |
| Naive Bayes | 0.81 | 0.78 | 0.78 |
