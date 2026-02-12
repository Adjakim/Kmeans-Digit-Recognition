#  Projet Kmeans Digit Recognition

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Description

Projet de reconnaissance de chiffres manuscrits (0-9) utilisant plusieurs algorithmes de Machine Learning. Ce projet compare les performances de différents modèles de classification sur le dataset MNIST et génère des prédictions pour de nouvelles données.

## 🎯 Objectifs

- Développer et comparer plusieurs modèles de classification
- Optimiser les hyperparamètres pour obtenir les meilleures performances
- Créer un système de prédiction fiable pour la reconnaissance de chiffres manuscrits
- Visualiser et analyser les résultats

##  Technologies Utilisées

### Librairies Python
- **Pandas** & **NumPy** - Manipulation et analyse de données
- **Matplotlib** & **Seaborn** - Visualisation de données
- **Scikit-learn** - Modèles de Machine Learning
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Machine (SVM)
- **Pickle** - Sauvegarde des modèles entraînés

##  Dataset

Le projet utilise le dataset MNIST de chiffres manuscrits :
- **Train Set** : 42,000 images (28×28 pixels = 784 features)
- **Test Set** : 28,000 images
- **Classes** : 10 chiffres (0-9)

###  Télécharger les données

 **Important** : Les fichiers de données ne sont pas inclus dans le repository GitHub car trop volumineux.

** Google Drive  :**

📦 **[Télécharger les datasets et le modèle depuis Google Drive](https://drive.google.com/drive/folders/1x0a4Kauqrky1490-vBnWdrZ_IbegDC3g?usp=sharing)**

Le dossier contient :
- `train.csv` (73 MB) - Dataset d'entraînement
- `test.csv` (18 MB) - Dataset de test  
- `best_model.pkl` (172 MB) - Modèle entraîné


### Structure des données
```
train.csv : [label, pixel0, pixel1, ..., pixel783]
test.csv  : [pixel0, pixel1, ..., pixel783]
```

##  Installation

### Prérequis
- Python 3.13+
- pip

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone https://github.com/Adjakim/Kmeans-Digit-Recognition.git
cd Kmeans-Digit-Recognition
```

2. **Télécharger les datasets** depuis le lien Google Drive ci-dessus et les placer dans le dossier du projet

3. **Créer un environnement virtuel (recommandation)**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

4. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 📁 Structure du Projet

```
Kmeans-Digit-Recognition/
│
├── mon_exo.ipynb           # Notebook principal avec tout le code
├── train.csv               # Dataset d'entraînement (à télécharger)
├── test.csv                # Dataset de test (à télécharger)
├── submission.csv          # Fichier de prédictions (généré)
├── best_model.pkl          # Meilleur modèle sauvegardé (à télécharger)
├── requirements.txt        # Dépendances Python
├── .gitignore             # Fichiers à ignorer
├── LICENSE                # Licence MIT
└── README.md              # Ce fichier
```

##  Méthodologie

### 1. Préparation des Données
- Chargement des datasets train et test
- Séparation features (X) et labels (y)
- Normalisation des pixels (0-255 → 0-1)
- Split train/validation (80/20)

### 2. Modèles Testés

#### Decision Tree
```python
DecisionTreeClassifier(max_depth=20, random_state=42)
```

#### Random Forest
```python
RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
```

#### Support Vector Machine (SVM)
```python
SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
```

### 3. Optimisation
- **GridSearchCV** pour l'optimisation des hyperparamètres
- Cross-validation pour éviter l'overfitting
- Comparaison des métriques de performance

### 4. Évaluation
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Visualisation des erreurs de classification

##  Résultats

Les performances des modèles sont comparées selon plusieurs métriques :

- **Accuracy** : Taux de classification correcte
- **Precision** : Exactitude des prédictions positives
- **Recall** : Capacité à identifier tous les cas positifs
- **F1-Score** : Moyenne harmonique de Precision et Recall

*Les résultats détaillés et visualisations sont disponibles dans le notebook.*

##  Utilisation

### Exécuter le Notebook

1. **Lancer Jupyter**
```bash
jupyter notebook mon_exo.ipynb
```

2. **Exécuter les cellules dans l'ordre**
   - Imports et configuration
   - Chargement des données
   - Exploration et visualisation
   - Entraînement des modèles
   - Évaluation et comparaison
   - Prédictions sur test set
   - Génération du fichier submission

### Utiliser le Modèle Sauvegardé

```python
import pickle
import numpy as np

# Charger le modèle
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Faire une prédiction
# image doit être un array de 784 pixels normalisés
prediction = model.predict(image.reshape(1, -1))
print(f"Chiffre prédit : {prediction[0]}")
```

##  Visualisations

Le projet inclut plusieurs types de visualisations :

- Distribution des classes dans le dataset
- Exemples d'images pour chaque chiffre
- Matrices de confusion
- Comparaison des performances des modèles
- Exemples de prédictions correctes et incorrectes
- Prédictions sur le test set

##  Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez votre branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

##  Améliorations Futures

- [ ] Implémenter des réseaux de neurones (CNN)
- [ ] Augmentation de données (data augmentation)
- [ ] Déploiement d'une API REST
- [ ] Interface web pour tester les prédictions
- [ ] Optimisation des performances
- [ ] Tests unitaires

##  Auteur

**Adja Kimy Fatima**  
Passionnée de Data Science & Deep Learning

- 🌐 GitHub : [@Adjakim](https://github.com/Adjakim)
- 📧 Email : adjakimfatima@gmail.com
- 💼 LinkedIn : [Adja Kimy Fatima](https://linkedin.com/in/adjakim)

**Parcours :**
- 🎓 Formation en Data, IA et DEV (2025-2026)

##  License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

##  Remerciements

- Dataset MNIST pour les données
- Communauté Scikit-learn pour les outils de ML
- Kaggle pour l'inspiration et les ressources

---

**Dernière mise à jour** : Décembre 2025

⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !