# Explorateur de Fonctions Machine Learning

Cet outil web permet de visualiser diverses fonctions d'activation utilisées en Machine Learning, ainsi que de tracer des fonctions mathématiques personnalisées en 2D et 3D.

## Fonctionnalités

*   **Visualisation de fonctions prédéfinies** : Explorez une vaste bibliothèque de fonctions d'activation (Sigmoid, ReLU, Tanh, etc.) et visualisez leur forme ainsi que leur dérivée.
*   **Créateur de fonctions personnalisées** : Définissez et tracez vos propres fonctions mathématiques en utilisant la syntaxe Python et NumPy.
*   **Graphiques 2D et 3D** : Affichez les fonctions en 2D (tracé de la fonction et de sa dérivée) ou en 3D (surface de la fonction).
*   **Interface web interactive** : Une interface utilisateur simple et intuitive pour une exploration fluide.

## Structure du Projet

*   `app.py`: L'application web principale basée sur Flask.
*   `ml_functions.py`: Contient la logique pour les fonctions d'activation et l'évaluation des fonctions personnalisées.
*   `plotting.py`: Gère la génération des graphiques avec Matplotlib.
*   `templates/index.html`: La page web principale de l'application.
*   `static/style.css`: La feuille de style pour la page web.

## Comment l'utiliser

1.  **Installer les dépendances**:
    ```bash
    pip install Flask numpy matplotlib
    ```

2.  **Lancer l'application**:
    ```bash
    python app.py
    ```

3.  **Ouvrir dans le navigateur**:
    Accédez à `http://127.0.0.1:5000` dans votre navigateur web.

## Captures d'écran

*(Insérez ici des captures d'écran de l'application si vous le souhaitez)*

## Auteur

Yahya Bouchak
