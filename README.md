# Explorateur de Fonctions Machine Learning

Cet outil web permet de visualiser diverses fonctions d'activation utilisées en Machine Learning, ainsi que de tracer des fonctions mathématiques personnalisées en 2D et 3D.

## Fonctionnalités

*   **Visualisation de fonctions prédéfinies** : Explorez une vaste bibliothèque de fonctions d'activation (Sigmoid, ReLU, Tanh, etc.) et visualisez leur forme ainsi que leur dérivée.
*   **Créateur de fonctions personnalisées** : Définissez et tracez vos propres fonctions mathématiques en utilisant la syntaxe Python et NumPy.
*   **Graphiques 2D et 3D** : Affichez les fonctions en 2D (tracé de la fonction et de sa dérivée) ou en 3D (surface de la fonction).
*   **Interface web interactive** : Une interface utilisateur simple et intuitive pour une exploration fluide.

## Exemples de Fonctions d'Optimisation

Voici quelques exemples des fonctions que vous pouvez visualiser avec cet outil :

*   **Sigmoid**: `1 / (1 + np.exp(-x))` - Une fonction lisse qui mappe les valeurs en entrée dans l'intervalle (0, 1).
*   **Tanh**: `np.tanh(x)` - Similaire à la sigmoïde mais mappe les valeurs dans l'intervalle (-1, 1).
*   **ReLU (Rectified Linear Unit)**: `np.maximum(0, x)` - Une fonction linéaire par morceaux qui renvoie l'entrée si elle est positive, sinon elle renvoie zéro.
*   **Leaky ReLU**: `np.where(x > 0, x, x * 0.01)` - Une variante de ReLU qui permet une petite pente pour les valeurs négatives.
*   **Gaussian**: `np.exp(-x**2)` - La fonction gaussienne, qui forme une courbe en forme de cloche.
*   **Sinc**: `np.sinc(x)` - La fonction sinus cardinal, utile dans le traitement du signal.

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

## Auteur

Yahya Bouchak