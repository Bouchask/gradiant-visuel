
from flask import Flask, render_template, send_file, request, abort
import io
import numpy as np

from ml_functions import get_function, get_all_function_names, evaluate_custom_function
from plotting import plot_2d_function, plot_3d_function, plot_custom_function_2d, plot_custom_function_3d

app = Flask(__name__)

@app.route('/')
def index():
    """Affiche la page d'accueil avec la liste des fonctions."""
    function_names = get_all_function_names()
    return render_template('index.html', function_names=function_names)

@app.route('/plot/predefined')
def plot_predefined():
    """Génère un graphique pour une fonction prédéfinie."""
    function_name = request.args.get('function_name')
    plot_type = request.args.get('plot_type', '2d')

    func, grad = get_function(function_name)
    if func is None:
        return abort(404, description="Fonction non trouvée")

    if plot_type == '2d':
        if grad is None:
            return abort(400, description="Dérivée non disponible pour cette fonction.")
        buf = plot_2d_function(func, grad, function_name)
    elif plot_type == '3d':
        buf = plot_3d_function(func, function_name, is_2d=False)
    else:
        return abort(400, description="Type de graphique non valide")

    return send_file(buf, mimetype='image/png')

@app.route('/plot/custom')
def plot_custom():
    """Génère un graphique pour une fonction personnalisée."""
    func_string = request.args.get('custom_function')
    plot_type = request.args.get('plot_type', '2d')

    if not func_string:
        return abort(400, description="Aucune fonction fournie.")

    try:
        custom_func = evaluate_custom_function(func_string)
    except (ValueError, SyntaxError) as e:
        return abort(400, description=f"Erreur dans l'évaluation de la fonction: {e}")

    if plot_type == '2d':
        buf = plot_custom_function_2d(custom_func, func_string)
    elif plot_type == '3d':
        is_2d = 'y' in func_string
        if is_2d:
            buf = plot_custom_function_3d(custom_func, func_string)
        else:
            # Si la fonction est 1D, nous devons l'adapter pour un tracé 3D
            func_3d = lambda x, y: custom_func(np.sqrt(x**2 + y**2))
            buf = plot_custom_function_3d(func_3d, func_string)
    else:
        return abort(400, description="Type de graphique non valide")

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
