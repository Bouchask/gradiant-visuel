import numpy as np
import matplotlib.pyplot as plt
import io
from mpl_toolkits.mplot3d import Axes3D

def _save_plot_to_buffer():
    """Sauvegarde le graphique actuel dans un buffer mémoire."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def plot_2d_function(func, grad, func_name):
    """
    Crée un graphique 2D pour une fonction et sa dérivée.
    """
    x = np.linspace(-5, 5, 400)
    y = func(x)
    y_grad = grad(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Fonction: {func_name}(x)', color='blue')
    plt.plot(x, y_grad, label=f"Dérivée: {func_name}'(x)", color='red', linestyle='--')
    
    plt.title(f'Fonction d\'activation: {func_name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.ylim(-2, 2) # Limite l'axe y pour une meilleure lisibilité
    
    return _save_plot_to_buffer()

def plot_3d_function(func, func_name, is_2d=False):
    """
    Crée un graphique de surface 3D pour une fonction.
    """
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    if is_2d:
        Z = func(X, Y)
    else:
        Z = func(np.sqrt(X**2 + Y**2))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    ax.set_title(f'Surface 3D pour: {func_name}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return _save_plot_to_buffer()

def plot_custom_function_2d(func, func_string):
    """
    Crée un graphique 2D pour une fonction personnalisée.
    """
    x = np.linspace(-10, 10, 400)
    y = func(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'f(x) = {func_string}', color='purple')
    
    plt.title('Fonction Personnalisée')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    
    return _save_plot_to_buffer()

def plot_custom_function_3d(func, func_string):
    """
    Crée un graphique 3D pour une fonction personnalisée.
    """
    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = func(X, Y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_title(f'Surface 3D pour f(x, y) = {func_string}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return _save_plot_to_buffer()