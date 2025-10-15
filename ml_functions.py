import numpy as np

# Dictionnaire des fonctions d'activation et de leurs dérivées
ACTIVATION_FUNCTIONS = {
    # Fonctions standards
    'identity': (lambda x: x, lambda x: np.ones_like(x)),
    'sigmoid': (lambda x: 1 / (1 + np.exp(-x)), lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))),
    'tanh': (lambda x: np.tanh(x), lambda x: 1 - np.tanh(x)**2),
    'relu': (lambda x: np.maximum(0, x), lambda x: np.where(x > 0, 1, 0)),
    'leaky_relu': (lambda x, alpha=0.01: np.where(x > 0, x, x * alpha), lambda x, alpha=0.01: np.where(x > 0, 1, alpha)),
    'softmax': (lambda x: np.exp(x) / np.sum(np.exp(x), axis=0), lambda x: x),  # La dérivée est plus complexe, ceci est un placeholder

    # Variantes de ReLU
    'elu': (lambda x, alpha=1.0: np.where(x > 0, x, alpha * (np.exp(x) - 1)), lambda x, alpha=1.0: np.where(x > 0, 1, alpha * np.exp(x))),
    'selu': (lambda x: 1.0507 * np.where(x > 0, x, 1.67326 * (np.exp(x) - 1)), lambda x: 1.0507 * np.where(x > 0, 1, 1.67326 * np.exp(x))),
    'prelu': (lambda x, alpha=0.1: np.where(x > 0, x, alpha * x), lambda x, alpha=0.1: np.where(x > 0, 1, alpha)),
    'crelu': (lambda x: np.concatenate((np.maximum(0, x), np.maximum(0, -x)), axis=0), lambda x: np.concatenate((np.where(x > 0, 1, 0), np.where(-x > 0, -1, 0)), axis=0)),
    'relu6': (lambda x: np.minimum(np.maximum(0, x), 6), lambda x: np.where((x > 0) & (x < 6), 1, 0)),
    
    # Fonctions sinusoïdales et périodiques
    'sin': (lambda x: np.sin(x), lambda x: np.cos(x)),
    'cos': (lambda x: np.cos(x), lambda x: -np.sin(x)),
    'sinc': (lambda x: np.sinc(x), lambda x: (np.cos(np.pi * x) * np.pi * x - np.sin(np.pi * x)) / (np.pi * x**2)),
    
    # Fonctions quadratiques et polynomiales
    'quadratic': (lambda x: x**2, lambda x: 2*x),
    'cubic': (lambda x: x**3, lambda x: 3*x**2),
    
    # Fonctions de type "Swish"
    'swish': (lambda x: x * (1 / (1 + np.exp(-x))), lambda x: (1 / (1 + np.exp(-x))) + x * ((1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x)))))),
    'hard_swish': (lambda x: x * np.maximum(0, np.minimum(1, (x + 3) / 6)), lambda x: np.where(x < -3, 0, np.where(x > 3, 1, (2*x + 3)/6))),

    # Fonctions de type "Gaussian"
    'gaussian': (lambda x: np.exp(-x**2), lambda x: -2 * x * np.exp(-x**2)),
    'gelu': (lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))), lambda x: 0.5 * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715 * x**3))) + 0.5 * x * (1 - np.tanh(np.sqrt(2/np.pi)*(x + 0.044715 * x**3))**2) * np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2)),

    # Fonctions diverses
    'binary_step': (lambda x: np.where(x >= 0, 1, 0), lambda x: np.zeros_like(x)),
    'softplus': (lambda x: np.log(1 + np.exp(x)), lambda x: 1 / (1 + np.exp(-x))),
    'softsign': (lambda x: x / (1 + np.abs(x)), lambda x: 1 / (1 + np.abs(x))**2),
    'bent_identity': (lambda x: (np.sqrt(x**2 + 1) - 1) / 2 + x, lambda x: x / (2 * np.sqrt(x**2 + 1)) + 1),
    'mish': (lambda x: x * np.tanh(np.log(1 + np.exp(x))), lambda x: (np.exp(x) * (4*(x+1) + 4*np.exp(2*x) + np.exp(3*x) + np.exp(x)*(4*x+6))) / (2*np.exp(x) + np.exp(2*x) + 2)**2),
    'hard_tanh': (lambda x: np.maximum(-1, np.minimum(1, x)), lambda x: np.where((x > -1) & (x < 1), 1, 0)),
    'thresholded_relu': (lambda x, theta=1.0: np.where(x > theta, x, 0), lambda x, theta=1.0: np.where(x > theta, 1, 0)),
    'srelu': (
        lambda x, a_l=0.1, t_l=-1.0, a_r=0.1, t_r=1.0: np.where(x <= t_l, t_l + a_l * (x - t_l), np.where(x >= t_r, t_r + a_r * (x - t_r), x)),
        lambda x, a_l=0.1, t_l=-1.0, a_r=0.1, t_r=1.0: np.where(x <= t_l, a_l, np.where(x >= t_r, a_r, 1))
    ),
    'lisht': (lambda x: x * np.tanh(x), lambda x: np.tanh(x) + x * (1 - np.tanh(x)**2)),
    'silu': (lambda x: x / (1 + np.exp(-x)), lambda x: (1 + np.exp(-x) + x * np.exp(-x)) / (1 + np.exp(-x))**2),
    'isru': (lambda x, alpha=1.0: x / np.sqrt(1 + alpha * x**2), lambda x, alpha=1.0: 1 / (1 + alpha * x**2)**(3/2)),
    'isrlu': (lambda x, alpha=1.0: np.where(x < 0, x / np.sqrt(1 + alpha * x**2), x), lambda x, alpha=1.0: np.where(x < 0, 1 / (1 + alpha * x**2)**(3/2), 1)),
    'soft_clipping': (lambda x, alpha=0.5: (1/alpha) * np.log((1 + np.exp(alpha * x)) / (1 + np.exp(alpha * (x - 1)))), lambda x, alpha=0.5: 1 / (1 + np.exp(-alpha * x)) - 1 / (1 + np.exp(-alpha * (x-1)))),
    'sqnl': (
        lambda x: np.where(x > 2, 1, np.where(x >= 0, x - x**2/4, np.where(x >= -2, x + x**2/4, -1))),
        lambda x: np.where(x > 2, 0, np.where(x >= 0, 1 - x/2, np.where(x >= -2, 1 + x/2, 0)))
    ),
    'inverse_sqrt_relu': (lambda x: np.where(x > 0, x / np.sqrt(1 + 1.0 * x**2), 0.1 * x), lambda x: np.where(x > 0, (1 + 1.0 * x**2)**(-3/2), 0.1)),
    'arc_tan': (lambda x: np.arctan(x), lambda x: 1 / (1 + x**2)),
    'arc_sinh': (lambda x: np.arcsinh(x), lambda x: 1 / np.sqrt(x**2 + 1)),
    'elliott_symmetric': (lambda x: x / (1 + np.abs(x)), lambda x: 1 / (1 + np.abs(x))**2),
    'bipolar_sigmoid': (lambda x: (1 - np.exp(-x)) / (1 + np.exp(-x)), lambda x: 2 * np.exp(-x) / (1 + np.exp(-x))**2),
}

def get_function(name):
    """
    Récupère une fonction et sa dérivée par leur nom.
    """
    return ACTIVATION_FUNCTIONS.get(name, (None, None))

def get_all_function_names():
    """
    Retourne une liste de tous les noms de fonctions disponibles.
    """
    return sorted(list(ACTIVATION_FUNCTIONS.keys()))

def evaluate_custom_function(func_string):
    """
    Évalue une chaîne de caractères en tant que fonction de manière sécurisée.
    """
    import ast

    allowed_nodes = [
        'Expression', 'BinOp', 'UnaryOp', 'Call', 'Name', 'Load', 'Store', 
        'Add', 'Sub', 'Mult', 'Div', 'Pow', 'USub', 'UAdd', 'Constant'
    ]

    # Fonctions numpy autorisées
    allowed_functions = [
        'sin', 'cos', 'tan', 'sqrt', 'exp', 'log', 'abs', 'sinh', 'cosh', 'tanh'
    ]

    tree = ast.parse(func_string, mode='eval')

    for node in ast.walk(tree):
        node_type = type(node).__name__
        if node_type not in allowed_nodes:
            raise ValueError(f"Opération non autorisée: {node_type}")
        
        if isinstance(node, ast.Name) and node.id not in ['x', 'y', 'np'] and node.id not in allowed_functions:
            raise ValueError(f"Variable ou fonction non autorisée: {node.id}")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in allowed_functions:
            if isinstance(node.func, ast.Attribute) and node.func.value.id == 'np':
                if node.func.attr not in allowed_functions:
                    raise ValueError(f"Fonction numpy non autorisée: {node.func.attr}")
            else:
                raise ValueError(f"Fonction non autorisée: {node.func.id}")


    # Compilation et évaluation
    code = compile(tree, '<string>', 'eval')

    def func(x, y=None):
        # L'environnement d'évaluation
        eval_globals = {'np': np}
        
        # Détermine si la fonction est 1D ou 2D en fonction des variables utilisées
        if 'y' in func_string:
            return eval(code, eval_globals, {'x': x, 'y': y})
        else:
            return eval(code, eval_globals, {'x': x})

    return func