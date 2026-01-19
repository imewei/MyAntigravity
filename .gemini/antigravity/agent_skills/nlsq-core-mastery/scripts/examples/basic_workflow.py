import numpy as np
import jax.numpy as jnp
from nlsq import fit, CurveFit
from nlsq.global_optimization import CMAESConfig

def exponential_decay(x, params):
    """
    Model: A * exp(-b * x) + c
    Params: [A, b, c]
    """
    A, b, c = params
    return A * jnp.exp(-b * x) + c

def run_demonstration():
    # 1. Generate synthetic data
    x = np.linspace(0, 10, 1000)
    # True params: A=5.0, b=0.5, c=1.0
    y_true = exponential_decay(x, [5.0, 0.5, 1.0])
    y_noise = y_true + 0.1 * np.random.normal(size=len(x))
    
    # 2. Define bounds (REQUIRED for auto_global)
    # Lower: [0, 0, 0], Upper: [10, 5, 5]
    bounds = ([0.0, 0.0, 0.0], [10.0, 5.0, 5.0])
    
    print("\n--- Running 'auto' workflow (Local Optimization) ---")
    # Uses smart memory management automatically
    result_local = fit(
        exponential_decay, x, y_noise, 
        p0=[4.0, 0.4, 0.8],  # Good initial guess
        workflow="auto",
        show_progress=True
    )
    print(f"Local fit result: {result_local.params}")

    print("\n--- Running 'auto_global' workflow ---")
    # Automatically selects Multi-Start or CMA-ES based on scale
    # Bad initial guess, robust global search needed
    result_global = fit(
        exponential_decay, x, y_noise,
        p0=[1.0, 1.0, 1.0], 
        bounds=bounds,
        workflow="auto_global",
        show_progress=True
    )
    print(f"Global fit result: {result_global.params}")

if __name__ == "__main__":
    run_demonstration()
