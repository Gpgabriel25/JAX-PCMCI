
import jax
import jax.numpy as jnp
import numpy as np
from jax_pcmci import PCMCI, ParCorr, DataHandler
import pickle
import os

def run_test():
    # Use a fixed seed for reproducibility
    key = jax.random.PRNGKey(42)
    T = 200
    N = 4
    tau_max = 2
    
    # Generate data
    data = jax.random.normal(key, (T, N))
    handler = DataHandler(data)
    
    pc_alpha = 0.05
    
    # Run PCMCI (Standard)
    pcmci = PCMCI(handler, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run(tau_max=tau_max, pc_alpha=pc_alpha)
    
    return {
        "parents": pcmci._parents,
        "pvals": np.array(results.pval_matrix),
        "vals": np.array(results.val_matrix)
    }

def main():
    baseline_file = "baseline_results.pkl"
    
    current_results = run_test()
    
    if os.path.exists(baseline_file):
        print("Loading baseline...")
        with open(baseline_file, "rb") as f:
            baseline = pickle.load(f)
            
        # Compare
        print("Comparing results...")
        parents_match = (str(baseline["parents"]) == str(current_results["parents"]))
        pvals_close = np.allclose(baseline["pvals"], current_results["pvals"], equal_nan=True)
        vals_close = np.allclose(baseline["vals"], current_results["vals"], equal_nan=True)
        
        print(f"Parents Match: {parents_match}")
        print(f"P-values Match: {pvals_close}")
        print(f"Values Match: {vals_close}")
        
        if not (parents_match and pvals_close and vals_close):
             print("FAILURE: mismatch detected!")
             # exit(1) # Don't exit yet during dev
        else:
             print("SUCCESS: Results match baseline.")
    else:
        print("Saving baseline...")
        with open(baseline_file, "wb") as f:
            pickle.dump(current_results, f)
        print("Baseline saved.")

if __name__ == "__main__":
    main()
