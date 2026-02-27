import matplotlib.pyplot as plt
import numpy as np
import os
from tabulation_module import FPVTabulation

def run_post_process(table_path='output/fpv_table.npz'):
    if not os.path.exists(table_path):
        print(f"Error: {table_path} not found. Run main.py first.")
        return

    # Load the tabulation
    print(f"Loading tabulation from {table_path}...")
    tabulation = FPVTabulation.load(table_path)
    
    # 1. Visualization of the 2D table (from main.py)
    plt.figure(figsize=(12, 5))
    Z_mesh, C_mesh = np.meshgrid(tabulation.grid_z, tabulation.c_uniform)
    
    plt.subplot(1, 2, 1)
    plt.pcolormesh(Z_mesh, C_mesh, tabulation.T_table.T, shading='auto')
    plt.colorbar(label='Temperature (K)')
    plt.xlabel('Mixture Fraction Z')
    plt.ylabel('Normalized PV c')
    plt.title('FPV Table: Temperature')
    
    plt.subplot(1, 2, 2)
    plt.pcolormesh(Z_mesh, C_mesh, np.log10(np.abs(tabulation.OmegaC_table.T) + 1e-10), shading='auto')
    plt.colorbar(label='log10(|Omega_C|)')
    plt.xlabel('Mixture Fraction Z')
    plt.ylabel('Normalized PV c')
    plt.title('FPV Table: Omega_C')
    
    plt.tight_layout()
    plt.savefig('output/fpv_table_visualization.png')
    print("Saved 2D table visualization to output/fpv_table_visualization.png")
    
    # 2. Plot Omega_C vs Z for specific normalized progress variables (c_norm)
    # Specified by the user: "specify a particular C... plot omegaDot_C as a function of Z"
    plt.figure(figsize=(10, 6))
    z_grid = np.linspace(0, 1, 200)
    c_norm_values = [0.8, 0.9, 0.95] # For example
    
    for c_norm in c_norm_values:
        omega_c_vals = []
        for z in z_grid:
            _, _, omega_c = tabulation.lookup(z, c_norm)
            omega_c_vals.append(omega_c)
        
        plt.plot(z_grid, omega_c_vals, label=f'c_norm = {c_norm}')
    
    plt.xlabel('Mixture Fraction Z')
    plt.ylabel('Source Term Omega_C (kg/m3/s)')
    plt.title('Reaction Progress Rate vs Z for constant c_norm')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('output/omega_c_vs_z_post.png')
    print("Saved 1D plot to output/omega_c_vs_z_post.png")

if __name__ == "__main__":
    run_post_process()
