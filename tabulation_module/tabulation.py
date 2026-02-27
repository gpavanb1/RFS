import numpy as np
from scipy.interpolate import interp1d

from .progress_variable import PV_H2O

class FPVTabulation:
    def __init__(self, problem=None, pv_definition=PV_H2O):
        """
        problem: FlameletProblem instance after continuation
        pv_definition: ProgressVariable instance
        """
        if problem is not None:
            self.problem = problem
            self.pv_definition = pv_definition
            self.grid_z = problem.grid
            self.num_solutions = len(problem.solutions)
            self.neq = problem.num_equations
            self.species_names = list(problem.species_names)
            
            # Mapping for progress variable calculation
            self.species_map = {name: i for i, name in enumerate(self.species_names)}
            
            # Precompute full profiles for each solution
            self.all_T = []
            self.all_Y = []
            self.all_C = []
            self.all_omega_C = []
            
            # Get molecular weights
            mw = problem.flamelet0.mechanism.molecular_weights
            gas = problem.flamelet0.mechanism.gas
            pressure = problem.p
            
            # 1. Add unburned solution (mixing line)
            # This ensures C_min is correctly identified as the unburned state
            T_unburned = problem.air_T + problem.grid * (problem.fuel_T - problem.air_T)
            Y_unburned = np.zeros((self.neq, len(problem.grid)))
            for i in range(self.neq):
                Y_unburned[i, :] = problem.air_y[i] + problem.grid * (problem.fuel_y[i] - problem.air_y[i])
            
            C_unburned = self.pv_definition.compute(self.species_map, Y_unburned)
            omega_C_unburned = np.zeros_like(C_unburned)
            
            self.all_T.append(T_unburned)
            self.all_Y.append(Y_unburned)
            self.all_C.append(C_unburned)
            self.all_omega_C.append(omega_C_unburned)

            # 2. Add solved flamelet solutions
            for sol in problem.solutions:
                T_profile = np.hstack((problem.air_T, sol[::self.neq], problem.fuel_T))
                Y_profiles = []
                
                # Map of species index to its full profile
                species_profiles_dict = {}
                for i in range(self.neq - 1):
                    Yi = np.hstack((problem.air_y[i], sol[i+1::self.neq], problem.fuel_y[i]))
                    species_profiles_dict[i] = Yi
                    Y_profiles.append(Yi)
                
                sum_Y = sum(species_profiles_dict.values())
                Y_last = 1.0 - sum_Y
                species_profiles_dict[self.neq - 1] = Y_last
                Y_profiles.append(Y_last)
                
                Y_profiles_arr = np.array(Y_profiles) # (n_species, n_grid)
                
                # Compute PV profile
                C_profile = self.pv_definition.compute(self.species_map, Y_profiles_arr)
                
                # Compute Omega_C profile
                omega_C_profile = np.zeros_like(C_profile)
                for j in range(len(T_profile)):
                    gas.TPY = T_profile[j], pressure, Y_profiles_arr[:, j]
                    # net_production_rates in kmol/m3/s
                    w_dot_species = gas.net_production_rates * mw # kg/m3/s
                    
                    # omega_C = sum(w_i * w_dot_i)
                    omega_C_val = 0.0
                    for s_name, weight in zip(self.pv_definition.species_list, self.pv_definition.weights):
                        s_idx = self.species_map[s_name]
                        omega_C_val += weight * w_dot_species[s_idx]
                    omega_C_profile[j] = omega_C_val
                    
                self.all_T.append(T_profile)
                self.all_Y.append(Y_profiles_arr)
                self.all_C.append(C_profile)
                self.all_omega_C.append(omega_C_profile)
                
            self.all_T = np.array(self.all_T)
            self.all_Y = np.array(self.all_Y)
            self.all_C = np.array(self.all_C)
            self.all_omega_C = np.array(self.all_omega_C)

    @classmethod
    def load(cls, filename='output/fpv_table.npz'):
        data = np.load(filename)
        obj = cls()
        obj.grid_z = data['grid_z']
        obj.c_uniform = data['c_uniform']
        obj.T_table = data['T_table']
        obj.Y_table = data['Y_table']
        obj.OmegaC_table = data['OmegaC_table']
        obj.C_max = data['C_max']
        obj.C_min = data['C_min']
        obj.species_names = data['species_names']
        obj.neq = obj.Y_table.shape[2]
        return obj
        
    def build_table(self, n_c=100):
        """
        Build the true FPV table by normalizing and interpolating onto a uniform c-grid.
        n_c: Number of points in the normalized PV dimension.
        """
        n_z = len(self.grid_z)
        self.n_c = n_c
        self.c_uniform = np.linspace(0, 1, n_c)
        
        # Initialize tables: [n_z, n_c]
        self.T_table = np.zeros((n_z, n_c))
        self.OmegaC_table = np.zeros((n_z, n_c))
        # [n_z, n_c, n_species]
        self.Y_table = np.zeros((n_z, n_c, self.neq))
        
        # Step 3: Determine C_max(Z) and C_min(Z)
        # We take the maximum and minimum across all solutions at each Z.
        # This assumes we have both the fully reacted (stable) and unburned branches.
        self.C_max = np.max(self.all_C, axis=0) # (n_z,)
        self.C_min = np.min(self.all_C, axis=0) # (n_z,)
        
        # Step 4 & 5: Normalize and interpolate
        for k in range(n_z):
            # For each Z grid point k
            # Collect all values of C, T, Y, OmegaC from all solutions at this Z
            Ck = self.all_C[:, k]
            Tk = self.all_T[:, k]
            Yk = self.all_Y[:, :, k] # (n_sol, n_species)
            OCk = self.all_omega_C[:, k]
            
            # Normalize to get c \in [0, 1]
            denom = self.C_max[k] - self.C_min[k]
            if denom > 1e-12:
                ck = (Ck - self.C_min[k]) / denom
            else:
                ck = np.zeros_like(Ck)
            
            # Step 5: Collapse chi dimension
            # Sort by c to ensure monotonicity for interpolation
            sort_idx = np.argsort(ck)
            ck_sorted = ck[sort_idx]
            Tk_sorted = Tk[sort_idx]
            Yk_sorted = Yk[sort_idx]
            OCk_sorted = OCk[sort_idx]
            
            # Remove duplicate or non-monotonic regions
            # (Important if the S-curve has regions where C is not strictly increasing)
            mask = np.ones(len(ck_sorted), dtype=bool)
            for i in range(1, len(ck_sorted)):
                if ck_sorted[i] <= ck_sorted[i-1]:
                    mask[i] = False
            
            ck_final = ck_sorted[mask]
            Tk_final = Tk_sorted[mask]
            Yk_final = Yk_sorted[mask]
            OCk_final = OCk_sorted[mask]
            
            # Interpolate onto the uniform c-grid
            if len(ck_final) > 1:
                self.T_table[k, :] = np.interp(self.c_uniform, ck_final, Tk_final)
                self.OmegaC_table[k, :] = np.interp(self.c_uniform, ck_final, OCk_final)
                for j in range(self.neq):
                    self.Y_table[k, :, j] = np.interp(self.c_uniform, ck_final, Yk_final[:, j])
            else:
                # Fallback for Z=0 or Z=1 where all solutions are identical
                self.T_table[k, :] = Tk[0]
                self.OmegaC_table[k, :] = OCk[0]
                self.Y_table[k, :, :] = Yk[0, :]

    def lookup(self, z, c_norm):
        """
        Lookup state at given mixture fraction Z and normalized PV c_norm.
        z: scalar (0 to 1)
        c_norm: scalar (0 to 1)
        """
        # Interpolate in Z dimension first to get profiles over c_norm
        # Then interpolate in c_norm
        
        # Finding the Z indices for interpolation
        z_idx = np.searchsorted(self.grid_z, z)
        if z_idx == 0:
            return self._interpolate_c(0, c_norm)
        if z_idx == len(self.grid_z):
            return self._interpolate_c(len(self.grid_z) - 1, c_norm)
        
        # Interpolate between z_idx-1 and z_idx
        z0, z1 = self.grid_z[z_idx-1], self.grid_z[z_idx]
        frac = (z - z0) / (z1 - z0)
        
        res0 = self._interpolate_c(z_idx-1, c_norm)
        res1 = self._interpolate_c(z_idx, c_norm)
        
        T = res0[0] + frac * (res1[0] - res0[0])
        Y = res0[1] + frac * (res1[1] - res0[1])
        OmegaC = res0[2] + frac * (res1[2] - res0[2])
        
        return T, Y, OmegaC

    def _interpolate_c(self, z_idx, c_norm):
        """Helper to interpolate in the c dimension at a fixed Z index."""
        T = np.interp(c_norm, self.c_uniform, self.T_table[z_idx, :])
        Y = np.zeros(self.neq)
        for j in range(self.neq):
            Y[j] = np.interp(c_norm, self.c_uniform, self.Y_table[z_idx, :, j])
        OmegaC = np.interp(c_norm, self.c_uniform, self.OmegaC_table[z_idx, :])
        return T, Y, OmegaC

    def save_to_file(self, filename='output/fpv_table.npz'):
        """
        Save the tabulation data for future use.
        """
        save_dict = {
            'grid_z': self.grid_z,
            'c_uniform': self.c_uniform,
            'T_table': self.T_table,
            'Y_table': self.Y_table,
            'OmegaC_table': self.OmegaC_table,
            'C_max': self.C_max,
            'C_min': self.C_min,
            'species_names': self.species_names,
            'pv_species': self.pv_definition.species_list,
            'pv_weights': self.pv_definition.weights
        }
        np.savez(filename, **save_dict)
        print(f'Saved proper FPV table to {filename}')
