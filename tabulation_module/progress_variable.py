import numpy as np

class ProgressVariable:
    def __init__(self, species_list=['H2O'], weights=None):
        """
        species_list: list of species names
        weights: optional list of weights for each species. If None, weights are 1.0.
        """
        self.species_list = species_list
        if weights is None:
            self.weights = [1.0] * len(species_list)
        else:
            self.weights = weights
            
    def compute(self, species_map, sol_profiles):
        """
        Compute PV profile from a set of species profiles.
        species_map: dict mapping species names to indices in sol_profiles
        sol_profiles: array of species profiles (N_species, N_grid)
        """
        C = np.zeros(sol_profiles.shape[1])
        for s, w in zip(self.species_list, self.weights):
            idx = species_map[s]
            C += w * sol_profiles[idx]
        return C

# Pre-defined PV choices
PV_H2O = ProgressVariable(['H2O'])
PV_H2_H2O = ProgressVariable(['H2', 'H2O'], weights=[-1.0, 1.0])
PV_CO_CO2_H2_H2O = ProgressVariable(['CO', 'CO2', 'H2', 'H2O'], weights=[1.0, 1.0, 1.0, 1.0])
