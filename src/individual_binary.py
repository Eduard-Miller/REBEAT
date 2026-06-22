import numpy as np

from individual import Individual

class Individual_binary(Individual):    
    def __init__(self, genome: np.ndarray):
        self.genome = genome
        self.fitness = None
        self.fenotype = None
        
    def update_individual(self, genome: np.ndarray=None, fenotype: np.ndarray=None, fitness: float=None):
        if genome is not None:
            self._update_genome(genome)
        if fenotype is not None:
            self._update_fenotype(fenotype)
        if fitness is not None:
            self._update_fitness(fitness)
            
    def _update_fenotype(self, fenotype: np.ndarray): # calculated externally
        self.fenotype = fenotype
        
    def _update_fitness(self, fitness: float): # calculated externally
        self.fitness = fitness
        
    def _update_genome(self, genome: np.ndarray): # calculated externally
        self.genome = genome