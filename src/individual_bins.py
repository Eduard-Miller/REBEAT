import numpy as np

from individual import Individual

class Individual_bins(Individual):
    max_bin_lenght: int
    genome_bin: np.ndarray # binary representation of the genome (1d array np.bool)
    
    def __init__(self, genome: list[tuple[int, int]], max_bin_lenght: int):
        self.max_bin_lenght = max_bin_lenght
        self.genome = genome
        self._update_genome_bin()
        self.fitness = None
        self.shared_fitness = None
        self.fenotype = None
        
    def update_individual(self, genome: list[tuple[int, int]]=None, fenotype: np.ndarray=None, fitness: float=None, shared_fitness: float=None):
        if genome is not None:
            self._update_genome(genome)
            self._update_genome_bin()
        if fenotype is not None:
            self._update_fenotype(fenotype)
        if fitness is not None:
            self._update_fitness(fitness)
        if shared_fitness is not None:
            self._update_shared_fitness(shared_fitness)
            
    def _update_genome_bin(self):
        '''
        Converts the genome (list of tuples) into a binary representation (1d numpy array of bools). This allows for efficient similarity calculations using bitwise operations.
        '''
        binary = np.zeros(self.max_bin_lenght, dtype=np.bool)
        
        for start, length in self.genome:
            end = start + length
            # Handle both positive and negative lengths
            idx_start, idx_end = (start, end) if length >= 0 else (end, start)
            
            # Clip to valid range
            idx_start = max(0, min(self.max_bin_lenght, idx_start))
            idx_end = max(0, min(self.max_bin_lenght, idx_end))
            
            binary[int(idx_start):int(idx_end)] = True
            
        self.genome_bin = binary
            
    def _update_fenotype(self, fenotype: np.ndarray): # calculated externally
        self.fenotype = fenotype
        
    def _update_fitness(self, fitness: float): # calculated externally
        self.fitness = fitness
    
    def _update_shared_fitness(self, shared_fitness: float): # calculated externally
        self.shared_fitness = shared_fitness
        
    def _update_genome(self, genome: list[tuple[int, int]]): # calculated externally
        self.genome = genome
        
        
def binary_to_bins(genome_bin: np.ndarray, max_bin_count: int) -> list[tuple[int, int]]:
    """
    Converts the binary representation of the genome back into a list of tuples (index and length).
    """
    padded = np.concatenate(([False], genome_bin, [False]))
    
    diff = np.diff(padded.astype(np.int8))
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    bins = list(zip(starts.tolist(), (ends - starts).tolist()))
    
    # should only occur during initialization
    # if len(bins) > max_bin_count:
    #     # If the number of bins exceeds the maximum genome length, we can keep only the longest bins until we reach the limit
    #     bins.sort(key=lambda x: x[1], reverse=True) # Sort bins by length
    #     bins = bins[:max_bin_count] # Keep only the longest bins up to the maximum genome length
    #     bins.sort(key=lambda x: x[0]) # Sort bins back
                    
    return bins
        
