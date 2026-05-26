import numpy as np

class Individual_bins():
    max_bin_lenght: int
    genome: list[tuple[int, int]] # list of (index i and length l) will be sorted on index i
    genome_bin: np.ndarray # binary representation of the genome (1d array np.bool)
    fitness: float
    fenotype: np.ndarray # median frame of the genome, stored to avoid redundant calculations during similarity comparisons
    
    def __init__(self, genome: list[tuple[int, int]], max_bin_lenght: int):
        self.max_bin_lenght = max_bin_lenght
        self.genome = genome
        self.update_genome_bin()
        self.fitness = None
        self.fenotype = None
        
    def update_individual(self, genome: list[tuple[int, int]]=None, fenotype: np.ndarray=None, fitness: float=None):
        if genome is not None:
            self._update_genome(genome)
            self._update_genome_bin()
        if fenotype is not None:
            self._update_fenotype(fenotype)
        if fitness is not None:
            self._update_fitness(fitness)
    
    def _update_genome_bin(self):
        '''
        Converts the genome (list of tuples) into a binary representation (1d numpy array of bools). This allows for efficient similarity calculations using bitwise operations.
        '''
        binary = np.zeros(self.max_bin_lenght, dtype=np.bool)
        
        for bin in self.genome:
            if bin[0] + bin[1] < self.max_bin_lenght: # normal case, the bin fits within bounds
                binary[bin[0]:bin[0]+bin[1]] = True
            elif bin[0] + bin[1] < 0: # if the bin exceeds the start index, just set all preceding bits to 1
                binary[:bin[0]] = True
            else:
                binary[bin[0]:] = True # if the bin exceeds the max length, just set all remaining bits to 1
        self.genome_bin = binary
            
    def _update_fenotype(self, fenotype: np.ndarray): # calculated externally
        self.fenotype = fenotype
        
    def _update_fitness(self, fitness: float): # calculated externally
        self.fitness = fitness
        
    def _update_genome(self, genome: list[tuple[int, int]]): # calculated externally
        self.genome = genome
        
        
def binary_to_bins(genome_bin: np.ndarray, max_bin_lenght: int) -> list[tuple[int, int]]:
    """
    Converts the binary representation of the genome back into a list of tuples (index and length).
    """
    padded = np.concatenate(([False], genome_bin, [False]))
    
    diff = np.diff(padded.astype(np.int8))
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    bins = list(zip(starts.tolist(), (ends - starts).tolist()))
    
    # should only occur during initialization
    if len(bins) > max_bin_lenght:
        # If the number of bins exceeds the maximum genome length, we can keep only the longest bins until we reach the limit
        bins.sort(key=lambda x: x[1], reverse=True) # Sort bins by length
        bins = bins[:max_bin_lenght] # Keep only the longest bins up to the maximum genome length
        bins.sort(key=lambda x: x[0]) # Sort bins back
                    
    return bins
        
