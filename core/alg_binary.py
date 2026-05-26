import numpy as np

import core.musicProcessing as mp
from core.alg import Alg
from core.individual import Individual
from core.individual_binary import Individual_binary

class AlgBinary(Alg):
    binary_length: int
      
    def __init__(self, mp: mp.MusicProcessingProvider, random_seed: int, num_generations: int, population_size: int,  mutation_probability: float, crossover_probability: float, crossover_method: str, selection_method: str, sim_metric: str, clearing_radius: float, clearing_interval: int, niche_capacity: int, population_initialization_method: str):
        self.mp = mp
        np.random.seed(random_seed)  
              
        self.num_generations = num_generations
        self.population_size = population_size
        self.clearing_interval = clearing_interval
        self.clearing_radius = clearing_radius
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.niche_capacity = niche_capacity
        
        
        self.population_initialization_method = population_initialization_method
        self.crossover_method = crossover_method
        self.selection_method = selection_method
        self.sim_metric = sim_metric
        
        self.population = []
        
        self.binary_length = mp.sim_matrix.shape[0]
        
        self.diversity = np.array([], dtype=np.float16)
        self.removed_by_clearing = np.array([], dtype=np.int16)
        self.best_fitness = np.array([], dtype=np.float16)
        self.avg_fitness = np.array([], dtype=np.float16)
        self.worst_fitness = np.array([], dtype=np.float16)
    
    def initialize_population(self):
        if self.population_initialization_method == "random":
            self._initialize_random_pop(self.population_size)
        elif self.population_initialization_method == "elite":
            self._initialize_elite_pop(self.population_size, k=self.binary_length//10) # top 10%
        else:
            raise ValueError(f"Unknown population initialization method: {self.population_initialization_method}")
        
        
    def _initialize_random_pop(self, population_size: int):
        # create random islands of 1s in the binary genome, with random lengths and positions
        num_islands = [2, 10] # number of islands per genome
        island_length = [(self.binary_length - 1)*0.01, (self.binary_length - 1)*0.1] # length of each island
        for _ in range(population_size):
            genome = np.zeros(self.binary_length, dtype=bool)
            num_islands = np.random.randint(num_islands[0], num_islands[1] + 1)
            for _ in range(num_islands):
                start = np.random.randint(0, self.binary_length - island_length[1])
                end = start + np.random.randint(island_length[0], island_length[1] + 1) # random length of island
                genome[start:end] = True
            self.population.append(Individual_binary(genome))
            
    
    def _initialize_elite_pop(self, population_size: int, k: int):
        '''
        Initializes population with a random spread of frames, but with the top k similar frames to each frame included in the population. This gives the population a warm start
        '''
        # Randomly select frames to include in the population, must be unique for better coverage of the search space
        frames = np.random.choice(self.binary_length, size=population_size, replace=False)

        def build_individual(frame: int) -> Individual_binary:
            top_k_indices = self.mp.get_top_k_similar_frames(frame, k - 1)

            # Create a temporary binary representation to find contiguous regions
            genome = np.zeros(self.binary_length, dtype=bool)
            genome[frame] = True
            genome[top_k_indices] = True

            return Individual_binary(genome)

        self.population = [build_individual(frame) for frame in frames]
    
    
    def similarity(self, individualA: Individual, individualB: Individual) -> float:
        if self.sim_metric == "jaccard":
            return self._similarity_Jaccard(individualA, individualB)
        elif self.sim_metric == "fenotype":
            return self._similarity_fenotype(individualA, individualB)
        else:
            raise ValueError(f"Unknown similarity metric: {self.sim_metric}")


    def _similarity_Jaccard(self, individualA: Individual_binary, individualB: Individual_binary) -> float:
        '''
        calculates the Jaccard similarity between two individuals
        '''
        intersection = np.sum(individualA.genome & individualB.genome)
        union = np.sum(individualA.genome | individualB.genome)
        return intersection / union if union != 0 else 0
    
    
    def _similarity_fenotype(self, individualA: Individual_binary, individualB: Individual_binary) -> float:
        '''
        compares the median frames of two individuals and returns the cosine similarity between them.
        '''
        if individualA.fenotype is None:
            median_A = self.mp.calc_median_frame(np.flatnonzero(individualA.genome))
        else:
            median_A = individualA.fenotype
        if individualB.fenotype is None:
            median_B = self.mp.calc_median_frame(np.flatnonzero(individualB.genome))
        else:
            median_B = individualB.fenotype

        # Calculate cosine similarity
        dot_product = np.dot(median_A, median_B)
        norm_A = np.linalg.norm(median_A)
        norm_B = np.linalg.norm(median_B)
        
        return dot_product / (norm_A * norm_B)
    
    
    def crossover(self, parentA: Individual, parentB: Individual) -> tuple[Individual, Individual]:
        if np.random.rand() < self.crossover_probability:
            if self.crossover_method == "single_point":
                return self._single_point_crossover(parentA, parentB)
            elif self.crossover_method == "two_point":
                return self._two_point_crossover(parentA, parentB)
            elif self.crossover_method == "rrc":
                return self._rrc_crossover(parentA, parentB)
            else:
                raise ValueError(f"Unknown crossover method: {self.crossover_method}")
        else:
            return parentA, parentB # no crossover, return parents as children


    def _single_point_crossover(self, parentA: Individual_binary, parentB: Individual_binary) -> tuple[Individual_binary, Individual_binary]:
        crossover_point = np.random.randint(1, self.binary_length - 1)
        
        childA_genome = parentA.genome[:crossover_point] + parentB.genome[crossover_point:]
        childB_genome = parentB.genome[:crossover_point] + parentA.genome[crossover_point:]
        
        return Individual_binary(childA_genome), Individual_binary(childB_genome)
        
        
    def _two_point_crossover(self, parentA: Individual_binary, parentB: Individual_binary) -> tuple[Individual_binary, Individual_binary]:
        point1 = np.random.randint(1, self.binary_length - 2)
        point2 = np.random.randint(point1 + 1, self.binary_length - 1)
        
        childA_genome = parentA.genome[:point1] + parentB.genome[point1:point2] + parentA.genome[point2:]
        childB_genome = parentB.genome[:point1] + parentA.genome[point1:point2] + parentB.genome[point2:]
        
        return Individual_binary(childA_genome), Individual_binary(childB_genome)
    
    
    def _rrc_crossover(self, parentA: Individual_binary, parentB: Individual_binary) -> tuple[Individual_binary, Individual_binary]:
        # Random Respectful Crossover (RRC): For each gene, randomly decide whether to take it from parentA or parentB, but always keep the gene if it's present in both parents. Operates on the binary genome representation.
        
        # XOR to find differing bits
        diff = np.bitwise_xor(parentA.genome, parentB.genome)
        
        childA_genome = np.where(diff, np.random.choice([parentA.genome, parentB.genome]), parentA.genome)

        childB_genome = np.where(diff, np.random.choice([parentA.genome, parentB.genome]), parentB.genome)
                
        return Individual_binary(childA_genome), Individual_binary(childB_genome)


    def mutation(self, individual: Individual):
        # mutation probability is shared across all mutation types, multiple mutations can occur in one pass
        prob = self.mutation_probability/3 # divide by number of mutation types 
        
        pass
        
        
    def burst_mutation(self, individual: Individual_binary):
        pass
    
    def shift_mutation(self, individual: Individual_binary):
        pass
    
    def edge_mutation(self, individual: Individual_binary):
        pass

    def calc_fitness_single(self, individual: Individual):
        frames = np.flatnonzero(individual.genome_bin)
        n_frames = len(frames)
        
        if n_frames < 2:
            individual.update_individual(fitness=0.0)
            return

        # Cohesion (Average pairwise similarity)
        sub_sim_matrix = self.mp.sim_matrix[np.ix_(frames, frames)]
        sum_similarities = np.sum(sub_sim_matrix) - n_frames # Remove diagonal 1s
        num_pairs = n_frames * (n_frames - 1)
        cohesion = sum_similarities / num_pairs

        # Length reward factor
        length_reward = np.log1p(n_frames) / np.log1p(self.mp.sim_matrix.shape[0])
        
        fitness = cohesion * length_reward
        
        individual.update_individual(fitness=fitness)
