import numpy as np

import core.musicProcessing as mp
from core.alg import Alg
from core.individual import Individual
from core.individual_bins import Individual_bins
from core.individual_bins import binary_to_bins


class AlgBins(Alg):
    individual_bin_count: int # number of tuples in genome of all individuals 
    individual_max_bin_lenght: int # total number of frames in song stored for easy access
      
    def __init__(self, mp: mp.MusicProcessingProvider, random_seed: int, num_generations: int, population_size: int, individual_bin_count: int, mutation_probability: float, crossover_probability: float, crossover_method: str, selection_method: str, sim_metric: str, clearing_radius: float, clearing_interval: int, niche_capacity: int, population_initialization_method: str):
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
        
        self.individual_bin_count = individual_bin_count
        self.individual_max_bin_lenght = mp.sim_matrix.shape[0]
        
        self.avg_similarity = np.array([], dtype=np.float16)
        self.removed_by_clearing = np.array([], dtype=np.int16)
        self.best_fitness = np.array([], dtype=np.float16)
        self.avg_fitness = np.array([], dtype=np.float16)
        self.worst_fitness = np.array([], dtype=np.float16)
    
    
    def initialize_population(self):
        if self.population_initialization_method == "random":
            self._initialize_random_pop(self.population_size)
        elif self.population_initialization_method == "elite":
            self._initialize_elite_pop(self.population_size, k=self.individual_max_bin_lenght//10) # top 10%
        else:
            raise ValueError(f"Unknown population initialization method: {self.population_initialization_method}")
        
        
    def _initialize_random_pop(self, population_size: int):
        min_total_coverage = 0.01
        max_total_coverage = 0.1
        min_len = max(1, int(np.floor((min_total_coverage * self.individual_max_bin_lenght) / self.individual_bin_count)))
        max_len = max(min_len + 1, int(np.ceil((max_total_coverage * self.individual_max_bin_lenght) / self.individual_bin_count)))
        
        for _ in range(population_size):
            new_genome = [
                (int(np.random.randint(0, self.individual_max_bin_lenght)), int(np.random.randint(min_len, max_len + 1)))
                for _ in range(self.individual_bin_count)
            ]
            new_genome.sort(key=lambda x: x[0]) # sort genome by index
            self.population.append(Individual_bins(new_genome, self.individual_max_bin_lenght))
          
            
    def _initialize_elite_pop(self, population_size: int, k: int):
        '''
        Initializes population with a random spread of frames, but with the top k similar frames to each frame included in the population. This gives the population a warm start
        '''
        # Randomly select frames to include in the population, must be unique for better coverage of the search space
        frames = np.random.choice(self.individual_max_bin_lenght, size=population_size, replace=False)

        def build_individual(frame: int) -> Individual_bins:
            top_k_indices = self.mp.get_top_k_similar_frames(frame, k - 1)

            # Create a temporary binary representation to find contiguous regions
            genome_bin = np.zeros(self.individual_max_bin_lenght, dtype=bool)
            genome_bin[frame] = True
            genome_bin[top_k_indices] = True

            genome = binary_to_bins(genome_bin, self.individual_bin_count)
            genome = [(int(i), int(length)) for i, length in genome]
            genome.sort(key=lambda x: x[0]) # sort genome by index

            return Individual_bins(genome, self.individual_max_bin_lenght)

        self.population = [build_individual(frame) for frame in frames]
    
    
    def similarity(self, individualA: Individual, individualB: Individual) -> float:
        if self.sim_metric == "jaccard":
            return self._similarity_Jaccard(individualA, individualB)
        elif self.sim_metric == "fenotype":
            return self._similarity_fenotype(individualA, individualB)
        else:
            raise ValueError(f"Unknown similarity metric: {self.sim_metric}")


    def _similarity_Jaccard(self, individualA: Individual_bins, individualB: Individual_bins) -> float:
        '''
        calculates the Jaccard similarity between two individuals
        '''
        intersection = np.sum(individualA.genome_bin & individualB.genome_bin)
        union = np.sum(individualA.genome_bin | individualB.genome_bin)
        return intersection / union if union != 0 else 0
    
    
    def _similarity_fenotype(self, individualA: Individual_bins, individualB: Individual_bins) -> float:
        '''
        compares the median frames of two individuals and returns the cosine similarity between them.
        '''
        if individualA.fenotype is None:
            median_A = self.mp.calc_median_frame(np.flatnonzero(individualA.genome_bin))
        else:
            median_A = individualA.fenotype
        if individualB.fenotype is None:
            median_B = self.mp.calc_median_frame(np.flatnonzero(individualB.genome_bin))
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


    def _single_point_crossover(self, parentA: Individual_bins, parentB: Individual_bins) -> tuple[Individual_bins, Individual_bins]:
        crossover_point = np.random.randint(1, self.individual_bin_count - 1)
        
        childA_genome = parentA.genome[:crossover_point] + parentB.genome[crossover_point:]
        childB_genome = parentB.genome[:crossover_point] + parentA.genome[crossover_point:]
        
        childA_genome.sort(key=lambda x: x[0]) # sort genome by index
        childB_genome.sort(key=lambda x: x[0]) # sort genome by index
        
        return Individual_bins(childA_genome, self.individual_max_bin_lenght), Individual_bins(childB_genome, self.individual_max_bin_lenght)
        
        
    def _two_point_crossover(self, parentA: Individual_bins, parentB: Individual_bins) -> tuple[Individual_bins, Individual_bins]:
        point1 = np.random.randint(1, self.individual_bin_count - 2)
        point2 = np.random.randint(point1 + 1, self.individual_bin_count - 1)
        
        childA_genome = parentA.genome[:point1] + parentB.genome[point1:point2] + parentA.genome[point2:]
        childB_genome = parentB.genome[:point1] + parentA.genome[point1:point2] + parentB.genome[point2:]
        
        childA_genome.sort(key=lambda x: x[0]) # sort genome by index
        childB_genome.sort(key=lambda x: x[0]) # sort genome by index
        
        return Individual_bins(childA_genome, self.individual_max_bin_lenght), Individual_bins(childB_genome, self.individual_max_bin_lenght)
    
    
    def _rrc_crossover(self, parentA: Individual_bins, parentB: Individual_bins) -> tuple[Individual_bins, Individual_bins]:
        # Random Respectful Crossover (RRC): For each gene, randomly decide whether to take it from parentA or parentB, but always keep the gene if it's present in both parents. Operates on the binary genome representation.
        
        # XOR to find differing bits
        diff = np.bitwise_xor(parentA.genome_bin, parentB.genome_bin)
        
        choose_parent_b = np.random.rand(*diff.shape) < 0.5
        childA_bin = np.where(diff, np.where(choose_parent_b, parentB.genome_bin, parentA.genome_bin), parentA.genome_bin)

        choose_parent_b = np.random.rand(*diff.shape) < 0.5
        childB_bin = np.where(diff, np.where(choose_parent_b, parentB.genome_bin, parentA.genome_bin), parentB.genome_bin)
        
        childA_genome = binary_to_bins(childA_bin, self.individual_max_bin_lenght)
        childB_genome = binary_to_bins(childB_bin, self.individual_max_bin_lenght)
        childA_genome = [(int(i), int(length)) for i, length in childA_genome]
        childB_genome = [(int(i), int(length)) for i, length in childB_genome]
        
        childA_genome.sort(key=lambda x: x[0]) # sort genome by index
        childB_genome.sort(key=lambda x: x[0]) # sort genome by index
        
        return Individual_bins(childA_genome, self.individual_max_bin_lenght), Individual_bins(childB_genome, self.individual_max_bin_lenght)
         

    def mutation(self, individual: Individual):
        # mutation probability is shared across all mutation types, multiple mutations can occur in one pass
        prob = self.mutation_probability/3 # divide by number of mutation types 
        
        if np.random.rand() < prob:
            self._shift_mutation(individual)
        if np.random.rand() < prob:
            self._length_mutation(individual)
        if np.random.rand() < prob:
            self._toggle_mutation(individual, min_length=-10, max_length=10)
        
    
    def _shift_mutation(self, individual: Individual_bins):
        random_bin_index = np.random.randint(0, len(individual.genome))
        shift_amount = np.random.normal(loc=0, scale=5) # Shift around the center of the bin with a standard deviation of 5 frames
        shift_amount = np.floor(shift_amount).astype(int)
        
        #ensure i doesn't go out of bounds
        new_i = int(np.clip(individual.genome[random_bin_index][0] + shift_amount, 0, self.individual_max_bin_lenght))
        
        new_genome = individual.genome.copy()
        new_genome[random_bin_index] = (new_i, int(new_genome[random_bin_index][1])) # keep the same length, just shift the index
        
        new_genome.sort(key=lambda x: x[0]) # sort genome by index
    
        individual.update_individual(genome=new_genome)
        
    
    def _length_mutation(self, individual: Individual_bins):
        random_bin_index = np.random.randint(0, len(individual.genome))
        length_change = np.random.normal(loc=0, scale=5) # Change length by a normal distribution with mean 0 and standard deviation of 5 frames
        length_change = int(np.floor(length_change).astype(int))
        
        new_length = int(individual.genome[random_bin_index][1] + length_change)
        
        # no need to ensure bounds or that the new length is positive, because the update_genome method will handle that by clipping the binary
        
        new_genome = individual.genome.copy()
        new_genome[random_bin_index] = (int(new_genome[random_bin_index][0]), new_length) # keep the same index, just change the length
        
        individual.update_individual(genome=new_genome)


    def _toggle_mutation(self, individual: Individual_bins, min_length: int, max_length: int):
        # Toggles a bin to have length 0 (effectively removing it) or from length 0 to a random length (effectively adding it back in)..
        random_bin_index = np.random.randint(0, len(individual.genome))
        current_length = individual.genome[random_bin_index][1]
        
        if current_length == 0:
            new_length = int(np.random.randint(min_length, max_length + 1))
        else:
            new_length = 0
        
        new_genome = individual.genome.copy()
        new_genome[random_bin_index] = (int(new_genome[random_bin_index][0]), int(new_length)) # keep the same index, just toggle the length
        
        individual.update_individual(genome=new_genome)


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
        length_reward = np.log1p(n_frames)
        
        fitness = cohesion * length_reward
        
        individual.update_individual(fitness=fitness)
