import numpy as np
from joblib import Parallel, delayed


import core.musicProcessing as mp
from core.individual import Individual, binary_to_bins

class AlgProvider:
    mp: mp.MusicProcessingProvider
    random_seed: int
    num_generations: int
    population_size: int
    population: list[Individual]
    individual_bin_count: int # number of tuples in genome of all individuals 
    individual_max_bin_lenght: int
    
    # metrics to track
    diversity: np.ndarray = np.array([], dtype=np.float16)
    best_fitness: np.ndarray = np.array([], dtype=np.float16)
    avg_fitness: np.ndarray = np.array([], dtype=np.float16)
    worst_fitness: np.ndarray = np.array([], dtype=np.float16)
        
    def __init__(self, mp: mp.MusicProcessingProvider, random_seed: int = 0, num_generations: int = 50, population_size: int = 100, individual_bin_count: int = 20):
        self.mp = mp
        self.individual_max_bin_lenght = mp.sim_matrix.shape[0]
        self.random_seed = random_seed
        self.num_generations = num_generations
        self.population_size = population_size
        self.individual_bin_count = individual_bin_count
        self.population = []

    def _initialize_random_pop(self, population_size: int, min_len: int, max_len: int):
        np.random.seed(self.random_seed)
        for _ in range(population_size):
            new_genome = [(np.random.randint(0, self.individual_max_bin_lenght), np.random.randint(min_len, max_len + 1)) for _ in range(self.individual_bin_count)]
            new_genome.sort(key=lambda x: x[0]) # sort genome by index
            self.population.append(Individual(new_genome, self.individual_max_bin_lenght))
            
    def _initialize_elite_pop(self, population_size: int, k: int):
        '''
        Initializes population with a random spread of frames, but with the top k similar frames to each frame included in the population. This gives the population a warm start
        '''
        np.random.seed(self.random_seed)
        # Randomly select frames to include in the population, must be unique for better coverage of the search space
        frames = np.random.choice(self.individual_max_bin_lenght, size=population_size, replace=False)

        def build_individual(frame: int) -> Individual:
            top_k_indices = self.mp.get_top_k_similar_frames(frame, k - 1)

            # Create a temporary binary representation to find contiguous regions
            genome_bin = np.zeros(self.individual_max_bin_lenght, dtype=bool)
            genome_bin[frame] = True
            genome_bin[top_k_indices] = True

            genome = binary_to_bins(genome_bin, self.individual_max_bin_lenght)

            return Individual(genome, self.individual_max_bin_lenght)

        self.population.extend(
            Parallel(n_jobs=-1)(
                delayed(build_individual)(frame) for frame in frames
            )
        )
            

    def _similarity_Jaccard(self, individualA: Individual, individualB: Individual) -> float:
        '''
        calculates the Jaccard similarity between two individuals
        '''
        intersection = np.sum(individualA.genome_bin & individualB.genome_bin)
        union = np.sum(individualA.genome_bin | individualB.genome_bin)
        return intersection / union if union != 0 else 0
    
    def _similarity_fenotype(self, individualA: Individual, individualB: Individual) -> float:
        '''
        compares the median frames of two individuals and returns the cosine similarity between them.
        '''
        median_A = self.mp.calc_median_frame(np.flatnonzero(individualA.genome_bin))
        median_B = self.mp.calc_median_frame(np.flatnonzero(individualB.genome_bin))
        
        # Calculate cosine similarity
        dot_product = np.dot(median_A, median_B)
        norm_A = np.linalg.norm(median_A)
        norm_B = np.linalg.norm(median_B)
        
        return dot_product / (norm_A * norm_B)
        

    def selection(self):
        pass
    
    def _tournament_selection(self, population: list[Individual], k: int) -> Individual:
        np.random.seed(self.random_seed)
        tournament = np.random.choice(population, size=k, replace=False)
        tournament.sort(key=lambda ind: ind.fitness, reverse=True) # sort by fitness
        return tournament[0] # return the fittest individual in the tournament
    
    def _roulette_selection(self, population: list[Individual]) -> Individual:
        np.random.seed(self.random_seed)
        fitnesses = np.array([ind.fitness for ind in population])
        total_fitness = np.sum(fitnesses)
        if total_fitness == 0:
            probabilities = np.ones(len(population)) / len(population) # Avoid division by zero, assign equal probabilities
        else:
            probabilities = fitnesses / total_fitness
        selected_index = np.random.choice(len(population), p=probabilities)
        return population[selected_index]
    
    def crossover(self):
        pass
    
    def _single_point_crossover(self, parentA: Individual, parentB: Individual) -> tuple[Individual, Individual]:
        np.random.seed(self.random_seed)
        crossover_point = np.random.randint(1, self.individual_bin_count - 1)
        
        childA_genome = parentA.genome[:crossover_point] + parentB.genome[crossover_point:]
        childB_genome = parentB.genome[:crossover_point] + parentA.genome[crossover_point:]
        
        return Individual(childA_genome, self.individual_max_bin_lenght), Individual(childB_genome, self.individual_max_bin_lenght)
        
    def _two_point_crossover(self, parentA: Individual, parentB: Individual) -> tuple[Individual, Individual]:
        np.random.seed(self.random_seed)
        point1 = np.random.randint(1, self.individual_bin_count - 2)
        point2 = np.random.randint(point1 + 1, self.individual_bin_count - 1)
        
        childA_genome = parentA.genome[:point1] + parentB.genome[point1:point2] + parentA.genome[point2:]
        childB_genome = parentB.genome[:point1] + parentA.genome[point1:point2] + parentB.genome[point2:]
        
        return Individual(childA_genome, self.individual_max_bin_lenght), Individual(childB_genome, self.individual_max_bin_lenght)
    
    def _rrc_crossover(self, parentA: Individual, parentB: Individual) -> tuple[Individual, Individual]:
        # Random Respectful Crossover (RRC): For each gene, randomly decide whether to take it from parentA or parentB, but always keep the gene if it's present in both parents. Operates on the binary genome representation.
        
        # XOR to find differing bits
        diff = np.bitwise_xor(parentA.genome_bin, parentB.genome_bin)
        
        np.random.seed(self.random_seed)
        childA_bin = np.where(diff, np.random.choice([parentA.genome_bin, parentB.genome_bin]), parentA.genome_bin)
        
        # change random seed to ensure different random choices for childB, but still deterministic and reproducible
        np.random.seed(-self.random_seed)
        childB_bin = np.where(diff, np.random.choice([parentA.genome_bin, parentB.genome_bin]), parentB.genome_bin)
        
        childA_genome = binary_to_bins(childA_bin, self.individual_max_bin_lenght)
        childB_genome = binary_to_bins(childB_bin, self.individual_max_bin_lenght)
        
        return Individual(childA_genome, self.individual_max_bin_lenght), Individual(childB_genome, self.individual_max_bin_lenght)
         

    def mutation(self):
        pass
    
    def _shift_mutation(self, individual: Individual):
        np.random.seed(self.random_seed)
        random_bin_index = np.random.randint(0, len(individual.genome))
        shift_amount = np.random.normal(loc=individual.genome[random_bin_index][0]+individual.genome[random_bin_index][1]/2, scale=5) # Shift around the center of the bin with a standard deviation of 5 frames
        shift_amount = np.floor(shift_amount).astype(int)
        
        #ensure i doesn't go out of bounds
        new_i = np.clip(individual.genome[random_bin_index][0] + shift_amount, 0, self.individual_max_bin_lenght)
        
        new_genome = individual.genome.copy()
        new_genome[random_bin_index] = (new_i, new_genome[random_bin_index][1]) # keep the same length, just shift the index
        
        individual.update_genome(new_genome)
        
    
    def _length_mutation(self, individual: Individual):
        np.random.seed(self.random_seed)
        random_bin_index = np.random.randint(0, len(individual.genome))
        length_change = np.random.normal(loc=0, scale=5) # Change length by a normal distribution with mean 0 and standard deviation of 5 frames
        length_change = np.floor(length_change).astype(int)
        
        new_length = individual.genome[random_bin_index][1] + length_change
        
        # no need to ensure bounds or that the new length is positive, because the update_genome method will handle that by clipping the binary
        
        new_genome = individual.genome.copy()
        new_genome[random_bin_index] = (new_genome[random_bin_index][0], new_length) # keep the same index, just change the length
        
        individual.update_genome(new_genome)

    def _toggle_mutation(self, individual: Individual, min_length: int, max_length: int):
        # Toggles a bin to have length 0 (effectively removing it) or from length 0 to a random length (effectively adding it back in)..
        np.random.seed(self.random_seed)
        random_bin_index = np.random.randint(0, len(individual.genome))
        current_length = individual.genome[random_bin_index][1]
        
        if current_length == 0:
            new_length = np.random.randint(min_length, max_length + 1)
        else:
            new_length = 0
        
        new_genome = individual.genome.copy()
        new_genome[random_bin_index] = (new_genome[random_bin_index][0], new_length) # keep the same index, just toggle the length
        
        individual.update_genome(new_genome)

    def calc_fitness_single(self, individual: Individual):
        frames = np.flatnonzero(individual.genome_bin)
        n_frames = len(frames)
        
        if n_frames < 2:
            return 0.0

        # Cohesion (Average pairwise similarity)
        sub_sim_matrix = self.mp.sim_matrix[np.ix_(frames, frames)]
        sum_similarities = np.sum(sub_sim_matrix) - n_frames # Remove diagonal 1s
        num_pairs = n_frames * (n_frames - 1)
        cohesion = sum_similarities / num_pairs

        # Length reward factor
        length_reward = np.log1p(n_frames) / np.log1p(self.mp.sim_matrix.shape[0])
        
        fitness = cohesion * length_reward
        
        individual.update_fitness(fitness)
    
    def calc_fitness_population(self):
        # Calculate fitness for all individuals in the population in parallel
        Parallel(n_jobs=-1)(
            delayed(self.calc_fitness_single)(ind) for ind in self.population
        )
        
    def clearing(self):
        pass
    
    def run(self):
        pass
    
        
        
