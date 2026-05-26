import numpy as np
from joblib import Parallel, delayed


import core.musicProcessing as mp
from core.individual import Individual, binary_to_bins

class AlgProvider:
    mp: mp.MusicProcessingProvider
    random_seed: int
    
    # parameters
    num_generations: int
    population_size: int
    individual_bin_count: int # number of tuples in genome of all individuals 
    individual_max_bin_lenght: int
    clearing_interval: int
    clearing_radius: float
    mutation_probability: float
    crossover_probability: float
    niche_capacity: int
    
    # experiment parameters
    population_initialization_method: str
    crossover_method: str
    selection_method: str
    sim_metric: str
    
    population: list[Individual] = []
    similarity_memo = None
    
    # metrics to track
    diversity: np.ndarray = np.array([], dtype=np.float16)
    removed_by_clearing: np.ndarray = np.array([], dtype=np.int16)
    best_fitness: np.ndarray = np.array([], dtype=np.float16)
    avg_fitness: np.ndarray = np.array([], dtype=np.float16)
    worst_fitness: np.ndarray = np.array([], dtype=np.float16)
        
    def __init__(self, mp: mp.MusicProcessingProvider, random_seed: int, num_generations: int, population_size: int, individual_bin_count: int, mutation_probability: float, crossover_probability: float, crossover_method: str, selection_method: str, sim_metric: str, clearing_radius: float, clearing_interval: int, niche_capacity: int, population_initialization_method: str):
        self.mp = mp
        self.random_seed = random_seed
        
        self.num_generations = num_generations
        self.population_size = population_size
        self.individual_bin_count = individual_bin_count
        self.individual_max_bin_lenght = mp.sim_matrix.shape[0]
        self.clearing_interval = clearing_interval
        self.clearing_radius = clearing_radius
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.niche_capacity = niche_capacity
        
        self.population_initialization_method = population_initialization_method
        self.crossover_method = crossover_method
        self.selection_method = selection_method
        self.sim_metric = sim_metric
        
        self.similarity_memo = np.full((population_size, population_size), np.nan, dtype=np.float16)
    
    def initialize_population(self):
        if self.population_initialization_method == "random":
            self._initialize_random_pop(self.population_size, min_len=1, max_len=10)
        elif self.population_initialization_method == "elite":
            self._initialize_elite_pop(self.population_size, k=5)
        else:
            raise ValueError(f"Unknown population initialization method: {self.population_initialization_method}")
        
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
    
    def similarity(self, individualA: Individual, individualB: Individual) -> float:
        if self.sim_metric == "jaccard":
            return self._similarity_Jaccard(individualA, individualB)
        elif self.sim_metric == "fenotype":
            return self._similarity_fenotype(individualA, individualB)
        else:
            raise ValueError(f"Unknown similarity metric: {self.sim_metric}")

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
        

    def selection(self, population: list[Individual]) -> Individual:
        if self.selection_method == "tournament":
            return self._tournament_selection(population)
        elif self.selection_method == "roulette":
            return self._roulette_selection(population)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _tournament_selection(self, population: list[Individual], k: int = 5) -> Individual:
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
    
    def crossover(self, parentA: Individual, parentB: Individual) -> tuple[Individual, Individual]:
        np.random.seed(self.random_seed)
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
         

    def mutation(self, individual: Individual):
        # mutation probability is shared across all mutation types, multiple mutations can occur in one pass
        np.random.seed(self.random_seed)
        prob = self.mutation_probability/3 # divide by number of mutation types 
        
        if np.random.rand() < prob:
            self._shift_mutation(individual)
        np.random.seed(self.random_seed + 1) # reroll random seed
        if np.random.rand() < prob:
            self._length_mutation(individual)
        np.random.seed(self.random_seed + 2) # reroll random seed
        if np.random.rand() < prob:
            self._toggle_mutation(individual, min_length=-10, max_length=10)
        
    
    def _shift_mutation(self, individual: Individual):
        np.random.seed(self.random_seed)
        random_bin_index = np.random.randint(0, len(individual.genome))
        shift_amount = np.random.normal(loc=0, scale=5) # Shift around the center of the bin with a standard deviation of 5 frames
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
        
    def clearing(self, population: list[Individual]):
        # sort the pop by fitness
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        # reset memo since population order changed
        self.similarity_memo[:] = np.nan
        # keep track of which individuals have been cleared
        cleared = np.zeros(len(population), dtype=bool)
        remove_indices: set[int] = set()
        
        for i in range(len(population)):
            if cleared[i]:
                continue
            else:
                niche_count = 0
                for j in range(i, len(population)):
                    if j in remove_indices:
                        continue

                    if i == j:
                        similarity = 1.0
                    else:
                        similarity = self.similarity_memo[i, j]
                        if np.isnan(similarity):
                            similarity = self.similarity(population[i], population[j])
                            self.similarity_memo[i, j] = similarity
                            self.similarity_memo[j, i] = similarity

                    if similarity >= self.clearing_radius:
                        if niche_count < self.niche_capacity:
                            if i != j:
                                population[j].update_fitness(0.0) # Clear the fitness of similar individuals, but not the niche winner
                            cleared[j] = True
                            niche_count += 1
                        else:
                            remove_indices.add(j)

        if remove_indices:
            population = [ind for idx, ind in enumerate(population) if idx not in remove_indices]

        # log stuff
        self.removed_by_clearing = np.append(self.removed_by_clearing, len(remove_indices)) # individuals removed by clearing
        
        # average similarity in similarity memo as a diversity metric, excluding NaN values and self-similarity
        sim_values = self.similarity_memo[~np.isnan(self.similarity_memo)]
        sim_values = sim_values[sim_values < 1.0] # exclude self-similarity
        avg_similarity = np.mean(sim_values) if len(sim_values) > 0 else 0.0
        
        self.diversity = np.append(self.diversity, 1 - avg_similarity)
        
        
        # reset memo for next time clearing is called, since the population has changed
        self.similarity_memo = np.full(self.population_size, self.population_size, np.nan, dtype=np.float16)
        
        return population
        
    
    def run(self):
        self.initialize_population()
        self.calc_fitness_population()
        
        for generation in range(self.num_generations):
            new_population = []
            
            while len(new_population) < self.population_size:
                parentA = self.selection(self.population)
                parentB = self.selection(self.population)
                
                childA, childB = self.crossover(parentA, parentB)
                
                self.mutation(childA)
                self.mutation(childB)
                
                new_population.extend([childA, childB])
            
            if generation % self.clearing_interval == 0:
                new_population = self.clearing(new_population)
                
            self.population = new_population
            self.calc_fitness_population()

            # log pop fitness to track progress
            fitness_values = [ind.fitness for ind in self.population]
            self.best_fitness = np.append(self.best_fitness, np.max(fitness_values))
            self.avg_fitness = np.append(self.avg_fitness, np.mean(fitness_values))
            self.worst_fitness = np.append(self.worst_fitness, np.min(fitness_values))
            
            