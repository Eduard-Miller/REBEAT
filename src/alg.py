import numpy as np
from joblib import Parallel, delayed
from abc import ABC, abstractmethod
from tqdm import tqdm

from individual import Individual
import musicProcessing as mp

class Alg(ABC):
    mp: mp.MusicProcessingProvider
    
    # parameters
    num_generations: int
    population_size: int
    sharing_radius: float
    mutation_probability: float
    crossover_probability: float
    
    # experiment parameters
    population_initialization_method: str
    crossover_method: str
    selection_method: str
    sim_metric: str
    
    population: list[Individual]
    similarity_memo: np.ndarray # memoization for similarity calculations, to avoid redundant calculations during clearing 2d array of shape (population_size, population_size) with values initialized to np.nan, where similarity_memo[i, j] stores the similarity between population[i] and population[j]
    
    # metrics to track
    avg_similarity: np.ndarray
    best_fitness: np.ndarray
    avg_fitness: np.ndarray
    worst_fitness: np.ndarray
    best_shared_fitness: np.ndarray
    avg_shared_fitness: np.ndarray
    worst_shared_fitness: np.ndarray
    
    @abstractmethod
    def initialize_population(self):
        pass
    
    
    @abstractmethod
    def similarity(self, individualA: Individual, individualB: Individual) -> float:
        pass
    
    def calc_similarity_pop(self, population: list[Individual]):
        # Calculate similarity for all pairs of individuals in the population
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                if i == j:
                    self.similarity_memo[i, j] = 1.0
                else:
                    similarity = self.similarity(population[i], population[j])
                    self.similarity_memo[i, j] = similarity
                    self.similarity_memo[j, i] = similarity
                    
    def reset_similarity_memo(self):
        self.similarity_memo = np.full((self.population_size, self.population_size), np.nan, dtype=np.float16)
    
    
    def selection(self, population: list[Individual]) -> Individual:
        if self.selection_method == "tournament":
            return self._tournament_selection(population)
        elif self.selection_method == "roulette":
            return self._roulette_selection(population)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    
    def _tournament_selection(self, population: list[Individual], k: int = 3) -> Individual:
        if len(population) < k:
            raise ValueError(f"Tournament size k={k} cannot be larger than the population size {len(population)}")
        tournament = list(np.random.choice(population, size=k, replace=False))
        tournament.sort(key=lambda ind: ind.fitness, reverse=True) # sort by shared fitness
        return tournament[0] # return the fittest individual in the tournament
    
    
    def _roulette_selection(self, population: list[Individual]) -> Individual:
        fitnesses = np.array([ind.fitness for ind in population])
        min_fitness = np.min(fitnesses)
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness
        total_fitness = np.sum(fitnesses)
        if total_fitness == 0:
            probabilities = np.ones(len(population)) / len(population) # Avoid division by zero, assign equal probabilities
        else:
            probabilities = fitnesses / total_fitness
        selected_index = np.random.choice(len(population), p=probabilities)
        return population[selected_index]
    
    
    @abstractmethod
    def crossover(self, parentA: Individual, parentB: Individual) -> tuple[Individual, Individual]:
        pass
    
    
    @abstractmethod
    def mutation(self, individual: Individual):
        pass
    
    
    @abstractmethod
    def calc_fitness_single(self, individual: Individual):
        pass
    
    
    def calc_fitness_population(self, population: list[Individual]):
        # Calculate fitness for all individuals in the population
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(self.calc_fitness_single)(individual) for individual in population
        )
        
        
    def calc_shared_fitness_population(self, population: list[Individual]):
        # Ensure sharing_radius is treated as a float
        sigma_sim = self.sharing_radius
        alpha = 1.0
        
        # 1. Vectorized calculation of the sharing matrix
        # Get the raw similarity matrix from your memo
        sim_matrix = np.array(self.similarity_memo)
        
        # Apply the threshold condition: if sim >= sigma_sim, apply formula, else 0
        # We use np.where to handle the conditional logic instantly across the whole matrix
        sharing_matrix = np.where(
            (sim_matrix >= sigma_sim) & (~np.isnan(sim_matrix)),
            ((sim_matrix - sigma_sim) / (1 - sigma_sim)) ** alpha,
            0.0
        )
        
        # 2. Calculate niche counts (sum of rows)
        # This automatically includes self-similarity (the diagonal will be 1.0)
        # and correctly counts actual identical clones in the population.
        niche_counts = np.sum(sharing_matrix, axis=1)
        
        # 3. Update individuals sequentially (incredibly fast now that math is done)
        for i, individual in enumerate(population):
            # Prevent any weird edge cases where niche_count < 1
            niche_count = max(1.0, niche_counts[i])
            
            shared_fitness = individual.fitness / niche_count
            individual.update_individual(shared_fitness=shared_fitness)
        
    
    def run(self):
        '''
        Run the genetic algorithm with set parameters and return the best individuals from every niche 
        '''
        
        self.initialize_population()
        self.calc_similarity_pop(self.population)
        self.calc_fitness_population(self.population)
        self.calc_shared_fitness_population(self.population)    
        
        for generation in tqdm(range(self.num_generations), desc="Generations"):
            new_population = []
            
            # elitism
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            new_population.extend(self.population[:self.population_size // 20])
                        
            while len(new_population) < self.population_size:
                parentA = self.selection(self.population)
                parentB = self.selection(self.population)
                
                childA, childB = self.crossover(parentA, parentB)
                
                self.mutation(childA)
                self.mutation(childB)
                
                new_population.extend([childA, childB])
                
            # ensure new population does not exceed population size (can only happen if population size is odd, since we add 2 children at a time)
            new_population = new_population[:self.population_size]
            
            self.calc_similarity_pop(new_population)
            self.calc_fitness_population(new_population)
            self.calc_shared_fitness_population(new_population)
            
            self.population = new_population

            # log pop fitness to track progress
            fitness_values = [ind.fitness for ind in self.population]
            self.best_fitness = np.append(self.best_fitness, np.max(fitness_values))
            self.avg_fitness = np.append(self.avg_fitness, np.mean(fitness_values))
            self.worst_fitness = np.append(self.worst_fitness, np.min(fitness_values))
            
            shared_fitness_values = [ind.shared_fitness for ind in self.population]
            self.best_shared_fitness = np.append(self.best_shared_fitness, np.max(shared_fitness_values))
            self.avg_shared_fitness = np.append(self.avg_shared_fitness, np.mean(shared_fitness_values))
            self.worst_shared_fitness = np.append(self.worst_shared_fitness, np.min(shared_fitness_values))

            # average similarity in similarity memo as a diversity metric, excluding NaN values and self-similarity
            sim_values = self.similarity_memo[~np.eye(self.population_size, dtype=bool) & ~np.isnan(self.similarity_memo)]
            new_avg_similarity = np.mean(sim_values) if len(sim_values) > 0 else 0.0
            
            self.avg_similarity = np.append(self.avg_similarity, new_avg_similarity)
            self.reset_similarity_memo()