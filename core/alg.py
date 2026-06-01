import numpy as np
from abc import ABC, abstractmethod

from core.individual import Individual
import core.musicProcessing as mp

class Alg(ABC):
    mp: mp.MusicProcessingProvider
    
    # parameters
    num_generations: int
    population_size: int
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
    
    population: list[Individual]
    similarity_memo: np.ndarray # memoization for similarity calculations, to avoid redundant calculations during clearing
    
    # metrics to track
    avg_similarity: np.ndarray
    removed_by_clearing: np.ndarray
    best_fitness: np.ndarray
    avg_fitness: np.ndarray
    worst_fitness: np.ndarray
    
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
        k = min(k, self.niche_capacity)
        if len(population) < k:
            raise ValueError(f"Tournament size k={k} cannot be larger than the population size {len(population)}")
        tournament = list(np.random.choice(population, size=k, replace=False))
        tournament.sort(key=lambda ind: ind.fitness, reverse=True) # sort by fitness
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
    
    
    def calc_fitness_population(self):
        # Calculate fitness for all individuals in the population
        for individual in self.population:
            self.calc_fitness_single(individual)
        
        
    def clearing(self, population: list[Individual]):
        # sort the pop by fitness
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        # reset memo since population order changed
        # self.reset_similarity_memo()
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
                                population[j].update_individual(fitness=0.0) # Clear the fitness of similar individuals, but not the niche winner
                            cleared[j] = True
                            niche_count += 1
                        else:
                            remove_indices.add(j)

        if remove_indices:
            population = [ind for idx, ind in enumerate(population) if idx not in remove_indices]

        # log stuff
        self.removed_by_clearing = np.append(self.removed_by_clearing, len(remove_indices)) # individuals removed by clearing
        
        # # average similarity in similarity memo as a diversity metric, excluding NaN values and self-similarity
        # sim_values = self.similarity_memo[~np.isnan(self.similarity_memo)]
        # sim_values = sim_values[sim_values < 1.0] # exclude self-similarity
        # new_avg_similarity = np.mean(sim_values) if len(sim_values) > 0 else 0.0
        
        # self.avg_similarity = np.append(self.avg_similarity, new_avg_similarity)
        
        return population
        
    
    def run(self):
        '''
        Run the genetic algorithm with set parameters and return the best individuals from every niche 
        '''
        
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
                
            # ensure new population does not exceed population size (can only happen if population size is odd, since we add 2 children at a time)
            new_population = new_population[:self.population_size]
            
            self.calc_fitness_population() # calculate fitness for the new population before clearing, to have the fitness values ready for clearing
            self.calc_similarity_pop(new_population) # calculate similarity for the new population before clearing, to have the memo ready for clearing
            
            if generation % self.clearing_interval == 0:
                new_population = self.clearing(new_population)
                
            self.population = new_population

            # log pop fitness to track progress
            fitness_values = [ind.fitness for ind in self.population]
            self.best_fitness = np.append(self.best_fitness, np.max(fitness_values))
            self.avg_fitness = np.append(self.avg_fitness, np.mean(fitness_values))
            self.worst_fitness = np.append(self.worst_fitness, np.min(fitness_values))
            
            # average similarity in similarity memo as a diversity metric, excluding NaN values and self-similarity
            sim_values = self.similarity_memo[~np.isnan(self.similarity_memo)]
            sim_values = sim_values[sim_values < 1.0] # exclude self-similarity
            new_avg_similarity = np.mean(sim_values) if len(sim_values) > 0 else 0.0
            
            self.avg_similarity = np.append(self.avg_similarity, new_avg_similarity)
            self.reset_similarity_memo()
            

    def results(self) -> list[Individual]:            
        # clear the final population to get the best individuals from each niche
        original_niche_capacity = self.niche_capacity
        self.niche_capacity = 1 # set niche capacity to 1 to only keep the best individual from each niche
        final_population = self.population.copy()
        final_population = self.clearing(final_population)
        
        self.niche_capacity = original_niche_capacity # restore original niche capacity for future runs if needed
        
        return final_population