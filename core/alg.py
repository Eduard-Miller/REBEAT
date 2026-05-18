from sortedcontainers import SortedSet
import numpy as np

import core.musicProcessing as mp

class AlgProvider:
    mp = None
    random_seed = 0
    num_generations = 0
    max_population_size = 0
    population = []    
    
    # metrics to track
    diversity = []
    avg_fitness = []
    avg_genome_length = []
    genome_length_variance = []
    
    
    def __init__(self, mp: mp.MusicProcessingProvider, random_seed: int = 0, num_generations: int = 50, max_population_size: int = 100):
        self.mp = mp
        self.random_seed = random_seed
        self.num_generations = num_generations
        self.max_population_size = max_population_size

    def initialize_random_pop(self, population_size: int, min_genome_length: int, max_genome_length: int, min_value: int, max_value: int):
        np.random.seed(self.random_seed)
        for _ in range(population_size):
            genome_length = np.random.randint(min_genome_length, max_genome_length + 1)
            genome = SortedSet(np.random.randint(min_value, max_value + 1, size=genome_length).tolist())
            self.population.append(genome)
            
    def initialize_elite_pop(self, similarity_matrix: np.ndarray, population_size: int, min_genome_length: int, max_genome_length: int, min_value: int, max_value: int):
        '''
        Initializes population with a random spread of frames, but with the top k similar frames to each frame included in the population. This gives the population a warm start
        '''
        np.random.seed(self.random_seed)
        frames = np.random.randint(min_value, max_value + 1, size=population_size) # Randomly select frames to include in the population
        frames = set(frames.tolist()) # ensure frames are unique
        for frame in frames:
            genome_length = np.random.randint(min_genome_length, max_genome_length + 1)
            top_k_frames = self.mp.get_top_k_similar_frames(frame, genome_length - 1)
            genome = SortedSet([frame] + list(top_k_frames)) # Create genome with the selected frame and its top k similar frames
            self.population.append(genome)

    def similarity_Jaccard(self, individualA: SortedSet, individualB: SortedSet) -> float:
        '''
        calculates the Jaccard similarity between two individuals
        '''
        intersection = len(individualA.intersection(individualB))
        union = len(individualA.union(individualB))
        
        return intersection / union
    
    def similarity_fenotype(self, individualA: SortedSet, individualB: SortedSet) -> float:
        '''
        compares the median frames of two individuals and returns the cosine similarity between them.
        '''
        median_A = self.mp.calc_median_frame(individualA)
        median_B = self.mp.calc_median_frame(individualB)
        
        # Calculate cosine similarity
        dot_product = np.dot(median_A, median_B)
        norm_A = np.linalg.norm(median_A)
        norm_B = np.linalg.norm(median_B)
        
        return dot_product / (norm_A * norm_B)
        

    def selection(self):
        pass

    def crossover(self):
        pass

    def mutation(self):
        pass

    def calc_fitness(self, similarity_matrix: np.ndarray, individual: SortedSet) -> float:
        '''
        Fitness is calculated as the average similarity between all pairs of frames in the genome. Additionally, a size reward is applied to account for it being easier to achieve high similarity with fewer frames, which can lead to trivial solutions.
        '''
        if len(individual) < 3:
            return 0.0 # Enforce a minimum genome size of 3 to avoid trivial solutions
        
        similarities = []
        
        for i in range(len(individual)):
            for j in range(i + 1, len(individual)):
                frame_i = individual[i]
                frame_j = individual[j]
                similarities.append(similarity_matrix[frame_i, frame_j])
        
        avg_similarity = sum(similarities) / len(similarities)
        
        return avg_similarity * np.log(len(individual)) # Apply size reward to encourage larger genomes