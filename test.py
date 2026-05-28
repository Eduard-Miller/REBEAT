from core.musicProcessing import MusicProcessingProvider as mp
from core.alg import Alg
from core.alg_bins import AlgBins
from core.alg_binary import AlgBinary
from core.individual import Individual
from core.individual_bins import Individual_bins
from core.individual_binary import Individual_binary

from time import time
import matplotlib.pyplot as plt
import numpy as np

start_time = time()

# mp params
path = "/home/bordduk69/Nextcloud/Music/Mario/Main 2D/New Super Mario Bros. Wii/New Super Mario Bros. Wii Music - Forest.mp3"
duration = None
nfft = 2048
win_length = 2048
hop_length = 1024
window_type = "hann"

music_processor = mp(path, duration, nfft, hop_length, win_length, window_type)
print("Music processing done in {:.2f} seconds".format(time() - start_time))

# alg params
num_generations = 100
population_size = 200
random_seed = 42
clearing_interval = 20
clearing_radius = 0.8
mutation_probability = 0.05
crossover_probability = 0.5
niche_capacity = 5
population_initialization_method = "elite" # "random" or "elite"
selection_method = "tournament"
crossover_method = "two_point"
sim_metric = "jaccard" # "jaccard" or "fenotype"
Individual_bin_count = 10

alg = AlgBins(mp=music_processor, random_seed=random_seed, num_generations=num_generations, population_size=population_size, individual_bin_count=Individual_bin_count, mutation_probability=mutation_probability, crossover_probability=crossover_probability, crossover_method=crossover_method, selection_method=selection_method, sim_metric=sim_metric, clearing_radius=clearing_radius, clearing_interval=clearing_interval, niche_capacity=niche_capacity, population_initialization_method=population_initialization_method)
best_individuals = alg.run()

end_time = time()
print(f"Execution time: {end_time - start_time:.2f} seconds")


# plot stored metrics
plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
plt.plot(alg.avg_similarity, label="Diversity")
plt.xlabel("Generation")
plt.ylabel("Diversity")
plt.title("Population Diversity Over Generations")
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(alg.removed_by_clearing, label="Removed by Clearing")
plt.xlabel("Clearing Event")
plt.ylabel("Number of Individuals")
plt.title("Number of Individuals Removed by Clearing at Each Event")
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(alg.best_fitness, label="Best Fitness")
plt.plot(alg.avg_fitness, label="Average Fitness")
plt.plot(alg.worst_fitness, label="Worst Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Best, Average, and Worst Fitness Over Generations")
plt.legend()
plt.tight_layout()
plt.show()

results = alg.results()
for i, ind in enumerate(results):
    print(f"Best individual from niche {i+1}: Fitness = {ind.fitness:.4f}, Genome = {ind.genome}")