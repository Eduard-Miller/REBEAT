import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class Individual(ABC):
    genome: Any # can be any representation of the genome, will be defined in the specific individual classes
    fitness: float
    fenotype: np.ndarray # median frame of the genome, stored to avoid redundant calculations during similarity comparisons
    
    @abstractmethod
    def update_individual(self, genome, fenotype):
        pass