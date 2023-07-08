import itertools
import numpy as np
import Levenshtein

import creature


class Population:
    def __init__(self, pop_size, gene_count):
        self.creatures = [creature.Creature(
                          gene_count=gene_count) 
                          for i in range(pop_size)]

    def levenshtein_distance(self):
        pairwise_distances = []
        for dna1, dna2 in itertools.combinations([creature.dna for creature in self.creatures], 2):
            distance = Levenshtein.distance(np.array(dna1).flatten(), np.array(dna2).flatten())
            pairwise_distances.append(distance)

        return np.mean(pairwise_distances)

    @staticmethod
    def get_fitness_map(fits):
        fitmap = []
        total = 0
        for f in fits:
            total = total + f
            fitmap.append(total)
        return fitmap
    
    @staticmethod
    def select_parent(fitmap):
        r = np.random.rand() # 0-1
        r = r * fitmap[-1]
        for i in range(len(fitmap)):
            if r <= fitmap[i]:
                return i

