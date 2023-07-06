import population
import simulation 
import genome 
import creature 
import numpy as np
from tqdm import tqdm
import time

class Engine:
    def __init__(self, pop_size=10, gene_count=3, verbose = False) -> None:
        self.pop = population.Population(pop_size, gene_count)
        self.sim = simulation.Simulation()
        self.verbose = verbose

        # Telemetry
        self.fittest_history = []
        self.mean_history = []
        self.iteration_time = []

    def evolution_speed_mean(self):
        np.mean(np.diff(self.mean_history))

    def evolution_speed_fittest(self):
        np.mean(np.diff(self.fittest_history))

    def run(self, epochs=1000):
        #sim = simulation.ThreadedSim(pool_size=1)

        for epoch in tqdm(range(epochs)):
            start_time = time.time()
            for cr in self.pop.creatures:
                self.sim.run_creature(cr, 2400)

            fits = [cr.get_distance_travelled() for cr in self.pop.creatures]
            links = [len(cr.get_expanded_links()) for cr in self.pop.creatures]

            # Record telemetry
            fittest = np.round(np.max(fits), 3)
            mean = np.round(np.mean(fits), 3)
            self.fittest_history.append(fittest)
            self.mean_history.append(mean)

            if self.verbose:
                print("Epoch #", epoch, "fittest:", fittest, "mean:", mean, "mean links", np.round(np.mean(links)), "max links", np.round(np.max(links)))
  
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []
            for i in range(len(self.pop.creatures)):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = self.pop.creatures[p1_ind]
                p2 = self.pop.creatures[p2_ind]
                # now we have the parents!
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=0.1, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=0.25)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)

            # elitism
            max_fit = np.max(fits)
            for cr in self.pop.creatures:
                if cr.get_distance_travelled() == max_fit:
                    new_cr = creature.Creature(1)
                    new_cr.update_dna(cr.dna)
                    new_creatures[0] = new_cr
                    # filename = "results/elite_"+str(iteration)+".csv"
                    filename = "results/elite.csv"
                    genome.Genome.to_csv(cr.dna, filename)
                    break

            self.pop.creatures = new_creatures

            # Record telemetry — time taken to run iteration
            end_time = time.time()
            self.iteration_time.append(end_time - start_time)
