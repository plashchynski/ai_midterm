import population
import simulation 
import genome 
import creature 
import numpy as np
from tqdm import tqdm
import time

class Engine:
    def __init__(self, pop_size=10, gene_count=3, point_mutation_rate=0.1, point_mutation_amount=0.25, shrink_mutation_rate=0.25, grow_mutation_rate=0.1, verbose = False, pool_size = 1, save_elite = False) -> None:
        self.pop = population.Population(pop_size, gene_count)
        self.pool_size = pool_size
        if pool_size > 1:
            self.sim = simulation.ThreadedSim(self.pool_size)
        else:
            self.sim = simulation.Simulation()
        
        self.verbose = verbose
        self.save_elite = save_elite

        self.point_mutation_rate = point_mutation_rate
        self.point_mutation_amount = point_mutation_amount
        self.shrink_mutation_rate = shrink_mutation_rate
        self.grow_mutation_rate = grow_mutation_rate

        # Telemetry
        self.fittest_history = []
        self.mean_history = []
        self.generation_processing_time = []
        self.levenstein_distances = []
        self.numbers_of_links = []

    # Metrics
    def best_fit(self):
        return np.max(self.fittest_history)

    def mean_fit(self):
        return np.mean(self.mean_history)

    def mean_improvement_speed(self):
        return np.mean(np.diff(self.mean_history))

    def fittest_improvement_speed(self):
        return np.mean(np.diff(self.fittest_history))

    def best_fitness_std(self):
        return np.std(self.fittest_history)

    def mean_levenstein_distance(self):
        return np.mean(self.levenstein_distances)

    def mean_generation_processing_time(self):
        return np.mean(self.generation_processing_time)
    
    def mean_numbers_of_links(self):
        return np.mean(self.numbers_of_links)

    def print_metrics(self):
        print("=====================================================")
        print("Best fitness: ", np.round(self.best_fit(), decimals=3))
        print("Mean fitness: ", np.round(self.mean_fit(), decimals=3))
        print("Mean improvement speed: ", np.round(self.mean_improvement_speed(), decimals=3))
        print("Fittest improvement speed: ", np.round(self.fittest_improvement_speed(), decimals=3))
        print("Best fitness std: ", np.round(self.best_fitness_std(), decimals=3))
        print("Mean Levenstein distance: ", np.round(self.mean_levenstein_distance(), decimals=3))
        print("Mean generation processing time: ", np.round(self.mean_generation_processing_time(), decimals=3))
        print("Mean numbers of links: ", np.round(self.mean_numbers_of_links(), decimals=3))

    def run(self, generations=1000):
        for generation in tqdm(range(generations)):
        # for generation in range(generations):
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
            self.levenstein_distances.append(self.pop.levenshtein_distance())
            self.numbers_of_links.append(np.mean(links))

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
                dna = genome.Genome.point_mutate(dna,
                                                 rate=self.point_mutation_rate,
                                                 amount=self.point_mutation_amount)
                dna = genome.Genome.shrink_mutate(dna, rate=self.shrink_mutation_rate)
                dna = genome.Genome.grow_mutate(dna, rate=self.grow_mutation_rate)

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
                    
                    if self.save_elite:
                        filename = "results/elite_"+str(generation)+".csv"
                        # filename = "results/elite.csv"
                        genome.Genome.to_csv(cr.dna, filename)

                    break

            self.pop.creatures = new_creatures

            # Record telemetry — time taken to run iteration
            end_time = time.time()
            self.generation_processing_time.append(end_time - start_time)
