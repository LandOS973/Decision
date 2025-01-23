from randomOneMax import RandomOneMax
from fitness import Fitness
import random
# plt
import matplotlib.pyplot as plt

def initialisation(npop):
    population = []
    for i in range(npop):
        population.append(RandomOneMax(100).getVector())
    return population

def selection2Best(population):
    population.sort(key=lambda x: Fitness(x).getFitness(), reverse=True)
    return population[:2]

def crossover(parent1, parent2):
    crossPoint = random.randint(0, 100)
    child1 = parent1[:crossPoint] + parent2[crossPoint:]
    child2 = parent2[:crossPoint] + parent1[crossPoint:]
    return child1, child2

def mutation1flip(individual):
    mutationPoint = random.randint(0, 99)
    if individual[mutationPoint] == 0:
        individual[mutationPoint] = 1
    else:
        individual[mutationPoint] = 0
    return individual

def mutation3flip(individual):
    for i in range(3):
        mutation1flip(individual)

def mutation5flip(individual):
    for i in range(5):
        mutation1flip(individual)

def replace_random(population, individual):
    population[random.randint(0, len(population)-1)] = individual

def evolution_with_mutation_test(npop, mutation_type):
    population = initialisation(npop)
    i = 0
    fitness_history = []

    while i < 2000 and Fitness(selection2Best(population)[0]).getFitness() < 100:
        parents = selection2Best(population)
        children = crossover(parents[0], parents[1])

        # Appliquer la mutation choisie
        for pop in population:
            if random.random() < 0.1:
                mutation_type(pop)

        for j in range(2):
            replace_random(population, children[j])
        i += 1
        fitness_history.append([i, Fitness(selection2Best(population)[0]).getFitness()])

    return fitness_history

def selectionRandom(population):
    return random.sample(population, 2)

def selectionTournament(population):
    tournament = random.sample(population, 5)
    tournament.sort(key=lambda x: Fitness(x).getFitness(), reverse=True)
    return tournament[:2]

def evolution_with_selection_test(npop, selection_type):
    population = initialisation(npop)
    i = 0
    fitness_history = []

    while i < 2000 and Fitness(selection2Best(population)[0]).getFitness() < 100:
        parents = selection_type(population)
        children = crossover(parents[0], parents[1])

        # Appliquer la mutation choisie
        for pop in population:
            if random.random() < 0.1:
                mutation1flip(pop)

        for j in range(2):
            replace_random(population, children[j])
        i += 1
        fitness_history.append([i, Fitness(selection2Best(population)[0]).getFitness()])

    return fitness_history





if __name__ == "__main__":
    for mutation_fn in [mutation1flip, mutation3flip, mutation5flip]:
        history = evolution_with_mutation_test(100, mutation_fn)
        x = [i[0] for i in history]
        y = [i[1] for i in history]
        plt.plot(x, y, label=mutation_fn.__name__)

    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title('Impact of Mutation on Fitness')
    plt.legend()
    plt.show()

    for selection_fn in [selectionRandom, selectionTournament, selection2Best]:
        history = evolution_with_selection_test(100, selection_fn)
        x = [i[0] for i in history]
        y = [i[1] for i in history]
        plt.plot(x, y, label=selection_fn.__name__)

    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title('Impact of Selection on Fitness')
    plt.legend()
    plt.show()

