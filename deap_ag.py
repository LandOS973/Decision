# Import des librairies/modules
from deap import base
from deap import creator
from deap import tools
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy

# Taille du problème
ONE_MAX_LENGTH = 100

# Paramètres AG
POPULATION_SIZE = 20
P_CROSSOVER = 1.0
P_MUTATION = 1.0
MAX_GENERATIONS = 500

#############################################
# Définition des éléments de base pour l'AG #
#############################################
toolbox = base.Toolbox()

# Opérateur qui retourne 0 ou 1
toolbox.register("zeroOrOne", random.randint, 0, 1)

# Fonction mono objectif qui maximise la première composante de fitness (c'est un tuple)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Classe Individual construite avec un conteneur list
creator.create("Individual", list, fitness=creator.FitnessMax)


# Initialisation des individus avec uniquement des 0
def zero():
    return 0


toolbox.register("individualCreator", tools.initRepeat, creator.Individual, zero, ONE_MAX_LENGTH)

# Initialisation de la population
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# Calcul de la fitness / fonction toolbox.evaluate()
def oneMaxFitness(individual):
    return sum(individual),  # Retourne un tuple


toolbox.register("evaluate", oneMaxFitness)

# Définition des opérateurs de sélection et de variation
# Vous pouvez remplacer 'tools.selTournament' par votre propre fonction de sélection
toolbox.register("select", tools.selTournament, tournsize=3)

# Croisement monopoint
toolbox.register("mate", tools.cxOnePoint)

# Mutation bitflip
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)


def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # Évaluation initiale de la population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Configuration des statistiques
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "max"]

    # Liste pour stocker les valeurs de fitness maximales
    maxFitnessValues = []

    # Boucle principale de l'algorithme génétique
    for gen in range(MAX_GENERATIONS):
        # Sélection des parents
        offspring = toolbox.select(population, len(population))
        # Clone les individus sélectionnés
        offspring = list(map(toolbox.clone, offspring))

        # Appliquer le croisement et la mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                # Supprimer la fitness des enfants
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Évaluation des individus avec une fitness invalide
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Mise à jour de la population
        population[:] = offspring

        # Collecte des statistiques
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        maxFitnessValues.append(record["max"])

        # Affichage des informations de la génération actuelle
        print(f"Generation {gen}: Max Fitness = {record['max']}")

    # Génération d'un graphique
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='blue', label='Max Fitness avec boucle personnalisée')
    plt.legend()
    plt.xlabel('Génération')
    plt.ylabel('Fitness')
    plt.title('Max Fitness au fil des Générations')
    plt.show()


if __name__ == '__main__':
    main()
