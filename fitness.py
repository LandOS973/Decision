class Fitness:
    def __init__(self, vector):
        self.vector = vector
        self.fitness = sum(vector)

    def getFitness(self):
        return self.fitness