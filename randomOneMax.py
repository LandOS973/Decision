import random

class RandomOneMax:
    def __init__(self, n):
        self.n = n
        self.vector = [random.randint(0, 1) for i in range(n)]

    def getVector(self):
        return self.vector    