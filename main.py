import argparse
import random

import numpy as np
import pandas


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cities')
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=2.0, type=float)
    parser.add_argument('--rho', default=0.5, type=float)
    parser.add_argument('--Q', default=100.0, type=float)
    parser.add_argument('--num_ants', default=10, type=int)
    parser.add_argument('--num_iterations', default=100, type=int)

    args = parser.parse_args()

    return args


class Ant:
    def __init__(self):
        self.total_distance = 0
        self.tour = []


class ACO:
    def __init__(self, cities, alpha=1.0, beta=2.0, rho=0.5, Q=100.0, num_ants=10, num_iterations=100):
        self.cities = cities
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.num_ants = num_ants
        self.num_iterations = num_iterations

        self.num_cities = len(cities)
        self.dismat = np.array([[self.distance(i, j) for j in range(self.num_cities)] for i in range(self.num_cities)])
        self.phemat = np.ones_like(self.dismat)
        self.visibility = 1 / (self.dismat + np.diag([np.inf] * self.num_cities))

    def distance(self, city1, city2):
        return np.linalg.norm(self.cities[city1] - self.cities[city2])

    def update_phemat(self, ants):
        self.phemat *= (1 - self.rho)
        for ant in ants:
            contribution = self.Q / ant.total_distance
            for i in range(-1, len(ant.tour) - 1):
                self.phemat[ant.tour[i], ant.tour[i + 1]] += contribution

    def run(self):
        best_distance = np.inf
        best_tour = None

        for iteration in range(self.num_iterations):
            ants = [Ant() for _ in range(self.num_ants)]

            for ant in ants:
                start_city = random.randrange(self.num_cities)
                ant.tour.append(start_city)

                unvisited_cities = set(range(self.num_cities)) - {start_city}
                while unvisited_cities:
                    cur_city = ant.tour[-1]

                    probabilities = []
                    ordered_unvisited_cities = list(unvisited_cities)
                    for city in ordered_unvisited_cities:
                        p = (self.phemat[cur_city, city] ** self.alpha) * (self.visibility[cur_city, city] ** self.beta)
                        probabilities.append(p)
                    probabilities = np.array(probabilities)
                    probabilities /= probabilities.sum()

                    next_city = random.choices(ordered_unvisited_cities, weights=probabilities)[0]
                    ant.total_distance += self.distance(cur_city, next_city)
                    ant.tour.append(next_city)

                    unvisited_cities.remove(next_city)

                ant.total_distance += self.distance(ant.tour[-1], start_city)
                ant.tour.append(start_city)

                if ant.total_distance < best_distance:
                    best_distance = ant.total_distance
                    best_tour = ant.tour

            self.update_phemat(ants)

            print(f'Iter {iteration + 1}: Best distance {best_distance}')

        print(f'Best tour: {best_tour}')

        return best_tour


if __name__ == '__main__':
    args = parse_args()

    cities = pandas.read_csv(args.cities, header=None).to_numpy()

    aco = ACO(cities, args.alpha, args.beta, args.rho, args.Q, args.num_ants, args.num_iterations)
    aco.run()
