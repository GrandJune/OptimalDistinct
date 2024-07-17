# -*- coding: utf-8 -*-
# @Time     : 6/25/2024 19:56
# @Author   : Junyi
# @FileName: Main.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from collections import Counter
from matplotlib.colors import ListedColormap, BoundaryNorm


class Environment:
    def __init__(self, N, num_ones=5):
        self.N = N
        # Step 1: Initialize the grid with 0s
        grid = [[0 for _ in range(self.N)] for _ in range(self.N)]

        # Step 2: Place a single 100 in the grid
        i, j = random.randint(0, N - 1), random.randint(0, N - 1)
        grid[i][j] = 100

        # Step 3: Place several 1s in the grid
        count = 0
        while count < num_ones:
            i, j = random.randint(0, N - 1), random.randint(0, N - 1)
            if grid[i][j] == 0:  # Ensure not to overwrite 100 or other 1s
                grid[i][j] = 1
                count += 1
        self.grid = grid
        self.occupation = [[0] * self.N] * self.N

    def show(self):
        # Visualize the grid using a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.grid, annot=True, cmap='Greys', linewidths=0.5, linecolor='black', cbar=True)
        plt.title('Resource Grid Heatmap')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

    def show_with_agents(self, position_list, distant_position, close_position):
         # Count the number of agents in each cell
         counter = {}
         for (x, y) in position_list:
             if (x, y) not in counter.keys():
                 counter[(x, y)] = 1
             else:
                 counter[(x, y)] += 1

         # Create a custom colormap
         cmap = ListedColormap(['white', 'grey', 'black'])
         norm = BoundaryNorm([0, 0.5, 1.5, 100.5], cmap.N)

         # Create the heatmap
         plt.figure(figsize=(8, 6))
         ax = sns.heatmap(self.grid, annot=False, cmap=cmap, norm=norm, linewidths=0.5, linecolor='black', cbar=True)

         # Overlay the agent counts
         for (x, y), count in counter.items():
             plt.text(y + 0.5, x + 0.5, str(count), color='red', ha='center', va='center', fontweight='bold')

         # Plot distant_position with a distinct marker
         distant_x, distant_y = distant_position
         plt.scatter(distant_y + 0.5, distant_x + 0.5, marker='o', color='blue', s=100, label='Distant Position')

         # Plot close_position with another distinct marker
         close_x, close_y = close_position
         plt.scatter(close_y + 0.5, close_x + 0.5, marker='x', color='green', s=100, label='Close Position')

         # Customize the color bar
         colorbar = ax.collections[0].colorbar
         colorbar.set_ticks([0.25, 1, 50])  # Center of each color interval
         colorbar.set_ticklabels([0, 1, 100])
         # Add border line to the color bar
         colorbar.outline.set_visible(True)
         colorbar.outline.set_linewidth(1)
         colorbar.outline.set_edgecolor('black')
         # Ensure the borders are shown
         ax.set_xlim(0, self.N)
         ax.set_ylim(0, self.N)

         # plt.title('Resource Grid with Agent Counts')
         plt.xlabel('X-axis')
         plt.ylabel('Y-axis')
         plt.tight_layout()
         plt.show()


    def get_payoff(self, position):
        cell_resource = self.grid[position[0]][position[1]]
        occupation = self.occupation[position[0]][position[1]]
        if occupation == 0:
            # for a potentially new position
            occupation = 1
        return cell_resource / occupation

class Agent:
    def __init__(self, environment):
        self.env = environment
        # randint will include two boundaries
        self.position = [random.randint(0, environment.N - 1), random.randint(0, environment.N - 1)]
        self.payoff = 0
        self.alpha = random.uniform(0, 1)  # emphasize on close position
        self.beta = random.uniform(0, 1)  # emphasize on distant position

    def show(self):
        print(self.position, self.payoff)

class Crowd:
    def __init__(self, environment, crowd_size, death_rate=0.5):
        self.env = environment
        # randint will include two boundaries
        self.crowd_size = crowd_size
        self.agent_list = []
        for index in range(self.crowd_size):
            agent = Agent(self.env)
            self.agent_list.append(agent)
        self.occupation = [[0 for _ in range(self.env.N)] for _ in range(self.env.N)]
        self.save_current_occupation()
        for agent in self.agent_list:
            agent.payoff = self.env.get_payoff(agent.position)
        self.death_rate = death_rate

    def save_current_occupation(self):
        self.occupation = [[0 for _ in range(self.env.N)] for _ in range(self.env.N)]
        for agent in self.agent_list:
            self.occupation[agent.position[0]][agent.position[1]] += 1
        # print("Occupation:", self.occupation)
        self.env.occupation = self.occupation

    def save_current_payoff(self):
        self.save_current_occupation()
        self.env.occupation = self.occupation
        for agent in self.agent_list:
            agent.payoff = self.env.get_payoff(agent.position)

    def manhattan_distance(self, x1, y1, x2, y2):
        return abs(x2 - x1) + abs(y2 - y1)

    def global_optimal_search(self, epoch=None):
        position_list = [agent.position for agent in self.agent_list]
        max_distance = -1
        min_distance = float('inf')
        distant_position, close_position = [], []
        for x in range(self.env.N):
            for y in range(self.env.N):
                # Calculate maximum Manhattan distance from (x, y) to all positions in the list
                distance_list = [self.manhattan_distance(x, y, px, py) for (px, py) in position_list]
                distance = sum(distance_list)
                if distance > max_distance:
                    max_distance = distance
                    distant_position = [x, y]
                if distance < min_distance:
                    min_distance = distance
                    close_position = [x, y]
        for agent in self.agent_list:
            # pursue a close position
            if random.uniform(0, 1) < agent.alpha:
                x, y = int(agent.position[0]), int(agent.position[1])
                if random.uniform(0, 1) < 0.5:  # randomly move x or y
                    x += 1 if agent.position[0] < close_position[0] else -1 if agent.position[0] > close_position[0] else 0
                else:
                    y += 1 if agent.position[1] < close_position[1] else -1 if agent.position[1] > close_position[1] else 0
                if self.env.get_payoff([x, y]) > agent.payoff:
                    agent.position = [x, y]
            # pursue a distant position
            if random.uniform(0, 1) < agent.beta:
                x, y = int(agent.position[0]), int(agent.position[1])
                if random.uniform(0, 1) < 0.5:  # randomly move x or y
                    x += 1 if agent.position[0] < distant_position[0] else -1 if agent.position[0] > distant_position[0] else 0
                else:
                    y += 1 if agent.position[1] < distant_position[1] else -1 if agent.position[1] > distant_position[1] else 0
                if self.env.get_payoff([x, y]) > agent.payoff:
                    agent.position = [x, y]
        self.save_current_payoff()

        # agent death
        for agent in self.agent_list:
            if agent.payoff == 0:
                if random.uniform(0, 1) < self.death_rate:
                    self.agent_list.remove(agent)

        if epoch % 5 == 0:
            position_list = [agent.position for agent in self.agent_list]
            self.env.show_with_agents(position_list, distant_position, close_position)

        # introduce new agents
        # while len(self.agent_list) < self.crowd_size:
        #     agent = Agent(self.env)
        #     self.agent_list.append(agent)
        # self.save_current_payoff()

    # def calculate_local_optimal_position(self, agents, k, m):
    #     # Find the k closest neighbors
    #     distances = [(agent, np.linalg.norm(np.array(self.position) - np.array(agent.position))) for agent in agents if
    #                  agent != self]
    #     closest_neighbors = sorted(distances, key=lambda x: x[1])[:k]
    #     closest_positions = [agent.position for agent, _ in closest_neighbors]
    #
    #     if not closest_positions:
    #         return self.position
    #
    #     # Calculate the center of the k closest neighbors
    #     center_position = tuple(np.mean(closest_positions, axis=0).astype(int))
    #
    #     # Find all possible positions that are exactly m distance from the center
    #     possible_positions = []
    #     for i in range(self.env.N):
    #         for j in range(self.env.N):
    #             if np.linalg.norm(np.array(center_position) - np.array((i, j))) == m:
    #                 possible_positions.append((i, j))
    #
    #     # Randomly select a new position from the possible positions
    #     if possible_positions:
    #         new_position = random.choice(possible_positions)
    #     else:
    #         new_position = self.position
    #
    #     return new_position


# def simulate(N, num_agents, num_iterations, k_neighbors, m_distance):
#     env = Environment(N)
#     agents = [Agent(env) for _ in range(num_agents)]
#     positions_over_time = []
#
#     for iteration in range(num_iterations):
#         positions = [agent.position for agent in agents]
#         positions_over_time.append(positions)
#
#         for agent in agents:
#             if random.random() < 0.5:
#                 new_position = agent.calculate_global_optimal_position(agents, m_distance)
#             else:
#                 new_position = agent.calculate_local_optimal_position(agents, k_neighbors, m_distance)
#             agent.move_to(new_position, agents)
#
#     return env, positions_over_time


# def animate_search_process(env, positions_over_time):
#     fig, ax = plt.subplots()
#     N = env.N
#
#     def update(frame):
#         ax.clear()
#         ax.imshow(env.grid, cmap='Greens', origin='upper', alpha=0.6)
#         ax.set_xticks(np.arange(-.5, N, 1), minor=True)
#         ax.set_yticks(np.arange(-.5, N, 1), minor=True)
#         ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
#
#         positions = positions_over_time[frame]
#         for pos in positions:
#             ax.plot(pos[1], pos[0], 'ro')  # note that matplotlib's plot uses (x, y), not (row, col)
#
#         ax.set_title(f'Step {frame + 1}')
#
#     anim = FuncAnimation(fig, update, frames=len(positions_over_time), repeat=False)
#     plt.show()

if __name__ == '__main__':

    # Parameters
    N = 10
    num_agents = 200
    num_iterations = 50
    k_neighbors = 3
    m_distance = 1
    environment = Environment(N)
    # position_list = [[1, 2], [5, 5] ]
    # environment.show()
    # environment.show_with_agents(position_list=position_list)
    crowd = Crowd(environment, num_agents)
    # crowd.save_current_occupation()
    for epoch in range(num_iterations):
        crowd.global_optimal_search(epoch)

    # env, positions_over_time = simulate(N, num_agents, num_iterations, k_neighbors, m_distance)
    # animate_search_process(env, positions_over_time)



