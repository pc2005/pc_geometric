import numpy as np
import matplotlib.pyplot as plt


class PSO(object):
    def __init__(self, population_size, max_steps):
        self.w = 0.5 # 0.6  # 惯性权重
        self.c1 = self.c2 = 2
        self.population_size = population_size  # 粒子群数量
        self.dim = 2  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        self.x_bound = [0, 40]  # 解空间范围

        # self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
        #                            (self.population_size, self.dim))  # 初始化粒子群位置

        # generate two clusters
        self.x = np.concatenate((np.random.uniform(self.x_bound[0], self.x_bound[0]+15,
                                                  (population_size//2, self.dim)),
                                np.random.uniform(self.x_bound[1]-15, self.x_bound[1],
                                                  ((population_size-population_size//2),
                                                   self.dim))))

        # # plot intial point set
        # plt.figure()
        # plt.scatter(self.x[:, 0], self.x[:, 1], s=10, color='b')
        # plt.show()

        self.v = np.random.rand(self.population_size, self.dim)  # 初始化粒子群速度

        fitness = self.calculate_fitness(self.x)

        self.p = self.x  # 个体的最佳位置
        self.pg = [np.argmin(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度

    def calculate_fitness(self, x):
        return np.sum(np.square(x), axis=1)

    def evolve(self):
        fig = plt.figure()
        prev_fitness = 10000
        for step in range(self.max_steps):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)

            # 更新速度和权重
            self.v = self.w*self.v \
                + self.c1*r1 * (self.p-self.x) \
                # + self.c2*r2*(self.pg-self.x)
            self.x = self.v + self.x

            # visualize
            plt.clf()
            plt.scatter(self.x[:, 0], self.x[:, 1], s=10, color='b')
            plt.xlim(self.x_bound[0], self.x_bound[1])
            plt.ylim(self.x_bound[0], self.x_bound[1])
            plt.pause(0.01)
            fitness = self.calculate_fitness(self.x)

            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]

            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)] 
                self.global_best_fitness = np.min(fitness)

            mean_fitness = np.mean(fitness)
            print('best fitness: %.8f, mean fitness: %.8f' %
                  (self.global_best_fitness, mean_fitness))

            # if step > 0:
            #     mean_fitness_diff = abs(mean_fitness - prev_fitness)
            #     if mean_fitness_diff<0.0005:
            #         print('Stopping at %d with fitness diff %.8f' % (step, mean_fitness_diff) )
            #         return
                
            # prev_fitness = mean_fitness



if __name__ == '__main__':
    pso = PSO(population_size=100, max_steps=100)
    pso.evolve()
    plt.pause(2)
    plt.close()
