import numpy.random as rnd


class OsloModel:
    def __init__(self, length: int, prob: float = 0.5, seed: int = 12345):
        self.length = length
        self.prob = prob
        self.rng = rnd.default_rng(seed)
        self.local_slopes = [0] * length
        self.cum_heights = [0] * (length + 1)
        self.threshold_slopes = [self.threshold_slope(self.rng.uniform()) for i in range(length)]
        self.num_iter = 0

    def threshold_slope(self, u: float) -> int:
        return 1 if u < self.prob else 2

    def drive(self):
        self.local_slopes[0] += 1

    def relaxation(self) -> int:
        avalanche_size = 0
        while True:
            num_relaxations = 0
            for i in range(self.length):
                if self.local_slopes[i] > self.threshold_slopes[i]:
                    if 0 < i < self.length - 1:
                        self.local_slopes[i] -= 2
                        self.local_slopes[i - 1] += 1
                        self.local_slopes[i + 1] += 1
                    elif i == 0:
                        self.local_slopes[0] -= 2
                        self.local_slopes[1] += 1
                    else:
                        self.local_slopes[i] -= 1
                        self.local_slopes[i - 1] += 1
                    self.threshold_slopes[i] = self.threshold_slope(self.rng.uniform())
                    num_relaxations += 1
            avalanche_size += num_relaxations
            if num_relaxations == 0:
                break
        return avalanche_size

    def get_heights(self) -> []:
        n = self.length + 1
        heights = [0] * n
        for i in range(n-2, -1, -1):
            heights[i] = self.local_slopes[i] + heights[i + 1]
        return heights

    def total_height(self) -> float:
        return sum(self.local_slopes)

    def crossover_time(self) -> int:
        return sum([(i+1) * self.local_slopes[i] for i in range(self.length)])

    def update(self):
        heights = self.get_heights()
        n = self.length + 1
        self.cum_heights = [self.cum_heights[i] + heights[i] for i in range(n)]
        self.num_iter += 1

    def get_average_heights(self) -> []:
        n = self.length + 1
        avg_heights = [float(self.cum_heights[i]) / self.num_iter for i in range(n)]
        return avg_heights

    def run(self, max_grains: int, sample_times: [] = None) -> ():
        heights = []
        avalanches = [0] * max_grains
        tc = 0
        sample_heights = None
        if sample_times is not None:
            sample_heights = []
        for grain in range(max_grains):
            self.drive()
            s = self.relaxation()
            self.update()
            avalanches[grain] = s
            heights.append(self.total_height())
            tc = max(tc, self.crossover_time())
            if sample_times is not None:
                if grain in sample_times:
                    sample_heights.append(self.get_heights())
        return tc, heights, avalanches 
    
