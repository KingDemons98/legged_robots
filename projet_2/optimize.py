import numpy as np


class Optimize(object):
    def __init__(self, val, type, nb_sim):
        self.start_val = val
        self.step = 0
        self.local_best_val = 0
        self.local_best_speed = 0
        self.new_val = 0
        self.step_dir = 1
        self.epsilon = 0.001
        self.run = 5
        self.best_speed = np.zeros(self.run)
        self.best_val = np.zeros(self.run)
        self.nb_sim = nb_sim-1
        self.count_run = 0
        self.multiplier = 0.4
        if type:
            self.sign = 1
        else:
            self.sign = -1

        self.reset_class()

    def reset_class(self):
        self.multiplier = self.multiplier + 0.2
        self.step = self.start_val * 0.05
        self.local_best_val = self.start_val * self.multiplier
        self.local_best_speed = 0
        self.new_val = self.start_val * self.multiplier
        self.step_dir = 1

    def opt(self, new_speed, sim):
        print(f"The speed is : {new_speed:.6f} [m/s]")
        if self.local_best_speed == 0:
            self.local_best_speed = new_speed
        elif self.sign * new_speed > self.sign * self. local_best_speed:
            if np.sign(self.step_dir) == np.sign(self.new_val + self.epsilon - self.local_best_val):
                self.step_dir = -1 * self.step_dir
                print("direction's change")
                self.step = self.step / 2
        else:
            self.local_best_speed = new_speed
            self.local_best_val = self.new_val
        self.new_val = self.new_val + self.step * self.step_dir
        print(f"The new val is :{self.new_val:.5f}")

        if (sim % 20) == self.nb_sim:

            print("END OF THE FIRST RUN, SWITCHING TO THE NEXT VALUE")
            self.best_val[self.count_run] = self.local_best_val
            self.best_speed[self.count_run] = self.local_best_speed
            self.count_run += 1

            if self.count_run == self.run:
                print("the 5 best runs are ")
                for i in range(5):
                    print(f" the best value is : {self.best_val[i]:.5f}, with a speed of : {self.best_speed[i]:.6f}")
            self.reset_class()
