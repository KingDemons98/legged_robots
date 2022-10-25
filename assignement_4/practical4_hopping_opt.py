# Hopping practical optimization
import datetime

import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.moo.nsga2 import NSGA2

from env.leg_gym_env import LegGymEnv
from leg_helpers import *

SINGLE_JUMP = False

class HoppingProblem(ElementwiseProblem):
    """Define interface to problem (see pymoo documentation). """
    def __init__(self):
        super().__init__(n_var=11,                 # number of variables to optimize (sample)
                         n_obj=1,                 # number of objectives
                         n_ieq_constr=0,          # no inequalities
                         xl=np.array([0.1, 20, 20, 200, 200, 10, 10, 2, 2, 0.1, 0.1]),   # variable lower limits (what makes sense?)
                         xu=np.array([2, 300, 300, 800, 800, 50, 50, 20, 20, 1., 1.]))   # variable upper limits (what makes sense?)
        # Define environment
        self.env = LegGymEnv(render=False,  # don't render during optimization
                on_rack=False, 
                motor_control_mode='TORQUE',
                action_repeat=1,
                )


    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate environment with variables chosen by optimization. """

        # Reset the environment before applying new profile (we want to start from the same conditions)
        self.env.reset()

        # Sample variables to optimize 
        f = x[0]        # hopping frequency
        Fz_max = x[1]   # max peak force in Z direction
        Fx_max = x[2]      # max peak force in X direction (can add)

        # cartesian controller gains
        kpCartesian_1 = x[3]
        kpCartesian_2 = x[4]
        kdCartesian_1 = x[5]
        kdCartesian_2 = x[6]

        # joints controller gains
        kpJoint_1 = x[7]
        kpJoint_2 = x[8]
        kdJoint_1 = x[9]
        kdJoint_2 = x[10]

        # [TODO] feel free to add more variables! What else could you optimize? 

        # Note: the below should look essentially the same as in practical4_hopping.py. 
        #   If you have some specific gains (or other variables here), make sure to test 
        #   the optimized variables under the same conditions.
        
        NUM_SECONDS = 5   # simulate N seconds (sim dt is 0.001)
        t = np.linspace(0,NUM_SECONDS,NUM_SECONDS*1000 + 1)

        # design Z force trajectory as a funtion of Fz_max, f, t
        #   Hint: use a sine function (but don't forget to remove positive forces)
        force_traj_z = Fz_max * np.sin(t * 2 * np.pi * f)
        force_traj_z[force_traj_z > 0] = 0

        if SINGLE_JUMP:
            # remove rest of profile (just keep the first peak)
            period = int((1 / f) * 1000)
            force_traj_z[period:] = 0

        # design X force trajectory as a funtion of Fx_max, f, t
        force_traj_x = Fx_max * np.sin(t * 2 * np.pi * f)
        force_traj_x[force_traj_x > 0] = 0
        
        # sample Cartesian PD gains (can change or optimize)
        kpCartesian = np.diag([kpCartesian_1,kpCartesian_2])
        kdCartesian = np.diag([kdCartesian_1,kdCartesian_2])

        kpJoint = np.array([kpJoint_1, kpJoint_2])
        kdJoint = np.array([kdJoint_1, kdJoint_2])

        # sample nominal foot position (can change or optimize)
        nominal_foot_pos = np.array([0.0,-0.2]) 

        # Keep track of environment states - what should you optimize? how about for max lateral jumping?
        #   sample states to consider 
        sum_z_height = 0
        max_base_z = 0
        max_base_x = 0

        # Track the profile: what kind of controller will you use? 
        for i in range(NUM_SECONDS*1000):
            # Torques
            tau = np.zeros(2) 

            # Compute jacobian and foot_pos in leg frame (use GetMotorAngles() )
            J, ee_pos_legFrame = jacobian_rel(self.env.robot.GetMotorAngles())

            foot_pos = ee_pos_legFrame
            des_foot_pos = nominal_foot_pos
            motor_vel = self.env.robot.GetMotorVelocities()
            foot_vel = np.matmul(J, motor_vel)
            des_foot_vel = 0
            tau += np.matmul(np.transpose(J), np.matmul(kpCartesian, des_foot_pos - foot_pos) + np.matmul(kdCartesian,
                                                                                                          des_foot_vel - foot_vel))

            qdes = ik_geometrical(xz=des_foot_pos)  # ik_geometrical

            tau += kpJoint * (qdes - self.env.robot.GetMotorAngles()) + kdJoint * (0 - motor_vel)

            # Add force profile contribution
            tau += J.T @ np.array([force_traj_x[i], force_traj_z[i]])

            # Apply control, simulate
            self.env.step(tau)

            # Record max base position (and/or other states)
            base_pos = self.env.robot.GetBasePosition()
            sum_z_height += base_pos[2]
            if base_pos[2] > max_base_z:
                max_base_z = base_pos[2]

            if base_pos[0] > max_base_x:
                max_base_x = base_pos[0]

        # objective function (what do we want to minimize?) 
        # f1 = max_base_x - 15
        f1 = -max_base_z - max_base_x

        # g1 = -max_base_z + 1

        out["F"] = [f1]
        # out["G"] = [g1]


# Define problem
problem = HoppingProblem()

# Define algorithms and initial conditions (depends on your variable ranges you selected above!)
algorithm = CMAES(x0=np.array([1.5, 50, 90, 500, 300, 30, 20, 6, 6, 0.6, 0.6])) #

# Run optimization
res = minimize(problem,
               algorithm,
               ('n_iter', 25), # may need to increase number of iterations
               seed=1,
               verbose=True)

#writing the results of the optimisation to a .txt file to simplify the tests
filename = "solutions" + '/' + datetime.datetime.now().strftime("solution-%Y-%m-%d-%H-%M-%S-%f") + ".txt"
f = open(filename, "x")
f.write(f"kpCartesian = np.diag([{res.X[3]}, {res.X[4]}])\n")
f.write(f"kdCartesian = np.diag([{res.X[5]}, {res.X[6]}])\n\n")

f.write(f"kpJoint = np.array([{res.X[7]}, {res.X[8]}])\n")
f.write(f"kdJoint = np.array([{res.X[9]}, {res.X[10]}])\n\n")

f.write(f"Fx_max = {res.X[2]}\n")
f.write(f"Fz_max = {res.X[1]}\n")
f.write(f"f = {res.X[0]}\n")

print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
print("Check your optimized variables in practical4_hopping.py")