# Hopping practical optimization
import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

from env.leg_gym_env import LegGymEnv
from leg_helpers import *

SINGLE_JUMP = True

class HoppingProblem(ElementwiseProblem):
    """Define interface to problem (see pymoo documentation). """
    def __init__(self):
        super().__init__(n_var=2,                 # number of variables to optimize (sample)
                         n_obj=1,                 # number of objectives
                         n_ieq_constr=0,          # no inequalities 
                         xl=np.array([0., 0.]),   # variable lower limits (what makes sense?)
                         xu=np.array([1., 1.]))   # variable upper limits (what makes sense?) 
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
        Fx_max = 0      # max peak force in X direction (can add)
        # [TODO] feel free to add more variables! What else could you optimize? 

        # Note: the below should look essentially the same as in practical4_hopping.py. 
        #   If you have some specific gains (or other variables here), make sure to test 
        #   the optimized variables under the same conditions.
        
        NUM_SECONDS = 5   # simulate N seconds (sim dt is 0.001)
        t = np.linspace(0,NUM_SECONDS,NUM_SECONDS*1000 + 1)

        # design Z force trajectory as a funtion of Fz_max, f, t
        #   Hint: use a sine function (but don't forget to remove positive forces)
        force_traj_z = np.zeros(len(t))

        if SINGLE_JUMP:
            # remove rest of profile (just keep the first peak)
            force_traj_z = np.zeros(len(t))

        # design X force trajectory as a funtion of Fx_max, f, t
        force_traj_x = np.zeros(len(t))
        
        # sample Cartesian PD gains (can change or optimize)
        kpCartesian = np.diag([500,300])
        kdCartesian = np.diag([30,20])

        # sample nominal foot position (can change or optimize)
        nominal_foot_pos = np.array([0.0,-0.2]) 

        # Keep track of environment states - what should you optimize? how about for max lateral jumping?
        #   sample states to consider 
        sum_z_height = 0
        max_base_z = 0

        # Track the profile: what kind of controller will you use? 
        for i in range(NUM_SECONDS*1000):
            # Torques
            tau = np.zeros(2) 

            # Compute jacobian and foot_pos in leg frame (use GetMotorAngles() )
            J, ee_pos_legFrame = jacobian_rel(self.env.robot.GetMotorAngles())

            # Add Cartesian PD (and/or joint PD? Think carefully about this, and try it out.)
            tau += np.zeros(2) # [TODO]

            # Add force profile contribution
            tau += J.T @ np.array([force_traj_x[i], force_traj_z[i]])

            # Apply control, simulate
            self.env.step(tau)

            # Record max base position (and/or other states)
            base_pos = self.env.robot.GetBasePosition()
            sum_z_height += base_pos[2]
            if base_pos[2] > max_base_z:
                max_base_z = base_pos[2]

        # objective function (what do we want to minimize?) 
        f1 = 0 # TODO

        out["F"] = [f1]


# Define problem
problem = HoppingProblem()

# Define algorithms and initial conditions (depends on your variable ranges you selected above!)
algorithm = CMAES(x0=np.array([0.,0.])) # TODO: change initial conditions

# Run optimization
res = minimize(problem,
               algorithm,
               ('n_iter', 20), # may need to increase number of iterations
               seed=1,
               verbose=True)

print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
print("Check your optimized variables in practical4_hopping.py")