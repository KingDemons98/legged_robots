# Inverse Kinematics practical
from env.leg_gym_env import LegGymEnv
import numpy as np

def jacobian_rel(q,l1=0.209,l2=0.195):
    """ Jacobian based on relative angles (like URDF)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    # Jacobian 
    J = np.zeros((2,2))
    J[0,0] = -l1 * np.cos(q[0]) - l2 * np.cos(q[0] + q[1])
    J[1,0] =  l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    J[0,1] = -l2 * np.cos(q[0] + q[1]) 
    J[1,1] =  l2 * np.sin(q[0] + q[1])
    
    # foot pos
    pos = np.zeros(2)
    pos[0] =  -l1 * np.sin(q[0]) - l2 * np.sin(q[0]+q[1])
    pos[1] =  -l1 * np.cos(q[0]) - l2 * np.cos(q[0]+q[1])

    return J, pos

def pseudoInverse(A,lam=0.001):
    """ Pseudo inverse of matrix A. 
        Make sure to take into account dimensions of A
            i.e. if A is mxn, what dimensions should pseudoInv(A) be if m>n 
        Also take into account potential singularities
    """
    m,n = np.shape(A)
    A_T = np.transpose(A)
    if m >= n and np.linalg.matrix_rank(A) == n:
        pinvA = np.matmul(np.linalg.inv(np.matmul(A_T, A) + lam**2 * np.identity(n)), A_T)
    elif m <= n and np.linalg.matrix_rank(A) == m:
        pinvA = np.matmul(A_T, np.linalg.inv(np.matmul(A, A_T) + lam**2 * np.identity(m)))
    else:
        print("problem with pseudo-inverse")
    return pinvA

def ik_geometrical(xz,angleMode="<",l1=0.209,l2=0.195):
    """ Inverse kinematics based on geometrical reasoning.
        Input: Desired foot xz position (array) 
               angleMode (whether leg should look like > or <) 
               link lengths
        return: joint angles
    """
    q = np.zeros(2)
    if angleMode == "<":
        q[1] = -np.arccos((-l1**2 - l2**2 + xz[0]**2 + xz[1]**2)/(2 * l1 * l2))
    elif angleMode == ">":
        q[1] = np.arccos((-l1 ** 2 - l2 ** 2 + xz[0] ** 2 + xz[1] ** 2) / (2 * l1 * l2))
    q[0] = np.arctan2(-xz[0], -xz[1]) - np.arctan2(l2 * np.sin(q[1]), (l1 + l2 * np.cos(q[1])))
    return q

def ik_numerical(q0,des_x,tol=1e-4):
    """ Numerical inverse kinematics
        Input: initial joint angle guess, desired end effector, tolerance
        return: joint angles
    """
    i = 0
    max_i = 100 # max iterations
    alpha = 0.5 # convergence factor
    lam = 0.001 # damping factor for pseudoInverse
    joint_angles = q0
    ee_error = 1

    # Condition to iterate: while fewer than max iterations, and while error is greater than tolerance
    while( i < max_i and np.linalg.norm(ee_error) > tol ):
        # Evaluate Jacobian based on current joint angles
        J, ee = jacobian_rel(q = joint_angles)

        # Compute pseudoinverse
        J_pinv = pseudoInverse(A = J, lam = lam)

        # Find end effector error vector
        ee_error = des_x - ee

        # update joint_angles
        joint_angles += alpha * np.matmul(J_pinv, ee_error)

        # update iteration counter
        i += 1

    return joint_angles


env = LegGymEnv(render=True, 
                on_rack=True,    # set True to debug
                motor_control_mode='TORQUE',
                action_repeat=1,
                )
dt = 1000
NUM_STEPS = 50 * dt  # simulate 5 seconds (sim dt is 0.001)
tau = np.zeros(2) # either torques or motor angles, depending on mode

# IK_mode = "GEOMETRICAL"
IK_mode = "NUMERICAL"

# sample joint PD gains
kpJoint = np.array([55,55])
kdJoint = np.array([0.8,0.8])

# desired foot position (sample)
des_foot_pos = np.array([0.1,-0.2])
qdes = -env._robot_config.INIT_MOTOR_ANGLES

for counter in range(NUM_STEPS):
    # Compute inverse kinematics in leg frame 
    if IK_mode == "GEOMETRICAL":
        # geometrical
        qdes = ik_geometrical(xz= des_foot_pos, angleMode= ">") # ik_geometrical
    else:
        # numerical
        qdes = ik_numerical(q0 = qdes , des_x = des_foot_pos) # ik_numerical
    
    # print 
    if counter % 500 == 0:
        J, ee_pos_legFrame = jacobian_rel(env.robot.GetMotorAngles())
        print('---------------', counter)
        print('q ik',qdes,'q real',env.robot.GetMotorAngles())
        print('ee pos',ee_pos_legFrame)

    # determine torque with joint PD
    tau = kpJoint * (qdes - env.robot.GetMotorAngles()) + kdJoint * (0 - env.robot.GetMotorVelocities())
    # tau = np.matmul(kpJoint, (qdes - env.robot.GetMotorAngles())) + np.matmul(kdJoint, -env.robot.GetMotorVelocities())

    # apply control, simulate
    env.step(tau)
