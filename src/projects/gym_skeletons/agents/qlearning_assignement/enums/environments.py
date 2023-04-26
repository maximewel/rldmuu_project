from enum import Enum

class Environments(Enum):
    #Discrete states
    CLIFF = "CliffWalking-v0"
    TAXI = "Taxi-v3"
    FROZEN_LAKE = "FrozenLake-v1"

    #Continuous observations (states)
    LUNAR_LANDER = "LunarLander-v2"
    CARTPOLE = "CartPole-v1"


class Bounds(Enum):
    #Express bounds in term of (min, max, quantization)
    LUNAR_LANDER = [
        #positions
        (-90, 90, 36),
        (-90, 90, 36),
        #velocities
        (-5, 5, 10),
        (-5, 5, 10),
        #angle
        (-3.1415927, 3.1415927, 10),
        #angular velocity
        (-5, 5, 10),
        #Arms touching ground or not
        (0, 1, 2),
        (0, 1, 2)
    ]

    CARTPOLE = [
        #Cart position
        (-2.5, 2.5, 50), #0.2 incr. The episode ends at 2.4 / -2.4 so we keep that range
        #Cart velocity
        (-5, 5, 50),
        #Pole angle
        (-0.21, 0.21, 42),
        #Pole velocity
        (-5, 5, 50)
    ]