import gym_env
from algorithms.qlearning import QLearning
from algorithms.qlearning_arithmetic_eps import QLearningAritEps
from algorithms.sarsa import Sarsa
from algorithms.randAlgo import RandAlgo
from algorithms.dyna_q import GreedyQIteration
from algorithms.deep_rl.test_dqn import DqnAlgorithm
import gym_env
from enums.environments import Environments
import matplotlib.pyplot as plt
from random import randint
import os

def test_one():
    #Hyper-parameters
    training_episodes = 1000
    epsilon = 1.0
    epislon_decay = 0.999
    alpha = 0.6
    gamma = 0.99
    dynaQ_update_per_iteration = 64

    #Init algos so that we can quickly switch
    qLearning = QLearning(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
    qLearningArEps = QLearningAritEps(epsilon=epsilon, alpha=alpha, gamma=gamma, k=10000)
    sarsa = Sarsa(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
    greedyQIteration = GreedyQIteration(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay, update_per_iteration=dynaQ_update_per_iteration)
    dqn = DqnAlgorithm(k = 1000, epsilon=epsilon, gamma=gamma, lr=1e-2, hidden_layer_neurons=64, batch_size=128, tau=0.1)

    #Taken for both "learning" phase and "showing off" phase
    algo = dqn
    env = Environments.LUNAR_LANDER
    seed = randint(0, 10000)

    #Training
    metrics, epsilons = gym_env.start(algo, env, show=False, show_episode=True, max_episodes=training_episodes, seed=seed)

    #Plot training results
    iterations = [iterations for iterations, reward in metrics]
    rewards = [reward for iterations, reward in metrics]

    fig, axes = plt.subplots(3)

    #Plot iterations
    axes[0].plot([x for x in range(len(iterations))], iterations)
    axes[0].set_title("Iterations")

    #Plot epsilon values
    axes[1].plot([x for x in range(len(epsilons))], epsilons)
    axes[1].set_title("Epsilon decay")

    #Plot reward
    axes[2].plot([x for x in range(len(rewards))], rewards)
    axes[2].set_title("Summed reward")

    plt.tight_layout()
    plt.show()

    #SHOW OFF !
    algo.epsilon = 0
    algo.k = 0
    gym_env.start(algo, env, show=True, max_episodes=50, init=False, seed=seed)

def test_all():
    #Hyper-parameters
    training_episodes = 2000
    epsilon = 1.0
    epislon_decay = 0.9995
    alpha = 0.6
    gamma = 0.90

    for env in Environments:
        seed = randint(0, 10000)
        for algo_class in [QLearning, Sarsa]:
            algo = algo_class(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
            print()
            print(f"Testing algorithm {algo.__class__.__name__} in environment {env.name}")

            #Training
            metrics, epsilons = gym_env.start(algo, env, show=False, show_episode=True, max_episodes=training_episodes, seed=seed)

            #Plot training results
            iterations = [iterations for iterations, reward in metrics]
            rewards = [reward for iterations, reward in metrics]

            fig, axes = plt.subplots(3)

            #Plot iterations
            axes[0].plot([x for x in range(len(iterations))], iterations)
            axes[0].set_title("Iterations")

            #Plot epsilon values
            axes[1].plot([x for x in range(len(epsilons))], epsilons)
            axes[1].set_title("Epsilon decay")

            #Plot reward
            axes[2].plot([x for x in range(len(rewards))], rewards)
            axes[2].set_title("Summed reward")

            plt.tight_layout()
            filepath = os.path.join(os.path.abspath(''), 'out', f"{env.name}_{algo.__class__.__name__}.png")
            plt.savefig(filepath)
            plt.close()

if __name__ == "__main__":
    test_one()

    #test_all()