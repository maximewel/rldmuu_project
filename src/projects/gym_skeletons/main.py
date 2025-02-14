import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), "./qlearning_assignement"))

from qlearning_assignement import gym_env

from qlearning_assignement.algorithms.qlearning import QLearning
from qlearning_assignement.algorithms.qlearning_arithmetic_eps import QLearningAritEps
from qlearning_assignement.algorithms.sarsa import Sarsa
from qlearning_assignement.algorithms.randAlgo import RandAlgo
from qlearning_assignement.algorithms.dyna_q import GreedyQIteration
from qlearning_assignement.algorithms.deep_rl.test_dqn import DqnAlgorithm

from qlearning_assignement.algorithms.my_agent import QLearningEliTra

from qlearning_assignement.enums.environments import Environments
import matplotlib.pyplot as plt
from random import randint

# Run the environments/__init__.py to let python know about our custom environments
import environments

def test_one():
    #Taken for both "learning" phase and "showing off" phase
    env = Environments.LUNAR_EXPLORER
    seed = randint(0, 10000)

    #Hyper-parameters
    training_episodes = 2000
    epsilon = 1.0
    epislon_decay = 0.9999
    alpha = 0.6
    gamma = 0.9999
    dynaQ_update_per_iteration = 64

    #Init algos so that we can quickly switch
    qLearning = QLearning(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
    qLearningArEps = QLearningAritEps(epsilon=epsilon, alpha=alpha, gamma=gamma, k=5000)
    qLearningEliTra = QLearningEliTra(learning_rate = alpha, discount = gamma, eps_decay=epislon_decay)
    sarsa = Sarsa(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
    greedyQIteration = GreedyQIteration(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay, update_per_iteration=dynaQ_update_per_iteration)
    dqn = DqnAlgorithm(k = 10000, epsilon=epsilon, gamma=gamma, lr=1e-3, hidden_layer_neurons=32, batch_size=128, tau=0.10)

    # selected algo
    algo = qLearningEliTra

    #Training
    iterations, rewards, epsilons = gym_env.start(algo, env, render=False, show_episode=True, max_episodes=training_episodes, seed=seed)

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
    gym_env.start(algo, env, render=True, max_episodes=50, init=False, seed=seed)

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