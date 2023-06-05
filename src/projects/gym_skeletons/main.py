import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), "./agents"))

from agents import gym_env
from agents import multi_gym_env

import numpy as np

from agents.algorithms.qlearning import QLearning
from agents.algorithms.qlearning_arithmetic_eps import QLearningAritEps
from agents.algorithms.sarsa import Sarsa
from agents.algorithms.randAlgo import RandAlgo
from agents.algorithms.dyna_q import GreedyQIteration
from agents.algorithms.deep_rl.test_dqn import DqnAlgorithm

#from agents.algorithms.elitra import QLearningEliTra

from agents.enums.environments import Environments
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
    epislon_decay = 0.99995
    alpha = 0.6
    gamma = 0.95
    dynaQ_update_per_iteration = 64

    #Init algos so that we can quickly switch
    qLearning = QLearning(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
    qLearningArEps = QLearningAritEps(epsilon=epsilon, alpha=alpha, gamma=gamma, k=5000)
    #qLearningEliTra = QLearningEliTra(env=env, discount=alpha, learning_rate=gamma, eps=1., eps_decay=0.999)
    sarsa = Sarsa(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
    greedyQIteration = GreedyQIteration(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay, update_per_iteration=dynaQ_update_per_iteration)
    dqn = DqnAlgorithm(terrain_size=10, k = 2500, epsilon=epsilon, gamma=gamma, lr=1e-3, hidden_layer_neurons=500, batch_size=32, tau=0.01, target_update_episodes=20)

    # selected algo
    algo = qLearningArEps

    #Training
    iterations, rewards, epsilons = gym_env.start(algo, env, render=False, max_episodes=training_episodes, seed=seed)

    fig, axes = plt.subplots(3)

    #Plot iterations
    axes[0].plot([x for x in range(len(iterations))], iterations)
    axes[0].set_title("Iterations")

    #Plot epsilon values<
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

def multi_train():
    env = Environments.LUNAR_EXPLORER_FOV

    #Hyper-parameters
    training_episodes = 2500
    epsilon = 1.0
    epislon_decay = 0.99995
    alpha = 0.25
    gamma = 0.95
    dynaQ_update_per_iteration = 64

    fov_on_player = True

    #Init algos so that we can quickly switch
    qLearning = QLearning(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
    qLearningArEps = QLearningAritEps(epsilon=epsilon, alpha=alpha, gamma=gamma, k=5000)
    #qLearningEliTra = QLearningEliTra(env=env, discount=alpha, learning_rate=gamma, eps=1., eps_decay=0.999)
    sarsa = Sarsa(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
    greedyQIteration = GreedyQIteration(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay, update_per_iteration=dynaQ_update_per_iteration)
    dqn = DqnAlgorithm(terrain_size=10, k = 7500, epsilon=epsilon, gamma=gamma, lr=1e-3, hidden_layer_neurons=500, batch_size=128, tau=0.001, target_update_episodes=20)

    # selected algo
    algo = dqn

    #Training
    iterations, rewards, epsilons = multi_gym_env.train(algo, env, max_episodes=training_episodes, fov_on_player=fov_on_player)

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

    for _ in range(50):
        seed = randint(0, 10000)
        gym_env.start(algo, env, render=True, max_episodes=1, init=False, seed=seed, fov_on_player=fov_on_player)
    

def test_q():
    #Taken for both "learning" phase and "showing off" phase
    env = Environments.LUNAR_EXPLORER

    #Hyper-parameters
    training_episodes = 2500
    epsilon = 1.0
    min_epsilon = 0.05
    k = 5000

    alphas = [i/10 for i in range(1,11,1)] 
    gammas = [i/10 for i in range(10, 0, -1)]

    alpha_gamma_matrix = []

    seed = randint(0, 1e6)

    tot = len(alphas) * len(gammas)
    curr = 0

    for i in range(len(alphas)):
        alpha_gamma_at_i_j = list()
        for j in range(len(gammas)):
            print(f"{curr}/{tot}")
            curr += 1
            alpha, gamma = alphas[i], gammas[j]
            # selected algo
            algo = QLearningAritEps(epsilon=epsilon, min_epsilon=min_epsilon, alpha=alpha, gamma=gamma, k=k)

            #Training
            iterations, rewards, epsilons = gym_env.start(algo, env, render=False, max_episodes=training_episodes, seed=seed)

            alpha_gamma_at_i_j.append(np.mean(rewards))

        alpha_gamma_matrix.append(alpha_gamma_at_i_j)

    figure = plt.figure()
    axes = figure.add_subplot(111)
    
    caxes = axes.matshow(alpha_gamma_matrix)


    for i in range(len(alpha_gamma_matrix)):
        for j in range(len(alpha_gamma_matrix[0])):
            plt.text(j, i, str(alpha_gamma_matrix[i][j]), ha='center', va='center', color='white')
    
    plt.xticks(list(range(len(alphas))), alphas)
    plt.yticks(list(range(len(gammas))), gammas)

    axes.set_xlabel("Alphas")
    axes.xaxis.set_label_position('top')
    axes.set_ylabel("Gamas")
    axes.yaxis.set_label_position('left')
    
    plt.show()

    print(seed)
    iterations, rewards, epsilons = gym_env.start(algo, env, render=True, max_episodes=training_episodes, seed=seed)

def test_k():
    #Taken for both "learning" phase and "showing off" phase
    env = Environments.LUNAR_EXPLORER

    #Hyper-parameters
    training_episodes = 2000
    epsilon = 1.0
    min_epsilon = 0.05
    alpha = 0.35
    gamma = 0.9

    Ks = list(range(0, 25001, 2500))

    seed = randint(0, 1e6)

    results = []

    for k in Ks:
        # selected algo
        algo = QLearningAritEps(epsilon=epsilon, min_epsilon=min_epsilon, alpha=alpha, gamma=gamma, k=k)

        #Training
        iterations, rewards, epsilons = gym_env.start(algo, env, render=False, max_episodes=training_episodes, seed=seed)
        results.append((f"k={k}", iterations, rewards, epsilons))

    fig, axes = plt.subplots(3)

    for name, iterations, rewards, epsilons in results:
        #Plot iterations
        axes[0].plot([x for x in range(len(iterations))], iterations, label=name)
        axes[0].set_title("Iterations")

        #Plot epsilon values<
        axes[1].plot([x for x in range(len(epsilons))], epsilons, label=name)
        axes[1].set_title("Epsilon decay")

        #Plot reward
        axes[2].plot([x for x in range(len(rewards))], rewards, label=name)
        axes[2].set_title("Summed reward")

    plt.legend()
    plt.tight_layout()
    plt.show()

def test_discrete():
    #Hyper-parameters
    training_episodes = 2000
    epsilon = 1.0
    epislon_decay = 0.9995
    alpha = 0.6
    gamma = 0.90

    for env in Environments:
        seed = randint(0, 1e6)

        for algo_class in [QLearning, Sarsa]:
            algo = algo_class(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
            print()
            print(f"Testing algorithm {algo.__class__.__name__} in environment {env.name}")

            #Training
            metrics, epsilons = gym_env.start(algo, env, show=False, show_episode=True, max_episodes=training_episodes, seed=seed)

            #Plot training results
            iterations, rewards = zip(*metrics)

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

def multi_train():
    #Taken for both "learning" phase and "showing off" phase
    env = Environments.LUNAR_EXPLORER_FOV

    #Hyper-parameters
    training_episodes = 2500
    epsilon = 1.0
    epislon_decay = 0.99995
    alpha = 0.25
    gamma = 0.95
    dynaQ_update_per_iteration = 64

    fov_on_player = True

    #Init algos so that we can quickly switch
    qLearning = QLearning(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
    qLearningArEps = QLearningAritEps(epsilon=epsilon, alpha=alpha, gamma=gamma, k=5000)
    #qLearningEliTra = QLearningEliTra(env=env, discount=alpha, learning_rate=gamma, eps=1., eps_decay=0.999)
    sarsa = Sarsa(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay)
    greedyQIteration = GreedyQIteration(epsilon=epsilon, alpha=alpha, gamma=gamma, epsilon_decay=epislon_decay, update_per_iteration=dynaQ_update_per_iteration)
    dqn = DqnAlgorithm(terrain_size=10, k = 7500, epsilon=epsilon, gamma=gamma, lr=1e-3, hidden_layer_neurons=500, batch_size=128, tau=0.001, target_update_episodes=20)

    # selected algo
    algo = dqn

    #Training
    iterations, rewards, epsilons = multi_gym_env.train(algo, env, max_episodes=training_episodes, fov_on_player=fov_on_player)

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

    for _ in range(50):
        seed = randint(0, 10000)
        gym_env.start(algo, env, render=True, max_episodes=1, init=False, seed=seed, fov_on_player=fov_on_player)

def test_dqn():
    env = Environments.LUNAR_EXPLORER
    seed = randint(0, 10000)

    #Hyper-parameters
    training_episodes = 2500
    epsilon = 1.0
    gamma = 0.90

    algo = DqnAlgorithm(terrain_size=10, k = 7500, epsilon=epsilon, gamma=gamma, lr=1e-3, hidden_layer_neurons=64, batch_size=128, tau=0.01, target_update_episodes=20)

    #Training
    iterations, rewards, epsilons = gym_env.start(algo, env, render=False, max_episodes=training_episodes, seed=seed)

    fig, axes = plt.subplots(3)

    #Plot iterations
    axes[0].plot([x for x in range(len(iterations))], iterations)
    axes[0].set_title("Iterations")

    #Plot epsilon values<
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
    iterations, rewards, epsilons = gym_env.start(algo, env, render=False, max_episodes=100, init=False, seed=seed)

    fig, axes = plt.subplots(3)

    #Plot iterations
    axes[0].plot([x for x in range(len(iterations))], iterations)
    axes[0].set_title("Iterations")

    #Plot epsilon values<
    axes[1].plot([x for x in range(len(epsilons))], epsilons)
    axes[1].set_title("Epsilon decay")

    #Plot reward
    axes[2].plot([x for x in range(len(rewards))], rewards)
    axes[2].set_title("Summed reward")

    plt.tight_layout()
    plt.show()

def test_dqn_updates():
    env = Environments.LUNAR_EXPLORER_CONTINUOUS
    seed = randint(0, 10000)

    #Hyper-parameters
    training_episodes = 2500
    epsilon = 1.0
    gamma = 0.90

    soft_updates_ratio = [0.0, 0.1, 0.2, 0.75, 1.0]
    
    fig, axes = plt.subplots(2)

    for update_ratio in soft_updates_ratio:

        algo = DqnAlgorithm(terrain_size=10, k = 7500, epsilon=epsilon, gamma=gamma, lr=1e-3, hidden_layer_neurons=64, batch_size=128, tau=update_ratio, target_update_episodes=20)

        #Training
        iterations, rewards, epsilons = gym_env.start(algo, env, render=False, max_episodes=training_episodes, seed=seed)

        N = 7
        #Plot iterations
        rolling_iter = np.convolve(iterations, np.ones(N)/N, mode='valid') 
        axes[0].plot([x for x in range(len(rolling_iter))], rolling_iter, label=f"update_ratio_{update_ratio}")
        axes[0].set_title("Iterations")

        #Plot reward
        rolling_rew = np.convolve(rewards, np.ones(N)/N, mode='valid') 
        axes[1].plot([x for x in range(len(rolling_rew))], rolling_rew, label=f"update_ratio_{update_ratio}")
        axes[1].set_title("Summed reward")

    plt.legend()
    plt.tight_layout()
    plt.show()

def test_dqn_normalization():
    env = Environments.LUNAR_EXPLORER_CONTINUOUS
    seed = randint(0, 10000)

    #Hyper-parameters
    training_episodes = 2500
    epsilon = 1.0
    gamma = 0.90
    
    fig, axes = plt.subplots(3)
    plt.title("Comparison of normalization of the x,y coordinates across DQN networks")

    for terrain_size in [1,10,1e6]:
        print(f"Terrain size: {terrain_size}")
        algo = DqnAlgorithm(terrain_size=terrain_size, k = 7500, epsilon=epsilon, gamma=gamma, lr=1e-3, hidden_layer_neurons=64, batch_size=128, tau=0.001, target_update_episodes=20)

        #Training
        iterations, rewards, epsilons = gym_env.start(algo, env, render=False, max_episodes=training_episodes, seed=seed)

        #Plot iterations
        axes[0].plot([x for x in range(len(iterations))], iterations, label=f"size_{terrain_size}")
        axes[0].set_title("Iterations")

        #Plot epsilon values<
        axes[1].plot([x for x in range(len(epsilons))], epsilons, label=f"size_{terrain_size}")
        axes[1].set_title("Epsilon decay")

        #Plot reward
        axes[2].plot([x for x in range(len(rewards))], rewards, label=f"size_{terrain_size}")
        axes[2].set_title("Summed reward")

    plt.legend()
    plt.tight_layout()
    plt.show()


def test_envs():
    """Test the discrete and continuous environments, make DQN and Q-learning compete"""
    for env in [Environments.LUNAR_EXPLORER, Environments.LUNAR_EXPLORER_CONTINUOUS]:
        plt.title(f"comparison of DQNs and Q-learning algorithm runing on environment {env}")
        #Hyper-parameters
        training_episodes = 2000
        epsilon = 1.0
        gamma = 0.90
        alpha = 0.3
        k  = 7500
        
        fig, axes = plt.subplots(3)

        seeds = [randint(0, 10000), randint(0, 10000), randint(0, 10000)]
        
        for i, seed in enumerate(seeds):
            dqn = DqnAlgorithm(terrain_size=10, k = k, epsilon=epsilon, gamma=gamma, lr=1e-3, hidden_layer_neurons=64, batch_size=64, tau=0.001, target_update_episodes=20)
            qLearningArEps = QLearningAritEps(epsilon=epsilon, min_epsilon=0.05, alpha=alpha, gamma=gamma, k=k)
            for algo in [dqn, qLearningArEps]:
                #Training
                iterations, rewards, epsilons = gym_env.start(algo, env, render=False, max_episodes=training_episodes, seed=seed)
                N = 9

                #Plot iterations
                iterations = np.convolve(iterations, np.ones(N)/N, mode='valid') 
                axes[0].plot([x for x in range(len(iterations))], iterations, label=f"{algo.__class__.__name__}_{i}")
                axes[0].set_title("Iterations")

                #Plot epsilon values
                axes[1].plot([x for x in range(len(epsilons))], epsilons, label=f"{algo.__class__.__name__}_{i}")
                axes[1].set_title("Epsilon decay")

                #Plot reward
                rewards = np.convolve(rewards, np.ones(N)/N, mode='valid') 
                axes[2].plot([x for x in range(len(rewards))], rewards, label=f"{algo.__class__.__name__}")
                axes[2].set_title("Summed reward")

        plt.legend()
        plt.tight_layout()
        plt.show()

def test_multi_training():
    env = Environments.LUNAR_EXPLORER_FOV

    fig, axes = plt.subplots(3)
    plt.title("Training of the DQN using FoV for generalization purposes")

    from typing import List, Tuple
    trained_algos: List[Tuple[str, bool, DqnAlgorithm]] = [] #Contain the triple (algo_name, fov, algo)

    for fov_on_player in [True, False]:
        algo_name = f"FoV_{'relative' if fov_on_player else 'gobal'}"
        #Hyper-parameters
        training_episodes = 2000
        epsilon = 1.0
        gamma = 0.95

        #Init algos so that we can quickly switch
        algo = DqnAlgorithm(terrain_size=10, k = 7000, epsilon=epsilon, gamma=gamma, lr=1e-3, hidden_layer_neurons=128, batch_size=128, tau=0.05, target_update_episodes=20)

        #Training
        iterations, rewards, epsilons = multi_gym_env.train(algo, env, max_episodes=training_episodes, fov_on_player=fov_on_player)

        N=9
        #Plot iterations
        iterations = np.convolve(iterations, np.ones(N)/N, mode='valid') 
        axes[0].plot([x for x in range(len(iterations))], iterations, label=f"{algo_name}")
        axes[0].set_title("Iterations")

        #Plot epsilon values
        axes[1].plot([x for x in range(len(epsilons))], epsilons, label=f"{algo_name}")
        axes[1].set_title("Epsilon decay")

        #Plot reward
        rewards = np.convolve(rewards, np.ones(N)/N, mode='valid') 
        axes[2].plot([x for x in range(len(rewards))], rewards, label=f"{algo_name}")
        axes[2].set_title("Summed reward")

        trained_algos.append((algo_name, fov_on_player, algo))

    plt.legend()
    plt.tight_layout()
    plt.show()
    
    import random
    seeds = random.sample(range(1, int(1e9)), 100)

    fig, axes = plt.subplots(3)
    plt.title("Fully trained DQN networks with FoV on new terrains")

    for algo_name, fov_on_player, algo in trained_algos:
        iterations, rewards, epsilons = [], [], []
        for seed in seeds:
            current_iterations, current_rewards, current_epsilons = gym_env.start(algo, env, render=False, max_episodes=1, init=False, seed=seed, fov_on_player=fov_on_player)
            iterations.extend(current_iterations)
            rewards.extend(current_rewards)
            epsilons.extend(current_epsilons)

        #Plot iterations
        iterations = np.convolve(iterations, np.ones(N)/N, mode='valid') 
        axes[0].plot([x for x in range(len(iterations))], iterations, label=f"{algo_name}")
        axes[0].set_title("Iterations")

        #Plot epsilon values<
        axes[1].plot([x for x in range(len(epsilons))], epsilons, label=f"{algo_name}")
        axes[1].set_title("Epsilon decay")

        #Plot reward
        rewards = np.convolve(rewards, np.ones(N)/N, mode='valid') 
        axes[2].plot([x for x in range(len(rewards))], rewards, label=f"{algo_name}")
        axes[2].set_title("Summed reward")

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # test_one()

    # test_all()

    # multi_train()

    ## Report test fct
    ## Q
    test_q()                      # Done
    
    test_k()                      # Done

    ## DQN
    test_dqn()                    # Done

    test_dqn_updates()            # Doing

    test_dqn_normalization()

    # ENV
    test_envs()

    test_multi_training()