import numpy as np
import gymnasium
from time import sleep

from agents import examples

# Run the environments/__init__.py to let python know about our custom environments
import environments

def test(
        env_id="GridWorldSparce",
        render=True,
        n_episodes=5,
        seed=42
):  
    env = gymnasium.make(env_id, render=False, size=10, max_episode_steps=500, seed=seed)
    agent = examples.SoftQlearning(env)

    n_avg = 100
    avg_score = np.full(n_avg, fill_value=np.nan, dtype=np.float32)

    show_off = False

    for i in range(n_episodes):
        if i == n_episodes - 5:
            env.close()
            env = gymnasium.make(env_id, render=True, size=10, max_episode_steps=500, seed=seed)
            show_off = True
        done = False
        episode_reward = 0
        observation, _ = env.reset()
        while not done:
            action = agent.make_decision(observation, explore=not show_off)
            if show_off:
                sleep(0.2)

            next_observation, reward, done, timeout, _ = env.step(action)
            done = done or timeout
            episode_reward += reward
            agent.learn(observation, action, reward, next_observation)
            observation = next_observation

        avg_score[i % n_avg] = episode_reward
        print(f"Episode {i+1}, mean score: {np.nanmean(avg_score)}")
        if render:
            sleep(1)


if __name__ == '__main__':
    env = "Lunar-explorer"
    n_episodes = 500
    render = False
    seed = 43

    test(env, render, n_episodes, seed)