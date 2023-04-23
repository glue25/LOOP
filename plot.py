import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_trajectory(env, dynamics):
    obs, done, done_episode = env.reset(), False, False

    obs_list = []
    _obs_list = []
    rew_list = []
    _rew_list = []

    _obs = obs
    obs_list.append(np.mean(obs))
    _obs_list.append(np.mean(_obs))
    done = torch.zeros(1,1)
    i = 0

    while done == False and i < 16:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print('next_obs')
        print(next_obs)
        print('rew')
        print(reward)

        # _next_obs, _rew, terminals, info = dynamics.get_forward_prediction(_obs, action)
        _sample = dynamics.get_forward_prediction_random_ensemble(_obs, action)
        _reward = _sample[0,:1]
        _next_obs = _sample[0,1:]
        print('fake_naxt_obs')
        print(_next_obs)
        obs = next_obs
        _obs = _next_obs

        i += 1
        obs_list.append(np.mean(obs))
        _obs_list.append(np.mean(_obs))
        rew_list.append(np.mean(reward))
        _rew_list.append(np.mean(_reward))


    episodes_list = list(range(len(obs_list)))
    rew_epi_list = list(range(len(rew_list)))

    plt.plot(episodes_list, obs_list,label='real_state')
    plt.plot(episodes_list, _obs_list,label='fake_state')
    plt.xlabel('steps')
    plt.ylabel('state_avg')
    plt.title('state trajectory - train using grad')
    plt.legend()
    plt.show()

    plt.plot(rew_epi_list, rew_list,label='real_reward')
    plt.plot(rew_epi_list, _rew_list,label='fake_reward')
    plt.xlabel('steps')
    plt.ylabel('rew_avg')
    plt.title('reward trajectory - train using grad')
    plt.legend()
    plt.show()