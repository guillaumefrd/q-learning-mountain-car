import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gym import wrappers


class Agent:
    def __init__(self,
                 lr_init,
                 lr_min,
                 lr_decay_rate,
                 gamma,
                 epsilon,
                 epsilon_decay_rate,
                 num_bins):

        self.lr = lr_init
        self.lr_min = lr_min
        self.lr_decay_rate = lr_decay_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.num_bins = num_bins
        self.state = None
        self.action = None

        # turn the continuous state space into a discrete space (with bins)
        # for the two observations: car position and car velocity
        self.discrete_states = [
            np.linspace(-1.2, 0.6, num=(num_bins + 1))[1:-1],
            np.linspace(-0.07, 0.07, num=(num_bins + 1))[1:-1],
        ]

        # initialize the Q-table with zeros
        self.num_actions = 3
        num_states = self.num_bins ** len(self.discrete_states)
        self.q = np.zeros(shape=(num_states, self.num_actions))
        print("Q-table shape (number_states, number_actions):", self.q.shape)
        print("Discretize bins of state space:", self.discrete_states)

    def to_state(self, observation):
        # turn the observation features into a space represented by an integer
        state = sum(np.digitize(feature, self.discrete_states[i]) * (self.num_bins ** i)
                    for i, feature in enumerate(observation))
        return state

    def start_episode(self, observation):
        # apply decay on exploration
        self.epsilon *= (1 - self.epsilon_decay_rate)

        # apply decay on learning rate
        self.lr = max(self.lr_min, self.lr * (1 - self.lr_decay_rate))

        # return the first action of the episode
        self.state = self.to_state(observation)
        return np.argmax(self.q[self.state])

    def make_action(self, observation, reward):
        next_state = self.to_state(observation)

        if (1 - self.epsilon) <= np.random.uniform():
            # make a random action to explore
            next_action = np.random.randint(0, self.num_actions)
        else:
            # take the best action
            next_action = np.argmax(self.q[next_state])

        # update the Q-table
        self.q[self.state, self.action] += self.lr * \
            (reward + self.gamma * np.max(self.q[next_state, :]) -
             self.q[self.state, self.action])

        self.state = next_state
        self.action = next_action
        return next_action


class Monitor:
    def __init__(self,
                 num_episodes):

        self.num_episodes = num_episodes
        self.rewards = np.zeros(num_episodes, dtype=int)
        self.episode_plot = None
        self.avg_plot = None
        self.fig = None
        self.ax = None

    def __getitem__(self, episode_index):
        return self.rewards[episode_index]

    def __setitem__(self, episode_index, episode_reward):
        self.rewards[episode_index] = episode_reward

    def create_plot(self):
        plt.style.use("ggplot")
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.canvas.set_window_title("Episode Reward History")
        self.ax.set_xlim(0, self.num_episodes + 5)
        self.ax.set_ylim(-210, -110)
        self.ax.set_title("Episode Reward History")
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Total Reward")
        self.episode_plot, = plt.plot([], [], linewidth=0.5, alpha=0.5,
                                      c="#1d619b", label="reward per episode")
        self.avg_plot, = plt.plot([], [], linewidth=3.0, alpha=0.8, c="#df3930",
                                  label="average reward over the 200 last episodes")
        self.ax.legend(loc="upper left")

    def update_plot(self, episode_index):
        # update the episode plot
        x = range(episode_index)
        y = self.rewards[:episode_index]
        self.episode_plot.set_xdata(x)
        self.episode_plot.set_ydata(y)

        # update the average plot
        mean_kernel_size = 201
        rolling_mean_data = np.concatenate((np.full(mean_kernel_size, fill_value=-200),
                                            self.rewards[:episode_index]))
        rolling_mean_data = pd.Series(rolling_mean_data)

        rolling_means = rolling_mean_data.rolling(window=mean_kernel_size,
                                                  min_periods=0).mean()[mean_kernel_size:]
        self.avg_plot.set_xdata(range(len(rolling_means)))
        self.avg_plot.set_ydata(rolling_means)

        plt.draw()
        plt.pause(0.0001)


def videos_to_record(episode_id):
    return episode_id in [100, 500, 1000, 2000, 3500, 4900]


def main():

    # parameters
    verbose = False
    seed = 42
    working_dir = "q-learning_logs"
    num_episodes = 5000
    plot_redraw_frequency = 10

    # create the environment
    env = gym.make("MountainCar-v0")

    # set seed to reproduce the same results
    env.seed(seed)
    np.random.seed(seed)

    # monitor the training
    env = wrappers.Monitor(env, working_dir, force=True, resume=False,
                           video_callable=videos_to_record)

    agent = Agent(
        lr_init=0.3,
        lr_min=1e-5,
        lr_decay_rate=5e-4,
        gamma=0.98,
        epsilon=0.9,
        epsilon_decay_rate=5e-3,
        num_bins=15
    )

    monitor = Monitor(num_episodes=num_episodes)
    monitor.create_plot()

    for episode_index in range(num_episodes):
        observation = env.reset()
        action = agent.start_episode(observation)
        total_reward = 0
        timestep = 0
        done = False

        while not done:
            # make an action and get the new observations
            observation, reward, done, info = env.step(action)
            total_reward += reward
            timestep += 1

            if verbose:
                env.render()
                print("Timestep: {0:3d}, Action: {1:2d}, Reward: {2:5.1f}, Car \
                      position: {3:6.3f}, Car velocity: {4:6.3f}"
                      .format(timestep, action, reward, *observation))

            # compute the next action
            action = agent.make_action(observation, reward)

        print("Episode {} finished after {} timesteps (total reward: {})"
              .format(episode_index + 1, timestep, total_reward))

        # update the plot
        monitor[episode_index] = total_reward
        if verbose or episode_index % plot_redraw_frequency == 0:
            monitor.update_plot(episode_index)

    # save the history in a csv file
    df = pd.DataFrame(monitor.rewards, columns=["reward"])
    df.to_csv(os.path.join(working_dir, "history.csv"), index_label="episode")
    env.env.close()


if __name__ == "__main__":
    main()
