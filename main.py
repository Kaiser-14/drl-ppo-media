#!/usr/bin/env python

import time

import gym
import ppo
import tensorflow as tf

from datetime import datetime

import media_env


def main():
	env_name = 'VideoStreaming-v0'
	# env_name = 'CartPole-v1'

	# Logging information
	model_path = './data/' + env_name + '/'
	timestamp = datetime.now().strftime('%y%m%d_%H%M')
	writer = tf.summary.create_file_writer(model_path + 'summary/' + timestamp)

	env = gym.make(env_name)

	episode_length = 150  # Total training episodes
	update_frequency = 20  # Defined for updates on media environment
	best_reward = 0

	try:

		agent = ppo.Model(
			obs_dim=env.observation_space.shape[0], act_dim=env.action_space.n, lr=0.001,
			gamma=0.99, clip_range=0.2, update_ep_epochs=4)

		for episode in range(episode_length):
			obs_cur = env.reset()
			episode_reward = 0
			ep_steps = 0

			while True:
				action, logprob = agent.step(obs_cur)
				obs_nxt, reward, ep_done, _ = env.step(action)
				ep_steps += 1
				print('Step', ep_steps, 'Action:', action, 'Reward:', reward)
				print(obs_nxt)
				print('-------')

				agent.memory.store(obs_cur, action, reward, ep_done, logprob)

				obs_cur = obs_nxt
				episode_reward += reward
				print('Reward accumulated:', episode_reward)

				if ep_done or ep_steps >= update_frequency:
					agent.learn()
					print('Episode', episode, 'Reward:', episode_reward/ep_steps)
					# print('episode: %i' % episode, ", reward: %i" % episode_reward)

					with writer.as_default():
						tf.summary.scalar("episode_reward", episode_reward/ep_steps, step=episode)

					break

			if best_reward < episode_reward:
				best_reward = episode_reward
				print(f"Saving best model to {model_path}. Reward: {episode_reward}")
				agent.save(model_path + 'summary/' + timestamp + '/model')

		env.close()

		print(f"Episodes: {episode_length}")
		print(f"Best score: {best_reward}")

	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		env.close()


if __name__ == '__main__':
	print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

	tic = time.perf_counter()
	main()
	toc = time.perf_counter()
	print(f"Training finished in {toc - tic:0.4f} seconds")

	# Load model
	# agent.load('./data/VideoStreaming-v0/best_model')
