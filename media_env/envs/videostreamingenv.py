import random

import gym
from gym import spaces
import requests
from kafka import KafkaConsumer
import json
import time
import subprocess
from datetime import datetime
import csv

import numpy as np


class VideoStreamingEnv(gym.Env):
	"""
	A custom media_env for optimizing video streaming.
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(VideoStreamingEnv, self).__init__()

		# Action space: We assume 3 bitrate profiles (0: high, 1: medium, 2: low)
		self.action_space = spaces.Discrete(3)

		# Observation space includes various metrics from Kafka and vCompression (vce)
		self.observation_space = spaces.Dict({
			'current_bitrate': spaces.Box(low=0, high=30, shape=(1,), dtype=np.float16),
			'max_bitrate': spaces.Box(low=0, high=25, shape=(1,), dtype=np.float16),
			'cpu_usage': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float16),
			'ram_usage': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float16),
			'enc_quality': spaces.Box(low=0, high=69, shape=(1,), dtype=np.float16),
			'blockiness': spaces.Box(low=0, high=1.5, shape=(1,), dtype=np.float16),
			'block_loss': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float16),
			'blur': spaces.Box(low=0, high=70, shape=(1,), dtype=np.float16),
			'noise': spaces.Box(low=0, high=30, shape=(1,), dtype=np.float16),
			'temporal_activity': spaces.Box(low=0, high=20, shape=(1,), dtype=np.float16),
			'spatial_activity': spaces.Box(low=0, high=60, shape=(1,), dtype=np.float16),
			'exposure': spaces.Box(low=0, high=255, shape=(1,), dtype=np.float16),
			'contrast': spaces.Box(low=0, high=120, shape=(1,), dtype=np.float16),
			'throughput': spaces.Box(low=0, high=50, shape=(1,), dtype=np.float16),
			'packet_loss': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float16),
			'latency': spaces.Box(low=0, high=500, shape=(1,), dtype=np.float16)
		})

		observation_keys = list(self.observation_space.spaces.keys())

		self.observation_space = spaces.Box(
			low=np.array([
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0
			]),
			high=np.array([
				30, 25, 100, 100, 69, 1.5, 1000, 70, 30, 255, 270, 255, 120, 50, 100, 500, 50
			]),
			dtype=np.float16
		)

		# Address for services
		localhost = '192.168.1.48'

		# Logging information
		model_path = './data/VideoStreaming-v0/summary/'
		timestamp = datetime.now().strftime('%y%m%d_%H%M')
		self.csv_file = model_path + timestamp + '/logs.csv'
		self.csv_columns = ['timestamp', 'action', 'reward', 'done', 'background'] + observation_keys

		with open(self.csv_file, mode='w', newline='') as file:
			self.writer = csv.DictWriter(file, fieldnames=self.csv_columns)
			self.writer.writeheader()

		# Kafka consumers from video quality probe and network performance probe
		# self.consumer_ffmpeg = KafkaConsumer(
		# 	'ffmpeg', bootstrap_servers='192.168.1.47:9092', auto_offset_reset='latest',
		# 	enable_auto_commit=True, value_deserializer=lambda x: json.loads(x))
		self.consumer_vqp = KafkaConsumer(
			'vqp', bootstrap_servers=[localhost + ':9092'], auto_offset_reset='latest',
			enable_auto_commit=True, value_deserializer=lambda x: json.loads(x))
		# self.consumer_npp = KafkaConsumer(
		# 	'npp', bootstrap_servers=['192.168.1.47:9092'], auto_offset_reset='latest',
		# 	enable_auto_commit=True, value_deserializer=lambda x: json.loads(x))

		# API for virtual compression engine and network performance probe requests
		self.vce = 'http://' + localhost + ':3000/'
		self.vce_traffic = 'http://' + localhost + ':3001/'
		self.npp = 'http://' + localhost + ':5000/'

		# Parameters
		self.br_profiles = {0: 25000, 1: 15000, 2: 7000}
		self.br_background = [15000, 5000, 1000]
		self.bandwidth = 35

		# FFmpeg
		self.main_process = self._start_stream('2')
		self.back_process = self._start_stream('3')

		# Cold start time to start providing adequate video quality metrics
		time.sleep(15)

	def step(self, action):
		# Execute one time step within the media_env
		self._take_action(self.br_profiles[action])
		obs, reward, done = self._next_observation(action)
		return obs, reward, done, {}

	def reset(self):
		# Reset the state of the media_env to an initial state
		action = 0
		self._reset_stream(action)
		return self._next_observation(action)[0]

	def render(self):
		pass

	def close(self):
		self.consumer_vqp.close()
		# self.consumer_npp.close()

		self.main_process.terminate()
		self.back_process.terminate()

	def _take_action(self, action):
		# Update the vcompression settings via a REST API
		response = requests.post(self.vce + 'bitrate/' + str(action))
		if response.status_code != 200:
			print("Failed to update vCE settings")

	def _next_observation(self, _action):
		done = False
		# Get the current state from components. Wait seconds to stabilize bitrate change on streaming
		time.sleep(3)

		# Get current streaming data from vCE
		vce_data = requests.get(self.vce).json()

		# time.sleep(10)

		# Get data from video quality probe
		# msg_vqp = self.consumer_vqp.poll(15.0)
		for message in self.consumer_vqp:
			msg_vqp = message.value
			if float('inf') == msg_vqp['value']['blockiness'] or msg_vqp['value']['blockiness'] > 1.0:
				msg_vqp['value']['blockiness'] = 1.0
			break

		# Trigger new measure from network performance probe and get data from Kafka. Add 2 seconds to stabilize network
		msg_npp = requests.get(self.npp + '/measure').json()
		time.sleep(2)

		# Change bitrate from traffic simulator
		background = random.choices(self.br_background, weights=[0.1, 0.3, 0.5], k=1)[0]
		requests.post(self.vce_traffic + str(background))

		if msg_vqp is None or msg_npp is None:
			print("Error in Kafka consumption or no new messages.")
			return {}

		# Fill observation set
		observation = {
			'current_bitrate': round(vce_data['stats']['act_bitrate']['value'] / 1000, 2),
			'max_bitrate': round(vce_data['stats']['max_bitrate']['value'] / 1000, 2),
			'cpu_usage': round(vce_data['stats']['pid_cpu']['value'], 2),
			'ram_usage': round(vce_data['stats']['pid_ram']['value'] / (1024 ** 3), 2),  # Convert to GB
			'enc_quality': round(vce_data['stats']['enc_quality']['value'], 2),
			'blockiness': round(msg_vqp['value']['blockiness'], 2),
			'block_loss': round(msg_vqp['value']['block_loss'], 2),
			'blur': round(msg_vqp['value']['blur'], 2),
			'noise': round(msg_vqp['value']['noise'], 2),
			'temporal_activity': round(msg_vqp['value']['temporal_activity'], 2),
			'spatial_activity': round(msg_vqp['value']['spatial_activity'], 2),
			'exposure': round(msg_vqp['value']['exposure'], 2),
			'contrast': round(msg_vqp['value']['contrast'], 2),
			'throughput': round(msg_npp['measurements']['throughput']['value'], 2),
			'packet_loss': round(msg_npp['measurements']['packet_loss']['value'], 2),
			'latency': round(msg_npp['measurements']['latency']['value'], 2),
		}

		_reward = self._calculate_reward(observation, _action)

		# Check if main video still running
		if self.main_process.poll() is not None:
			self.main_process = self._start_stream('2')
			done = True

		log_data = {
			'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
			'action': _action,
			'reward': _reward,
			'done': done,
			'background': str(background),
			**observation,
		}
		with open(self.csv_file, mode='a', newline='') as file:
			writer = csv.DictWriter(file, fieldnames=self.csv_columns)
			writer.writerow(log_data)

		return np.array(list(observation.values())), _reward, done

	@staticmethod
	def _calculate_reward(observation, _action):
		# reward = observation['mos'] * 2

		reward = 0

		penalty = 0

		# Reward by image quality

		# Blockiness
		if observation['blockiness'] > 0.8:
			blockiness_reward = 20 * observation['blockiness']
		elif observation['blockiness'] > 0.6:
			blockiness_reward = 10 * observation['blockiness']
		else:
			blockiness_reward = 0

		# Block loss
		blockloss_reward = max(2*(5-observation['block_loss']), -50)

		# Profiles
		if observation['blockiness'] > 0.7 and observation['block_loss'] < 5.0:
			# if observation['packet_loss'] < 10.0:
			if _action == 0:
				profile_reward = 10
			elif _action == 1:
				profile_reward = 5
			else:
				profile_reward = 0
		else:
			profile_reward = 0

		reward += blockiness_reward + blockloss_reward + profile_reward

		return reward

	def _reset_stream(self, action):
		# Reset the stream before starting a new episode
		response = requests.post(self.vce + 'bitrate/' + str(action))
		if response.status_code != 200:
			print("Failed to reset vCE settings")
		else:
			time.sleep(4)

	@staticmethod
	def _start_stream(idx):
		# Start FFmpeg process to launch local video
		if idx == '2':
			cmd = [
				'ffmpeg', '-re', '-i', 'video.mkv', '-c:v', 'copy', '-c:a', 'copy',
				'-f', 'mpegts', 'udp://172.17.0.' + idx + ':1234'
			]
		elif idx == '3':
			cmd = [
				'ffmpeg', '-re', '-stream_loop', '-1', '-i', 'video.mkv', '-c:v', 'copy', '-c:a', 'copy',
				'-f', 'mpegts', 'udp://172.17.0.' + idx + ':1234'
			]
		else:
			pass
		# Launch FFmpeg as a subprocess and return the process handle
		return subprocess.Popen(cmd)
