import copy
import logging
import random

import cv2
import hydra
import os
import time
import torch
from omegaconf import DictConfig

import logging
from logger import Logger
from models.PPOLSTMAgent import PPOLSTMAgent
from models.RuleBasedAgent import *
from models.CPCAgent import *
from models.utils.RolloutStorage import RolloutStorage
from models.utils.ManagerReplayStorage import ManagerReplayStorage
from recorder import VideoRecorder
from tocenv.env import *
from utils.logging import *
import numpy as np
from utils.svo import svo
from utils.observation import ma_obs_to_numpy

logger = logging.getLogger(os.path.basename(__file__))



class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'Workspace: {self.work_dir}')

        self.cfg = cfg

        # set_seed_everywhere(cfg.seed)

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.ra_agent.name)
        self.preferences = list(eval(self.cfg.svo))
        prefer = ['green', 'purple', 'blue', 'orange']

        self.num_agent = len(prefer)
        self.env = TOCEnv(agents=prefer,
                          map_size=(cfg.env.width, cfg.env.height),
                          episode_max_length=cfg.env.episode_length,
                          apple_spawn_ratio=cfg.env.apple_spawn_ratio,
                          )

        self.device = torch.device(cfg.device)
        self.env.reset()

        cfg.ra_agent.obs_dim = self.env.observation_space.shape
        print(cfg.ra_agent.obs_dim)
        cfg.ra_agent.action_dim = self.env.action_space.n

        cfg.ma_agent.obs_dim = (1, 256, 256, 3)
        cfg.ma_agent.action_dim = 5

        try:
            cfg.ra_agent.seq_len = self.env.episode_length
        except:
            pass

        try:
            cfg.ma_agent.seq_len = self.env.episode_length
        except:
            pass

        self.ra_agent = hydra.utils.instantiate(cfg.ra_agent)
        self.ma_agent = hydra.utils.instantiate(cfg.ma_agent)

        if type(self.ra_agent) in [CPCAgentGroup]:
            self.ra_replay_buffer = RolloutStorage(agent_type='ac',
                                                num_agent=self.num_agent,
                                                num_step=cfg.env.episode_length,
                                                batch_size=cfg.ra_agent.batch_size,
                                                num_obs=(self.ra_agent.obs_dim[1], self.ra_agent.obs_dim[2], 3),
                                                num_action=8,
                                                num_rec=128)

        if type(self.ma_agent) in [CPCAgentGroup]:
            self.ma_replay_buffer = RolloutStorage(agent_type='ac',
                                                num_agent=1,
                                                num_step=cfg.env.episode_length,
                                                batch_size=cfg.ra_agent.batch_size,
                                                num_obs=(self.ma_agent.obs_dim[1], self.ma_agent.obs_dim[2], 3),
                                                num_action=5,
                                                num_rec=128)
        self.writer = None

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        # self.video_recorder_blue = VideoRecorder(self.work_dir if cfg.save_video else None)
        # self.video_recorder_red = VideoRecorder(self.work_dir if cfg.save_video else None)

        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        average_ma_reward = 0

        self.video_recorder.init(enabled=True)

        for episode in range(self.cfg.num_eval_episodes):
            obs, _ = self.env.reset()
            episode_step = 0

            done = False
            episode_reward = 0

            arr_ma_obs = []
            epi_ma_reward = 0

            while not done:

                if type(self.ra_agent) in [RuleBasedAgent, RuleBasedAgentGroup]:
                    obs = self.env.get_numeric_observation()

                if type(self.ra_agent) is CPCAgentGroup:
                    action, cpc_info = self.ra_agent.act(self.ra_replay_buffer, obs, episode_step, sample=True)
                else:
                    action = self.ra_agent.act(obs, sample=True)

                obs, rewards, dones, _ = self.env.step(action)

                done = True in dones
                if episode_step == self.env.episode_length:
                    done = True

                ma_obs = self.env.render(coordination=False)
                self.video_recorder.record(self.env)

                arr_ma_obs.append(ma_obs)

                if len(arr_ma_obs) == self.cfg.ma_agent_action_interval:

                    ma_obs = ma_obs_to_numpy(arr_ma_obs)

                    # MA reward shaping
                    ma_reward = sum(rewards)
                    epi_ma_reward += ma_reward


                    if type(self.ma_agent) is PPOLSTMAgent:
                        ma_action = self.ma_agent.act(ma_obs, store_action=False)
                    else:
                        ma_action = self.ma_agent.act(ma_obs)

                    self.env.punish_agent(ma_action)

                    arr_ma_obs.clear()

                episode_reward += sum(rewards)
                episode_step += 1

            average_episode_reward += episode_reward
            average_ma_reward += epi_ma_reward

        self.video_recorder.save(f'{self.step}.mp4')

        if self.cfg.save_model:
            self.ra_agent.save(self.step)

        average_episode_reward /= self.cfg.num_eval_episodes
        average_ma_reward /= self.cfg.num_eval_episodes

        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.log('eval/ma_reward', average_ma_reward, self.step)
        self.logger.dump(self.step)

    def run(self):

        episode, episode_step, episode_reward, done = 1, 0, 0, True
        start_time = time.time()
        env_info = None

        ''' MA variables '''
        prev_ma_obs = None
        arr_ma_obs = []
        ma_action = 0
        ma_reward = 0
        self.env.set_apple_color_ratio(ma_action) # Initial MA action

        while self.step < self.cfg.num_train_steps + 1:

            if done or self.step % self.cfg.eval_frequency == 0:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))
                    if hasattr(self, 'ra_replay_buffer'):
                        self.ra_agent.train(self.ra_replay_buffer, self.logger, self.step)
                    if hasattr(self, 'ma_replay_buffer'):
                        self.ma_agent.train(self.ma_replay_buffer, self.logger, self.step)

                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode - 1, self.step)
                    self.evaluate()
                    start_time = time.time()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                ''' Log Environment Statistics '''

                if env_info:
                    log_statistics_to_writer(self.logger, self.step, env_info['statistics'])
                    log_agent_to_writer(self.logger, self.step, env_info['agents'])

                self.logger.log('train/episode', episode, self.step)

                obs, env_info = self.env.reset()

                episode_reward = 0
                episode_step = 0
                episode += 1

                ma_reward = 0

            if type(self.ra_agent) in [RuleBasedAgent, RuleBasedAgentGroup]:
                obs = self.env.get_numeric_observation()
            '''RA actions'''
            if self.step < self.cfg.num_seed_steps:
                # Define random actions
                if type(self.ra_agent) is CPCAgentGroup:
                    print('RA : ', obs.shape)
                    action, cpc_info = self.ra_agent.act(self.ra_replay_buffer, obs, episode_step, sample=True)
                else:
                    action = self.ra_agent.act(obs, sample=True)
            else:
                if type(self.ra_agent) is CPCAgentGroup:
                    action, cpc_info = self.ra_agent.act(self.ra_replay_buffer, obs, episode_step, sample=False)
                else:
                    action = self.ra_agent.act(obs, sample=False)

            #self.env.set_apple_color_ratio(random.random())

            next_obs, rewards, dones, env_info = self.env.step(action)
            ma_obs = self.env.render(coordination=False)
            ma_obs_in = np.expand_dims(ma_obs, axis=0)
            '''MA action'''
            if self.step < self.cfg.num_seed_steps:
                # Define random actions
                if type(self.ma_agent) is CPCAgentGroup:
                    ma_action, ma_cpc_info = self.ma_agent.act(self.ma_replay_buffer, ma_obs_in, episode_step, sample=True)
                else:
                    ma_action = self.ma_agent.act(ma_obs_in, sample=True)
            else:
                if type(self.ma_agent) is CPCAgentGroup:
                    ma_action, ma_cpc_info = self.ma_agent.act(self.ma_replay_buffer, ma_obs_in, episode_step, sample=False)
                else:
                    ma_action = self.ma_agent.act(ma_obs_in, sample=False)

            if self.cfg.render:
                cv2.imshow('TOCEnv', ma_obs)
                cv2.waitKey(1)

            done = True in dones
            if episode_step >= self.env.episode_length:
                done = True

            episode_reward += sum(rewards)
            modified_rewards = np.zeros(self.num_agent)
            # Applying prosocial SVO
            for i in range(self.num_agent):
                modified_rewards[i] = svo(rewards, i, self.preferences)
            if type(self.ra_agent) in [CPCAgentGroup]:
                self.ra_replay_buffer.add(obs, action, modified_rewards, dones, cpc_info)
            # If This is episode's first step, add nothing
            if episode_step == 0:
                ma_reward = np.zeros((1, 1))
            else:
                ma_reward =  np.reshape(np.sum(rewards),(-1, 1))
            if type(self.ma_agent) in [CPCAgentGroup]:
                self.ma_replay_buffer.add(ma_obs_in, ma_action[0], ma_reward, dones, ma_cpc_info)

            # logger.info('MA Agent Acted - {0}'.format(ma_action))
            self.env.punish_agent(ma_action[0])

            if self.cfg.render:
                ma_obs = self.env.render(coordination=False)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path="config", config_name="train_env")
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
