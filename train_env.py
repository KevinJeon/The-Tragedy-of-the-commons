import cv2
import hydra
import os
import time
import torch
from omegaconf import DictConfig

from logger import Logger
from models.RuleBasedAgent import *
from models.CPCAgent_test import *
from models.utils.RolloutStorage import RolloutStorage
from recorder import VideoRecorder
from tocenv.env import *
from utils.logging import *
from utils.svo import svo
import numpy as np


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
        cfg.ra_agent.action_dim = self.env.action_space.n

        try:
            cfg.ra_agent.seq_len = self.env.episode_length
        except:
            pass


        cfg.ma_agent.obs_dim = (1, 256, 256, 3)
        cfg.ma_agent.action_dim = 5

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
                                                batch_size=cfg.ma_agent.batch_size,
                                                num_obs=(256, 256, 3),
                                                num_action=5,
                                                num_rec=128)



        self.writer = None

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)

        self.step = 0

        self.MA_ENGAGED = False

    def evaluate(self):
        average_episode_reward = 0
        average_ma_reward = 0
        average_svo_reward = [0] * 4
        self.video_recorder.init(enabled=True)

        for episode in range(self.cfg.num_eval_episodes):
            obs, _ = self.env.reset()
            episode_step = 0

            done = False
            episode_reward = 0
            epi_ma_reward = 0
            episode_svo_reward = [0] * 4
            while not done:

                if type(self.ra_agent) in [RuleBasedAgent, RuleBasedAgentGroup]:
                    obs = self.env.get_numeric_observation()

                if type(self.ra_agent) is CPCAgentGroup:
                    action, cpc_info = self.ra_agent.act(self.ra_replay_buffer, obs, episode_step, sample=True)
                else:
                    action = self.ra_agent.act(obs, sample=True)

                obs, rewards, dones, env_info = self.env.step(action)

                done = True in dones
                if episode_step == self.env.episode_length:
                    done = True

                ma_obs = self.env.render(coordination=False)
                self.video_recorder.record(self.env)

                ma_obs_in = np.expand_dims(ma_obs, axis=0)

                if type(self.ma_agent) is CPCAgentGroup:
                    ma_action, ma_cpc_info = self.ma_agent.act(self.ma_replay_buffer, ma_obs_in, episode_step, sample=True)
                else:
                    ma_action = self.ma_agent.act(ma_obs_in, sample=True)

                # MA reward shaping

                for i in range(self.num_agent):
                    episode_svo_reward[i] += svo(rewards, i, self.preferences)
                if type(self.ra_agent) in [CPCAgentGroup]:
                    self.ra_replay_buffer.add(obs, action, rewards, dones, cpc_info)

                if episode_step == 0:
                    ma_reward = np.zeros((1, 1))
                else:
                    ma_reward = np.reshape(env_info['step_eaten_apple'], (1, -1))
                epi_ma_reward += ma_reward[0]
                if type(self.ma_agent) in [CPCAgentGroup]:
                    self.ma_replay_buffer.add(ma_obs_in, ma_action[0], ma_reward, dones, ma_cpc_info)

                self.env.punish_agent(ma_action[0])

                episode_reward += sum(rewards)
                episode_step += 1

            average_episode_reward += episode_reward
            average_ma_reward += epi_ma_reward
            average_svo_reward += episode_svo_reward

        self.video_recorder.save(f'{self.step}.mp4')

        if self.cfg.save_model:
            self.ra_agent.save(self.step)

        average_episode_reward /= self.cfg.num_eval_episodes
        average_ma_reward /= self.cfg.num_eval_episodes
        # Clear Buffer
        self.ma_replay_buffer.after_update()
        self.ra_replay_buffer.after_update()

        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.log('eval/ma_reward', average_ma_reward, self.step)
        for i in range(4):
            self.logger.log('eval/agent{}_SVO_reward'.format(i), average_svo_reward[i], self.step)
        self.logger.dump(self.step)

    def run(self):

        episode, episode_step, episode_reward, done = 1, 0, 0, True
        start_time = time.time()
        env_info = None

        while self.step < self.cfg.num_train_steps + 1:
            if done or self.step % self.cfg.eval_frequency == 0:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode - 1, self.step)
                    self.evaluate()
                    start_time = time.time()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                ''' Log Environment Statistics '''

                if env_info:
                    total_apples = 0
                    for item in env_info['agents']:
                        total_apples += item['eaten_apples']
                    log_statistics_to_writer(self.logger, self.step, env_info['statistics'])
                    log_agent_to_writer(self.logger, self.step, env_info['agents'])

                if (hasattr(self, 'ra_replay_buffer')) and (self.step > 0):

                    if (self.ra_replay_buffer.n + 1) % self.ra_agent.batch_size == 0:
                        self.MA_ENGAGED = not self.MA_ENGAGED

                    self.ra_agent.train(self.ra_replay_buffer, self.logger, self.step)

                if (hasattr(self, 'ma_replay_buffer')) and (self.step > 0):
                    if self.MA_ENGAGED:
                        self.ma_agent.train(self.ma_replay_buffer, self.logger, self.step)

                self.logger.log('train/episode', episode, self.step)

                obs, env_info = self.env.reset()

                episode_reward = 0
                episode_step = 0
                episode += 1


            if type(self.ra_agent) in [RuleBasedAgent, RuleBasedAgentGroup]:
                obs = self.env.get_numeric_observation()

            if self.step < self.cfg.num_seed_steps:
                # Define random actions
                if type(self.ra_agent) is CPCAgentGroup:
                    action, cpc_info = self.ra_agent.act(self.ra_replay_buffer, obs, episode_step, sample=True)
                else:
                    action = self.ra_agent.act(obs, sample=True)
            else:
                if type(self.ra_agent) is CPCAgentGroup:
                    action, cpc_info = self.ra_agent.act(self.ra_replay_buffer, obs, episode_step, sample=True)
                else:
                    action = self.ra_agent.act(obs, sample=True)

            next_obs, rewards, dones, env_info = self.env.step(action)
            ma_obs = self.env.render(coordination=False)
            ma_obs_in = np.expand_dims(ma_obs, axis=0)

            '''MA action'''
            if self.MA_ENGAGED:
                if type(self.ma_agent) is CPCAgentGroup:
                    ma_action, ma_cpc_info = self.ma_agent.act(self.ma_replay_buffer, ma_obs_in, episode_step, sample=True)
                else:
                    ma_action = self.ma_agent.act(ma_obs_in, sample=True)

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

            if self.MA_ENGAGED:
                if episode_step == 0:
                    ma_reward = np.zeros((1, 1))
                else:
                    ma_reward = np.reshape(env_info['step_eaten_apple'], (1, -1))

                    if int(ma_action[0]) > 0:
                        ma_reward = ma_reward + np.array([[float(self.cfg.ma_beam_reward)]])

            if type(self.ma_agent) in [CPCAgentGroup]:
                if self.MA_ENGAGED:
                    self.ma_replay_buffer.add(ma_obs_in, ma_action[0], ma_reward, dones, ma_cpc_info)

            if self.MA_ENGAGED:
                self.env.punish_agent(ma_action[0])

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path="config", config_name="train_env")
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
