import cv2 as cv
import numpy as np

import hydra, time, os
from omegaconf import DictConfig

from tocenv.env import TOCEnv

import torch
from torch.utils.tensorboard import SummaryWriter

from models.utils.RolloutStorage import RolloutStorage
from logger import Logger
from recorder import VideoRecorder
from utils.logging import log_statistics_to_writer

from models.RuleBasedAgent import RuleBasedAgent


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'Workspace: {self.work_dir}')

        self.cfg = cfg

        # set_seed_everywhere(cfg.seed)

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        prefer = ['blue'] * cfg.blue_agent_count + ['red'] * cfg.red_agent_count
        self.env = TOCEnv(agents=prefer,
                          episode_max_length=cfg.episode_length,
                          apple_color_ratio=0.5,
                          apple_spawn_ratio=0.1)

        self.device = torch.device(cfg.device)
        self.save_replay_buffer = cfg.save_replay_buffer
        self.env.reset()

        cfg.agent.obs_dim = self.env.observation_space.shape
        cfg.agent.action_dim = self.env.action_space.shape

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = RolloutStorage(agent_type='ac', num_agent=cfg.blue_agent_count + cfg.red_agent_count,
                                            num_step=cfg.env.episode_length,
                                            batch_size=cfg.agent.batch_size, num_obs=(88, 88, 3), num_action=8,
                                            num_rec=128)

        # self.writer = SummaryWriter(log_dir="tb")
        self.writer = None

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0

        self.video_recorder.init(enabled=True)
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            episode_step = 0

            done = False
            episode_reward = 0
            while not done:

                if type(self.agent) == RuleBasedAgent:
                    obs = self.env.get_numeric_observation()

                action = self.agent.act(obs, sample=False)
                obs, rewards, dones, _ = self.env.step(action)

                done = True in dones
                if episode_step == self.env.episode_length:
                    done = True

                self.video_recorder.record(self.env)
                episode_reward += sum(rewards)

                episode_step += 1

            average_episode_reward += episode_reward
        self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes

    def run(self):

        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        env_info = None

        while self.step < self.cfg.num_train_steps + 1:
            if done or self.step % self.cfg.eval_frequency == 0:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    start_time = time.time()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                ''' Log Environment Statistics '''

                self.logger.log('train/episode', episode, self.step)
                if env_info:
                    log_statistics_to_writer(self.logger, self.step, env_info['statistics'])

                obs, env_info = self.env.reset()

                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            if type(self.agent) == RuleBasedAgent:
                obs = self.env.get_numeric_observation()

            if self.step < self.cfg.num_seed_steps:
                # Define random actions
                action = self.agent.act(obs, sample=True)
            else:
                action = self.agent.act(obs, sample=True)

            next_obs, rewards, dones, env_info = self.env.step(action)

            done = True in dones
            if episode_step + 1 == self.env.episode_length:
                done = True

            episode_reward += sum(rewards)

            # self.replay_buffer.add(obs, action, rewards, next_obs, dones)

            if self.step >= self.cfg.num_seed_steps and self.step >= self.agent.batch_size:
                self.agent.train(self.replay_buffer, self.logger, self.step)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
