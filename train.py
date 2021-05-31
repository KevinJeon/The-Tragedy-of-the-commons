import cv2
import hydra
import os
import time
import torch
from omegaconf import DictConfig

from logger import Logger
from models.RuleBasedAgent import *
from models.CPCAgent import *
from models.utils.RolloutStorage import RolloutStorage
from recorder import VideoRecorder
from tocenv.env import *
from utils.logging import *
import numpy as np

def svo(rews, aind):
    exc_rews = (np.sum(rews) - rews[aind]) / (len(rews) - 1)
    my_rew = rews[aind]
    if my_rew == 0:
        rew_angle = np.pi / 2
    else:
        rew_angle = np.arctan(exc_rews / my_rew)

    target_angle = np.pi / 2
    w = 0.2
    U = my_rew - w * abs(target_angle - rew_angle)
    return U

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

        prefer = ['blue'] * cfg.env.blue_agent_count + ['red'] * cfg.env.red_agent_count
        self.env = TOCEnv(agents=prefer,
                          map_size=(cfg.env.width, cfg.env.height),
                          episode_max_length=cfg.env.episode_length,
                          apple_color_ratio=cfg.env.apple_color_ratio,
                          apple_spawn_ratio=cfg.env.apple_spawn_ratio,
                          patch_count=cfg.env.patch_count,
                          patch_distance=cfg.env.patch_distance,
                          reward_same_color=cfg.env.reward_same_color,
                          reward_oppo_color=cfg.env.reward_oppo_color
                          )

        self.device = torch.device(cfg.device)
        self.env.reset()

        cfg.agent.obs_dim = self.env.observation_space.shape
        cfg.agent.action_dim = self.env.action_space.n
        cfg.agent.agent_types = prefer
        self.obs_dim = 15
        try:
            cfg.agent.seq_len = self.env.episode_length
        except:
            pass
        self.num_agent = cfg.env.blue_agent_count + cfg.env.red_agent_count
        self.agent = hydra.utils.instantiate(cfg.agent)

        if type(self.agent) in [CPCAgentGroup]:
            self.replay_buffer = RolloutStorage(agent_type='ac',
                                                num_agent=cfg.env.blue_agent_count + cfg.env.red_agent_count,
                                                num_step=cfg.env.episode_length,
                                                batch_size=cfg.agent.batch_size, num_obs=(8 * self.obs_dim, 8 * self.obs_dim, 3), num_action=8,
                                                num_rec=128)

        # self.writer = SummaryWriter(log_dir="tb")
        self.writer = None

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.video_recorder_blue = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.video_recorder_red = VideoRecorder(self.work_dir if cfg.save_video else None)

        self.step = 0

    def evaluate(self):
        average_episode_reward = 0

        self.video_recorder.init(enabled=True)
        self.video_recorder_blue.init(enabled=True)
        self.video_recorder_red.init(enabled=True)

        for episode in range(self.cfg.num_eval_episodes):
            obs, _ = self.env.reset()
            episode_step = 0

            done = False
            episode_reward = 0
            while not done:

                if type(self.agent) in [RuleBasedAgent, RuleBasedAgentGroup]:
                    obs = self.env.get_numeric_observation()

                if type(self.agent) is CPCAgentGroup:
                    action, cpc_info = self.agent.act(self.replay_buffer, obs, episode_step, sample=True)
                else:
                    action = self.agent.act(obs, sample=True)

                obs, rewards, dones, _ = self.env.step(action)

                done = True in dones
                if episode_step == self.env.episode_length:
                    done = True

                self.video_recorder.record(self.env)

                ''' Render Individual Sight-view '''
                self.video_recorder_blue.record_observation(obs[0])
                self.video_recorder_red.record_observation(obs[4])

                episode_reward += sum(rewards)

                episode_step += 1

            average_episode_reward += episode_reward
        self.video_recorder.save(f'{self.step}.mp4')
        self.video_recorder_blue.save(f'{self.step}_blue.mp4')
        self.video_recorder_red.save(f'{self.step}_red.mp4')

        if self.cfg.save_model:
            self.agent.save(self.step)

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
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
                    log_statistics_to_writer(self.logger, self.step, env_info['statistics'])
                    log_agent_to_writer(self.logger, self.step, env_info['agents'])

                if hasattr(self, 'replay_buffer'):
                    self.agent.train(self.replay_buffer, self.logger, self.step)
                self.logger.log('train/episode', episode, self.step)

                obs, env_info = self.env.reset()

                episode_reward = 0
                episode_step = 0
                episode += 1

            if type(self.agent) in [RuleBasedAgent, RuleBasedAgentGroup]:
                obs = self.env.get_numeric_observation()

            if self.step < self.cfg.num_seed_steps:
                # Define random actions
                if type(self.agent) is CPCAgentGroup:
                    action, cpc_info = self.agent.act(self.replay_buffer, obs, episode_step, sample=True)
                else:
                    action = self.agent.act(obs, sample=True)
            else:
                if type(self.agent) is CPCAgentGroup:
                    action, cpc_info = self.agent.act(self.replay_buffer, obs, episode_step, sample=False)
                else:
                    action = self.agent.act(obs, sample=False)
            next_obs, rewards, dones, env_info = self.env.step(action)

            if self.cfg.render:
                cv2.imshow('TOCEnv', self.env.render())
                cv2.waitKey(1)

            done = True in dones
            if episode_step >= self.env.episode_length:
                print(episode_step, self.env.episode_length, 'oot')
                done = True

            episode_reward += sum(rewards)
            modified_rewards = np.zeros(self.num_agent)
            # Applying prosocial SVO
            for i in range(self.num_agent):
                modified_rewards[i] = svo(rewards, i)
            if type(self.agent) in [CPCAgentGroup]:
                self.replay_buffer.add(obs, action, modified_rewards, dones, cpc_info)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
