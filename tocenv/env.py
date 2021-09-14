import copy


class TOCEnv(object):
    pass


import random
import numpy as np
import cv2 as cv
import json
import os

import tocenv.components.skill as skills

from tocenv.components.block import BlockType
from tocenv.components.position import Position

from tocenv.utils.image import put_rgba_to_image, put_rgb_to_image

from collections import namedtuple
from tocenv.components.agent import Action

from tocenv.components.agent import Agent
from tocenv.components.agent import Color

ObservationSpace = namedtuple('ObservationSpace', 'shape')
ActionSpace = namedtuple('ActionSpace', 'shape n')


class TOCEnv(object):

    def __init__(self,
                 agents: list,

                 map_size=(16, 16),
                 episode_max_length=300,
                 obs_type='rgb_array',

                 apple_color_ratio=0.5,
                 apple_spawn_ratio=0.3,

                 patch_count=3,
                 patch_distance=5,

                 reward_same_color=3,
                 reward_oppo_color=1
                 ):

        self.agents = agents
        self.num_agents = len(self.agents)
        self.obs_dim = 11
        self.map_size = map_size
        self.episode_max_length = episode_max_length
        self.obs_type = obs_type
        assert self.obs_type in ['rgb_array', 'numeric']  # Observation type should be in [rgb_array|numeric]

        self._step_count = 0
        self.apple_count = 0

        self.pixel_per_block = 16
        self._individual_render_pixel = 8

        ''' Info variables '''
        self._total_red_eaten_count = 0
        self._total_blue_eaten_count = 0

        self._red_team_red_apple_count = 0
        self._red_team_blue_apple_count = 0

        self._blue_team_red_apple_count = 0
        self._blue_team_blue_apple_count = 0

        self._punishing_count = 0
        self._punished_count = 0

        self._ma_punishing_cnt = 0

        self._movement_count = 0
        self._rotate_count = 0

        ''' Patch settings '''
        self.patch_count = patch_count
        self.patch_distance = patch_distance

        ''' Apple spawning settings '''
        self.apple_color_ratio = apple_color_ratio
        self.apple_spawn_ratio = apple_spawn_ratio

        ''' Reward settings '''
        self.reward_same_color = float(reward_same_color)
        self.reward_oppo_color = float(reward_oppo_color)

        ''' Debug '''
        self._debug_buffer_line = []

        self._create_world()
        self.reset()

        self._debug_buffer_line.clear()

    def step(self, actions):
        assert len(actions) is self.num_agents

        # Clear skill (You should bew clear effects before tick
        self.world.clear_effect()

        _durating_effects = list()

        for pos, effect in self.world.durating_effects:
            _effect = copy.deepcopy(effect)

            self.world.effects[pos.y][pos.x] = np.bitwise_or(int(self.world.effects[pos.y][pos.x]), BlockType.Punish)

            _effect.effect_duration -= 1
            if _effect.effect_duration > 1:
                _durating_effects.append((pos, _effect))

        self.world.durating_effects = _durating_effects

        for agent, action in zip(self.world.agents, actions):
            agent.act(action)

        self.world.tick()

        common_reward = 0
        individual_rewards = []
        for iter_agent in self.world.agents:
            _individual_reward = iter_agent.get_reward()
            common_reward += _individual_reward
            individual_rewards.append(_individual_reward)

        if self.obs_type == 'rgb_array':
            obs = [self._render_individual_view(iter_agent.get_view()) for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.float32)
        elif self.obs_type == 'numeric':
            obs = [iter_agent.get_view_as_type() for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.uint8)

        self._step_count += 1

        done = False
        if self._step_count >= self.episode_max_length:
            done = True
        if done:
            pass
            # self.reset()

        done = [done for _ in self.agents]

        info = self._gather_info()
        [iter_agent.tick() for iter_agent in self.world.agents]

        return obs, np.array(individual_rewards), np.array(done), info

    def reset(self) -> np.array:
        del self.world
        self._create_world()
        self._step_count = 0

        for color in self.agents:
            pos = Position(x=random.randint(0, self.world.width - 1), y=random.randint(0, self.world.height - 1))

            if color == 'green':
                self.world.spawn_agent(pos=pos, color=Color.Green)
            elif color == 'purple':
                self.world.spawn_agent(pos=pos, color=Color.Purple)
            elif color == 'blue':
                self.world.spawn_agent(pos=pos, color=Color.Blue)
            elif color == 'orange':
                self.world.spawn_agent(pos=pos, color=Color.Orange)
            else:
                raise Exception('Unknown color type')

        if self.obs_type == 'rgb_array':
            obs = [self._render_individual_view(iter_agent.get_view()) for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.float32)
        elif self.obs_type == 'numeric':
            obs = [iter_agent.get_view_as_type() for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.uint8)

        common_reward = 0
        individual_rewards = []
        for iter_agent in self.world.agents:
            _individual_reward = iter_agent.get_reward()
            common_reward += _individual_reward
            individual_rewards.append(_individual_reward)

        if self.obs_type == 'rgb_array':
            obs = [self._render_individual_view(iter_agent.get_view()) for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.float32)
        elif self.obs_type == 'numeric':
            obs = [iter_agent.get_view_as_type() for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.uint8)

        info = self._gather_info()
        [iter_agent.tick() for iter_agent in self.world.agents]

        self._reset_statistics()
        [iter_agent.reset_statistics() for iter_agent in self.world.agents]

        return obs, info

    def get_numeric_observation(self) -> np.array:
        obs = [iter_agent.get_view_as_type() for iter_agent in self.world.agents]
        obs = np.array(obs, dtype=np.uint8)
        return obs

    def _reset_statistics(self) -> None:
        self._total_red_eaten_count = 0
        self._total_blue_eaten_count = 0

        self._red_team_red_apple_count = 0
        self._red_team_blue_apple_count = 0

        self._blue_team_red_apple_count = 0
        self._blue_team_blue_apple_count = 0

        self._punishing_count = 0
        self._punished_count = 0

        self._movement_count = 0
        self._rotate_count = 0

        self._ma_punishing_cnt = 0

        [agent.reset_accumulated_reward() for agent in self.world.agents]

    def _create_world(self):
        from tocenv.components.world import World
        self.world = World(env=self, size=self.map_size, \
                           patch_distance=self.patch_distance, patch_count=self.patch_count,
                           apple_color_ratio=self.apple_color_ratio, apple_spawn_ratio=self.apple_spawn_ratio
                           )

    def _render_layers(self) -> None:
        raise NotImplementedError

    def _render_actor(self) -> np.array:
        raise NotImplementedError

    def _render_item(self) -> np.array:
        raise NotImplementedError

    def _render_view(self) -> np.array:
        raise NotImplementedError

    def render(self, coordination=False) -> np.array:
        image_size = (self.world.height * self.pixel_per_block, self.world.width * self.pixel_per_block, 3)

        layer_field = np.zeros(shape=image_size)

        # Draw respawn fields
        for field in self.world.fruits_fields:
            cv.rectangle(layer_field, pt1=(field.p1 * self.pixel_per_block).to_tuple(reverse=True), \
                         pt2=((field.p2 + Position(1, 1)) * self.pixel_per_block).to_tuple(reverse=True), \
                         color=(50, 50, 50), \
                         thickness=-1 \
                         )
        layer_field = cv.flip(layer_field, 0)  # Vertical flip

        layer_actors = np.zeros(shape=image_size)

        # Draw items
        layer_items = np.zeros(shape=image_size)

        resized_apple = cv.resize(Resource.Apple, dsize=(self.pixel_per_block, self.pixel_per_block))
        resized_apple_red = cv.resize(Resource.AppleRed, dsize=(self.pixel_per_block, self.pixel_per_block))
        resized_apple_blue = cv.resize(Resource.AppleBlue, dsize=(self.pixel_per_block, self.pixel_per_block))

        for y in range(self.world.height):
            for x in range(self.world.width):
                content = self.world.grid[y][x]
                if content is None: continue

                pos_y, pos_x = (Position(x=x, y=y) * self.pixel_per_block).to_tuple()

                if isinstance(content, items.BlueApple):
                    put_rgba_to_image(resized_apple_blue, layer_items, pos_x,
                                      image_size[0] - pos_y - self.pixel_per_block)
                elif isinstance(content, items.RedApple):
                    put_rgba_to_image(resized_apple_red, layer_items, pos_x,
                                      image_size[0] - pos_y - self.pixel_per_block)
                elif isinstance(content, items.Apple):
                    put_rgba_to_image(resized_apple, layer_items, pos_x, image_size[0] - pos_y - self.pixel_per_block)

        gray_layer_items = cv.cvtColor(layer_items.astype(np.uint8), cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(gray_layer_items, 1, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)
        masked_layer_field = cv.bitwise_and(layer_field, layer_field, mask=mask_inv)
        masked_layer_items = cv.bitwise_and(layer_items, layer_items, mask=mask)

        layer_field = cv.add(masked_layer_field, masked_layer_items)

        # Draw agents
        resized_agent_blue = cv.resize(Resource.AgentBlue, dsize=(self.pixel_per_block, self.pixel_per_block))
        resized_agent_green = cv.resize(Resource.AgentGreen, dsize=(self.pixel_per_block, self.pixel_per_block))
        resized_agent_orange = cv.resize(Resource.AgentOrange, dsize=(self.pixel_per_block, self.pixel_per_block))
        resized_agent_purple = cv.resize(Resource.AgentPurple, dsize=(self.pixel_per_block, self.pixel_per_block))

        for iter_agent in self.world.agents:
            pos_y, pos_x = (iter_agent.get_position() * self.pixel_per_block).to_tuple()

            from tocenv.components.agent import BlueAgent, GreenAgent, OrangeAgent, PurpleAgent

            if type(iter_agent) == BlueAgent:
                put_rgba_to_image(resized_agent_blue, layer_field, pos_x, image_size[0] - pos_y - self.pixel_per_block)
            elif type(iter_agent) == GreenAgent:
                put_rgba_to_image(resized_agent_green, layer_field, pos_x, image_size[0] - pos_y - self.pixel_per_block)
            elif type(iter_agent) == OrangeAgent:
                put_rgba_to_image(resized_agent_orange, layer_field, pos_x, image_size[0] - pos_y - self.pixel_per_block)
            elif type(iter_agent) == PurpleAgent:
                put_rgba_to_image(resized_agent_purple, layer_field, pos_x, image_size[0] - pos_y - self.pixel_per_block)

        gray_layer_actors = cv.cvtColor(layer_actors.astype(np.uint8), cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(gray_layer_actors, 1, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)
        masked_layer_field = cv.bitwise_and(layer_field, layer_field, mask=mask_inv)
        masked_layer_actors = cv.bitwise_and(layer_actors, layer_actors, mask=mask)

        output_layer = cv.add(masked_layer_field, masked_layer_actors)

        layer_heading = np.zeros(shape=image_size)
        heading_tile = np.zeros((self.pixel_per_block, self.pixel_per_block, 4))
        heading_tile[:, :, 2] = np.ones((self.pixel_per_block, self.pixel_per_block)) * 255
        heading_tile[:, :, 2] = np.ones((self.pixel_per_block, self.pixel_per_block)) * 255
        tile_height = int(self.pixel_per_block * 0.3)
        heading_tile[tile_height:, :, :] = 0

        heading_tile[:, :, 3] = np.ones((self.pixel_per_block, self.pixel_per_block)) * 255 * 0.7

        for iter_agent in self.world.agents:
            pos_y, pos_x = (iter_agent.get_position() * self.pixel_per_block).to_tuple()
            put_rgba_to_image(np.rot90(heading_tile, k=iter_agent.direction.value), layer_heading, pos_x,
                              image_size[0] - pos_y - self.pixel_per_block)

        gray_layer_heading = cv.cvtColor(layer_heading.astype(np.uint8), cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(gray_layer_heading, 1, 255, cv.THRESH_BINARY)

        masked_dest_layer = cv.bitwise_and(output_layer, output_layer, mask=mask_inv)
        masked_src_layer = cv.bitwise_and(layer_heading, layer_heading, mask=mask)
        output_layer = cv.add(masked_dest_layer, masked_src_layer)

        # Draw Effects
        resized_flame = cv.resize(Resource.Flame, dsize=(self.pixel_per_block, self.pixel_per_block))
        resized_flame[:, :, 3] = resized_flame[:, :, 3] * 0.7

        for y in range(self.world.height):
            for x in range(self.world.width):

                effects = self.world.effects[y][x]
                if np.bitwise_and(int(effects), BlockType.Punish):
                    pos_y, pos_x = (Position(x=x, y=y) * self.pixel_per_block).to_tuple()
                    put_rgba_to_image(resized_flame, output_layer, pos_x, image_size[0] - pos_y - self.pixel_per_block)

        if coordination:
            for y in range(self.world.height + 1):
                for x in range(self.world.width + 1):
                    cv.putText(output_layer, '{0:2},{1:2}'.format(self.world.width - x, self.world.height - y),
                               (image_size[1] - x * self.pixel_per_block, y * self.pixel_per_block - 10),
                               cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (255, 255, 255), 1, cv.LINE_AA)

            for pos1, pos2, color in self._debug_buffer_line:
                coord1 = ((pos1.x) * self.pixel_per_block + (self.pixel_per_block // 2),
                          image_size[0] - pos1.y * self.pixel_per_block - (self.pixel_per_block // 2))
                coord2 = ((pos2.x) * self.pixel_per_block + (self.pixel_per_block // 2),
                          image_size[0] - pos2.y * self.pixel_per_block - (self.pixel_per_block // 2))

                output_layer = cv.line(output_layer, coord1, coord2, color)

            self._debug_buffer_line.clear()

        return (output_layer / 255.).astype(np.float32)

    def _render_individual_view(self, view: np.array) -> np.array:
        height, width = view.shape[0], view.shape[1]
        image_size = (height * self._individual_render_pixel, width * self._individual_render_pixel, 3)
        layer_output = np.zeros(shape=image_size)

        resized_agent = cv.resize(Resource.Agent, dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_agent_blue = cv.resize(Resource.AgentBlue,
                                       dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_agent_green = cv.resize(Resource.AgentGreen,
                                       dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_agent_orange = cv.resize(Resource.AgentOrange,
                                       dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_agent_purple = cv.resize(Resource.AgentPurple,
                                       dsize=(self._individual_render_pixel, self._individual_render_pixel))

        resized_apple_red = cv.resize(Resource.AppleRed,
                                      dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_apple_blue = cv.resize(Resource.AppleBlue,
                                       dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_wall = cv.resize(Resource.Wall, dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_flame = cv.resize(Resource.Flame, dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_flame[:, :, 3] = resized_flame[:, :, 3] * 0.7

        # Draw blocks
        for y, row in enumerate(reversed(view)):
            for x, item in enumerate(row):
                if item == BlockType.Empty:
                    continue

                pos_y, pos_x = (Position(x=x, y=y) * self._individual_render_pixel).to_tuple()

                if np.bitwise_and(int(item), BlockType.OutBound):
                    put_rgb_to_image(resized_wall, layer_output, pos_x,
                                     image_size[0] - pos_y - self._individual_render_pixel)
                if np.bitwise_and(int(item), BlockType.Self):
                    put_rgba_to_image(resized_agent, layer_output, pos_x,
                                      image_size[0] - pos_y - self._individual_render_pixel)

                if np.bitwise_and(int(item), BlockType.BlueAgent):
                    put_rgba_to_image(resized_agent_blue, layer_output, pos_x,
                                      image_size[0] - pos_y - self._individual_render_pixel)
                if np.bitwise_and(int(item), BlockType.GreenAgent):
                    put_rgba_to_image(resized_agent_green, layer_output, pos_x,
                                      image_size[0] - pos_y - self._individual_render_pixel)
                if np.bitwise_and(int(item), BlockType.OrangeAgent):
                    put_rgba_to_image(resized_agent_orange, layer_output, pos_x,
                                      image_size[0] - pos_y - self._individual_render_pixel)
                if np.bitwise_and(int(item), BlockType.PurpleAgent):
                    put_rgba_to_image(resized_agent_purple, layer_output, pos_x,
                                      image_size[0] - pos_y - self._individual_render_pixel)

                if np.bitwise_and(int(item), BlockType.BlueApple):
                    pos_y, pos_x = (Position(x=x, y=y) * self._individual_render_pixel).to_tuple()
                    put_rgba_to_image(resized_apple_blue, layer_output, pos_x,
                                      image_size[0] - pos_y - self._individual_render_pixel)
                if np.bitwise_and(int(item), BlockType.RedApple):
                    pos_y, pos_x = (Position(x=x, y=y) * self._individual_render_pixel).to_tuple()
                    put_rgba_to_image(resized_apple_red, layer_output, pos_x,
                                      image_size[0] - pos_y - self._individual_render_pixel)
                if np.bitwise_and(int(item), BlockType.Punish):
                    pos_y, pos_x = (Position(x=x, y=y) * self._individual_render_pixel).to_tuple()
                    put_rgba_to_image(resized_flame, layer_output, pos_x,
                                      image_size[0] - pos_y - self._individual_render_pixel)

        return layer_output / 255.

    def get_full_state(self):
        return self.world.grid

    def _get_apple_count(self):
        positions = []
        for field in self.world.fruits_fields:
            positions.extend(field.positions)

        count = 0
        for position in positions:
            item = self.world.get_item(pos=position)
            if isinstance(item, items.Apple):
                count += 1
        return count

    def respawn_apple(self):
        raise NotImplementedError

    def _gather_info(self) -> dict:
        info = dict()
        # Individuals infos
        info['agents'] = [agent.gather_info() for agent in self.world.agents]

        # Environment infos
        #   Eaten Apples
        eaten_apples = dict()
        total_eaten_apples = dict()
        total_eaten_apples['red'] = self._total_red_eaten_count
        total_eaten_apples['blue'] = self._total_blue_eaten_count
        eaten_apples['total'] = total_eaten_apples

        cnt_eaten_apple = 0
        for agent in info['agents']:
            if agent['eaten'] == 'apple':
                cnt_eaten_apple += 1
        info['step_eaten_apple'] = cnt_eaten_apple

        team_eaten_apples = dict()
        team_eaten_apples['red'] = {'red': self._red_team_red_apple_count, 'blue': self._red_team_blue_apple_count}
        team_eaten_apples['blue'] = {'red': self._blue_team_red_apple_count, 'blue': self._blue_team_blue_apple_count}
        eaten_apples['team'] = team_eaten_apples

        punishment = dict()
        punishment['punishing'] = self._punishing_count
        punishment['punished'] = self._punished_count
        punishment['valid_rate'] = self._punished_count / self._punishing_count if self._punishing_count else 0.

        movement = dict()
        movement['move'] = self._movement_count
        movement['rotate'] = self._rotate_count

        info['statistics'] = dict()
        info['statistics']['eaten_apples'] = eaten_apples
        info['statistics']['punishment'] = punishment
        info['statistics']['movement'] = movement
        info['timestamp'] = self._step_count

        map_info = dict()
        size = {'width': self.world.width, 'height': self.world.height}
        map_info['size'] = size

        info['map'] = map_info

        info['statistics']['ma_agent_punishing'] = self._ma_punishing_cnt
        info['statistics']['alive_patches'] = len(self.world.get_alive_patches())

        return info

    def increase_movement_count(self) -> int:
        self._movement_count += 1
        return self._movement_count

    def increase_rotate_count(self) -> int:
        self._rotate_count += 1
        return self._rotate_count

    def increase_punishing_count(self) -> int:
        self._punishing_count += 1
        return self._punishing_count

    def increase_punished_count(self) -> int:
        self._punished_count += 1
        return self._punished_count

    def increase_red_apple_count(self, eaten_by: Agent) -> int:
        self._total_red_eaten_count += 1

        from tocenv.components.agent import RedAgent, BlueAgent

        if type(eaten_by) == RedAgent:
            self._red_team_red_apple_count += 1
        elif type(eaten_by) == BlueAgent:
            self._blue_team_red_apple_count += 1

        return self._total_red_eaten_count

    def increase_blue_apple_count(self, eaten_by: Agent) -> int:
        self._total_blue_eaten_count += 1

        from tocenv.components.agent import RedAgent, BlueAgent

        if type(eaten_by) is RedAgent:
            self._red_team_blue_apple_count += 1
        elif type(eaten_by) is BlueAgent:
            self._blue_team_blue_apple_count += 1

        return self._total_blue_eaten_count

    ''' Environment Variables Settings '''

    def set_patch_count(self, count: int) -> None:
        self.patch_count = count

    def set_patch_distance(self, distance: int) -> None:
        self.patch_distance = distance

    def set_apple_color_ratio(self, ratio: float) -> None:
        self.apple_color_ratio = ratio

    def apple_spawn_ratio(self, ratio: float) -> None:
        self.apple_spawn_ratio = ratio

    def punish_agent(self, ma_action: np.array) -> None:
        '''
        :param ma_action: shape(5, )
         [NO_OP, AGENT_1, AGENT_2, AGENT_3, AGENT_4]
        :return: None
        '''

        ma_action = ma_action[0]

        if ma_action % 2 == 1:
            pass  # No-op
        else:
            punish = skills.Punish()
            punish.effect_duration = 2
            self.world.apply_effect(self.world.agents[ma_action // 2].position, punish)
            self._ma_punishing_cnt += 1

    ''' Debug settings '''

    def draw_line(self, pos1: Position, pos2: Position, color: Color):
        self._debug_buffer_line.append((pos1, pos2, color))

    @property
    def observation_space(self):
        if self.obs_type == 'rgb_array':
            return np.zeros(
                shape=(self.num_agents, self._individual_render_pixel * self.obs_dim, self._individual_render_pixel * self.obs_dim, 3),
                dtype=np.float32)

        elif self.obs_type == 'numeric':
            return np.zeros(
                shape=(self.num_agents, self.obs_dim, self.obs_dim),
                dtype=np.float32)

    @property
    def action_space(self):
        action_space = ActionSpace(shape=(self.num_agents, Action().action_count), n=Action().action_count)
        return action_space

    def get_observation_space(self):
        if self.obs_type == 'rgb_array':
            return np.zeros(
                shape=(self.num_agents, self._individual_render_pixel * self.obs_dim, self._individual_render_pixel * self.obs_dim, 3),
                dtype=np.float32)

        elif self.obs_type == 'numeric':
            return np.zeros(
                shape=(self.num_agents, 11, 11),
                dtype=np.float32)

    def get_action_space(self):
        action_space = ActionSpace(shape=(self.num_agents, Action().action_count), n=Action().action_count)
        return action_space

    @property
    def episode_length(self) -> int:
        return self.episode_max_length


from tocenv.components.resource import Resource
import tocenv.components.item as items
from tocenv.components.agent import RedAgent, BlueAgent
