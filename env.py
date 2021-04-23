class TOCEnv(object):
    pass

import random
import numpy as np
import cv2 as cv

from components.block import BlockType
from components.position import Position


from utils.image import put_rgba_to_image, put_rgb_to_image

from collections import namedtuple
from components.agent import Action

from components.agent import BlueAgent, RedAgent
from components.world import VariousAppleField
from components.agent import Color

ObservationSpace = namedtuple('ObservationSpace', 'shape')
ActionSpace = namedtuple('ActionSpace', 'shape n')


class TOCEnv(object):

    def __init__(self,
                 agents: list,
                 apple_respawn_rate=1,

                 map_size=(16, 16),
                 episode_max_length=300,
                 obs_type='rgb_array',

                 apple_color_ratio=0.5,
                 apple_spawn_ratio=0.3,
                 
                 patch_count=3,
                 patch_distance=5,
                 ):

        self.agents = agents
        self.num_agents = len(self.agents)
        self._apple_color_ratio = apple_color_ratio
        self._apple_spawn_ratio = apple_spawn_ratio

        self.map_size = map_size
        self.episode_max_length = episode_max_length
        self.obs_type = obs_type
        assert self.obs_type in ['rgb_array', 'numeric']  # Observation type should be in [rgb_array|numeric]

        self._step_count = 0
        self.apple_count = 0

        self.pixel_per_block = 32
        self._individual_render_pixel = 8

        self.apple_respawn_rate = apple_respawn_rate

        ''' Patch settings '''
        self.patch_count = patch_count
        self.patch_distance = patch_distance

        ''' Debug '''
        self._debug_buffer_line = []

        self._create_world()
        self.reset()


    def step(self, actions):
        assert len(actions) is self.num_agents

        # Clear skill (You should bew clear effects before tick
        self.world.clear_effect()

        for agent, action in zip(self.world.agents, actions):
            agent.act(action)

        self.world.tick()

        common_reward = 0
        individual_rewards = []
        for iter_agent in self.world.agents:
            _individual_reward = iter_agent.reset_reward()
            common_reward += _individual_reward
            individual_rewards.append(_individual_reward)

        if self.obs_type == 'rgb_array':
            obs = [self._render_individual_view(iter_agent.get_view()) for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.float32)
        elif self.obs_type == 'numeric':
            obs = [iter_agent.get_view_as_type() for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.uint8)

        directions = [iter_agent.direction.value for iter_agent in self.world.agents]
        color_agents = [agent.color for agent in self.world.agents]

        infos = {
            'agents': {
                'directions': directions,
                'colors': color_agents,
            },
            'reward': individual_rewards
        }

        self._step_count += 1

        done = False
        if self._step_count >= self.episode_max_length:
            done = True
        if done:
            self.reset()

        return obs, common_reward, done, infos

    def reset(self) -> np.array:
        del self.world
        self._create_world()
        self._step_count = 0

        # This is for two-color resource allocation experiemnts
        # self.world.add_fruits_field(VariousAppleField(
        #     world=self.world,
        #     p1=Position(1, 1),
        #     p2=Position(self.world.width - 2, self.world.height - 2),
        #     prob=self._apple_spawn_ratio,
        #     ratio=self._apple_color_ratio
        # ))

        for color in self.agents:
            pos = Position(x=random.randint(0, self.world.width - 1), y=random.randint(0, self.world.height - 1))

            if color == 'red':
                self.world.spawn_agent(pos=pos, color=Color.Red)
            elif color == 'blue':
                self.world.spawn_agent(pos=pos, color=Color.Blue)
            else:
                raise Exception('Unknown color type')

        if self.obs_type == 'rgb_array':
            obs = [self._render_individual_view(iter_agent.get_view()) for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.float32)
        elif self.obs_type == 'numeric':
            obs = [iter_agent.get_view_as_type() for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.uint8)

        color_agents = [agent.color for agent in self.world.agents]

        common_reward = 0
        individual_rewards = []
        for iter_agent in self.world.agents:
            _individual_reward = iter_agent.reset_reward()
            common_reward += _individual_reward
            individual_rewards.append(_individual_reward)

        if self.obs_type == 'rgb_array':
            obs = [self._render_individual_view(iter_agent.get_view()) for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.float32)
        elif self.obs_type == 'numeric':
            obs = [iter_agent.get_view_as_type() for iter_agent in self.world.agents]
            obs = np.array(obs, dtype=np.uint8)

        directions = [iter_agent.direction.value for iter_agent in self.world.agents]

        infos = {
            'agents': {
                'directions': directions,
                'colors': color_agents,
            },
            'reward': individual_rewards
        }

        return obs, infos

    def _create_world(self):
        print(self.patch_distance, self.patch_count)
        self.world = World(env=self, size=self.map_size, \
                           patch_distance=self.patch_distance, patch_count=self.patch_count)

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
        resized_agent_red = cv.resize(Resource.AgentRed, dsize=(self.pixel_per_block, self.pixel_per_block))
        resized_agent_blue = cv.resize(Resource.AgentBlue, dsize=(self.pixel_per_block, self.pixel_per_block))

        for iter_agent in self.world.agents:
            pos_y, pos_x = (iter_agent.get_position() * self.pixel_per_block).to_tuple()
            if isinstance(iter_agent, BlueAgent):
                put_rgba_to_image(resized_agent_blue, layer_field, pos_x, image_size[0] - pos_y - self.pixel_per_block)
            elif isinstance(iter_agent, RedAgent):
                put_rgba_to_image(resized_agent_red, layer_field, pos_x, image_size[0] - pos_y - self.pixel_per_block)

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
                    # cv.putText(output_layer, '{0},{1}'.format(y, x),
                    #            (y * self.pixel_per_block, image_size[1] - x * self.pixel_per_block - 10),
                    #            cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (255, 255, 255), 1, cv.LINE_AA)

                    print(self.world.width, x)
                    cv.putText(output_layer, '{0:2},{1:2}'.format(self.world.width - x, self.world.height - y),
                               (image_size[1] - x * self.pixel_per_block, y * self.pixel_per_block - 10),
                               cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (255, 255, 255), 1, cv.LINE_AA)



        ''' Debug '''
        [print(agent.position) for agent in self.world.agents]
        for pos1, pos2 in self._debug_buffer_line:
            pass
            # print(pos1, pos2)

        return (output_layer / 255.).astype(np.float32)

    def _render_individual_view(self, view: np.array) -> np.array:
        height, width = view.shape[0], view.shape[1]
        image_size = (height * self._individual_render_pixel, width * self._individual_render_pixel, 3)
        layer_output = np.zeros(shape=image_size)

        resized_agent = cv.resize(Resource.Agent, dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_agent_red = cv.resize(Resource.AgentRed,
                                      dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_agent_blue = cv.resize(Resource.AgentBlue,
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
                if np.bitwise_and(int(item), BlockType.RedAgent):
                    put_rgba_to_image(resized_agent_red, layer_output, pos_x,
                                      image_size[0] - pos_y - self._individual_render_pixel)
                if np.bitwise_and(int(item), BlockType.BlueAgent):
                    put_rgba_to_image(resized_agent_blue, layer_output, pos_x,
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

    def set_patch_count(self, count: int) -> None:
        self.patch_count = count

    def set_patch_distance(self, distance: int) -> None:
        self.patch_distance = distance

    def draw_line(self, pos1: Position, pos2: Position):
        self._debug_buffer_line.append((pos1, pos2))

    @property
    def observation_space(self):
        if self.obs_type == 'rgb_array':
            return np.zeros(
                shape=(self.num_agents, self._individual_render_pixel * 11, self._individual_render_pixel * 11),
                dtype=np.float32)

        elif self.obs_type == 'numeric':
            return np.zeros(
                shape=(self.num_agents, 11, 11),
                dtype=np.float32)

    @property
    def action_space(self):
        action_space = ActionSpace(shape=self.num_agents, n=Action().count)
        return action_space



from components.world import World, Position
from components.resource import Resource
import components.item as items
