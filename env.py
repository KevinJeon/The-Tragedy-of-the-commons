
import numpy as np
import cv2 as cv

from components.block import BlockType


from utils.image import put_rgba_to_image, put_rgb_to_image


class TOCEnv(object):

    def __init__(self,
                 apple_respawn_rate=1,
                 num_agents=4,
                 map_size=(16, 16),
                 episode_max_length=200,
                 ):

        self.num_agents = num_agents
        self.map_size = map_size
        self.episode_max_length = episode_max_length
        self._step_count = 0
        self.apple_count = 0

        self.world = World(num_agents=num_agents, size=map_size)

        self.pixel_per_block = 16
        self._individual_render_pixel = 8

        self.apple_respawn_rate = apple_respawn_rate

        self.redered_layer = None

        self.reset()

    def step(self, actions):
        assert len(actions) is self.world.num_agents

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

        obs = [self._render_individual_view(iter_agent.get_view()) for iter_agent in self.world.agents]
        obs = np.array(obs, dtype=np.float64)

        directions = [iter_agent.direction.value for iter_agent in self.world.agents]

        infos = {
            'agents': {
                'directions': directions,
            },
            'reward': individual_rewards
        }

        self._step_count += 1

        done = False
        if self._step_count >= self.episode_max_length:
            done = True
        elif self._get_apple_count() == 0:
            done = True
        if done:
            self.reset()

        return obs, common_reward, done, infos

    def reset(self):
        del self.world
        self.world = World(num_agents=self.num_agents, size=self.map_size)
        self._step_count = 0

    def _render_layers(self) -> None:
        raise NotImplementedError

    def _render_actor(self) -> np.array:
        raise NotImplementedError

    def _render_item(self) -> np.array:
        raise NotImplementedError

    def _render_view(self) -> np.array:
        raise NotImplementedError

    def render(self) -> np.array:
        image_size = (self.world.height * self.pixel_per_block, self.world.width * self.pixel_per_block, 3)

        layer_field = np.zeros(shape=image_size)

        # Draw respawn fields
        for field in self.world.fruits_fields:
            cv.rectangle(layer_field, pt1=(field.p1 * self.pixel_per_block).to_tuple(reverse=True), \
                         pt2=((field.p2 + Position(1, 1)) * self.pixel_per_block).to_tuple(reverse=True), \
                         color=(50, 50, 50), \
                         thickness=-1 \
                         )
        layer_field = cv.flip(layer_field, 0) # Vertical flip

        layer_actors = np.zeros(shape=image_size)

        # Draw items

        layer_items = np.zeros(shape=image_size)

        resized_apple = cv.resize(Resource.Apple, dsize=(self.pixel_per_block, self.pixel_per_block))
        for y in range(self.world.height):
            for x in range(self.world.width):
                content = self.world.grid[y][x]
                if content is None: continue

                if isinstance(content, items.Apple):
                    pos_y, pos_x = (Position(x=x, y=y) * self.pixel_per_block).to_tuple()
                    put_rgba_to_image(resized_apple, layer_items, pos_x, image_size[0] - pos_y - self.pixel_per_block)


        gray_layer_items = cv.cvtColor(layer_items.astype(np.uint8), cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(gray_layer_items, 1, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)
        masked_layer_field = cv.bitwise_and(layer_field, layer_field, mask=mask_inv)
        masked_layer_items = cv.bitwise_and(layer_items, layer_items, mask=mask)

        layer_field = cv.add(masked_layer_field, masked_layer_items)

        # Draw agents
        resized_agent = cv.resize(Resource.Agent, dsize=(self.pixel_per_block, self.pixel_per_block))

        for iter_agent in self.world.agents:
            pos_y, pos_x = (iter_agent.get_position() * self.pixel_per_block).to_tuple()
            put_rgba_to_image(resized_agent, layer_field, pos_x, image_size[0] - pos_y - self.pixel_per_block)

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
            put_rgba_to_image(np.rot90(heading_tile, k=iter_agent.direction.value), layer_heading, pos_x, image_size[0] - pos_y - self.pixel_per_block)

        gray_layer_heading = cv.cvtColor(layer_heading.astype(np.uint8), cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(gray_layer_heading, 1, 255, cv.THRESH_BINARY)

        masked_dest_layer = cv.bitwise_and(output_layer, output_layer, mask=mask_inv)
        masked_src_layer = cv.bitwise_and(layer_heading, layer_heading, mask=mask)
        output_layer = cv.add(masked_dest_layer, masked_src_layer)

        # Draw Effects
        layer_effects = np.zeros(shape=image_size)

        resized_flame = cv.resize(Resource.Flame, dsize=(self.pixel_per_block, self.pixel_per_block))
        resized_flame[:, :, 3] = resized_flame[:, :, 3] * 0.7

        for y in range(self.world.height):
            for x in range(self.world.width):

                effects = self.world.effects[y][x]

                if np.bitwise_and(int(effects), BlockType.Punish):
                    pos_y, pos_x = (Position(x=x, y=y) * self.pixel_per_block).to_tuple()
                    put_rgba_to_image(resized_flame, output_layer, pos_x, image_size[0] - pos_y - self.pixel_per_block)

        return (output_layer / 255.).astype(np.float32)

    def _render_individual_view(self, view: np.array) -> np.array:
        height, width = view.shape[0], view.shape[1]
        image_size = (height * self._individual_render_pixel, width * self._individual_render_pixel, 3)
        layer_output = np.zeros(shape=image_size)

        resized_agent = cv.resize(Resource.Agent, dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_agent2 = cv.resize(Resource.AgentBlue, dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_apple = cv.resize(Resource.Apple, dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_wall = cv.resize(Resource.Wall, dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_flame = cv.resize(Resource.Flame, dsize=(self._individual_render_pixel, self._individual_render_pixel))
        resized_flame[:, :, 3] = resized_flame[:, :, 3] * 0.7

        # Draw blocks
        for y, row in enumerate(reversed(view)):
            for x, item in enumerate(row):
                if item == BlockType.Empty:
                    continue
                elif  np.bitwise_and(int(item), BlockType.Apple):
                    pos_y, pos_x = (Position(x=x, y=y) * self._individual_render_pixel).to_tuple()
                    put_rgba_to_image(resized_apple, layer_output, pos_x, image_size[0] - pos_y - self._individual_render_pixel)
                elif  np.bitwise_and(int(item), BlockType.OutBound):
                    pos_y, pos_x = (Position(x=x, y=y) * self._individual_render_pixel).to_tuple()
                    put_rgb_to_image(resized_wall, layer_output, pos_x, image_size[0] - pos_y - self._individual_render_pixel)
                elif  np.bitwise_and(int(item), BlockType.Self):
                    pos_y, pos_x = (Position(x=x, y=y) * self._individual_render_pixel).to_tuple()
                    put_rgba_to_image(resized_agent2, layer_output, pos_x, image_size[0] - pos_y - self._individual_render_pixel)
                elif  np.bitwise_and(int(item), BlockType.Others):
                    pos_y, pos_x = (Position(x=x, y=y) * self._individual_render_pixel).to_tuple()
                    put_rgba_to_image(resized_agent, layer_output, pos_x, image_size[0] - pos_y - self._individual_render_pixel)
                elif np.bitwise_and(int(item), BlockType.Punish):
                    pos_y, pos_x = (Position(x=x, y=y) * self._individual_render_pixel).to_tuple()
                    put_rgba_to_image(resized_flame, layer_output, pos_x, image_size[0] - pos_y - self._individual_render_pixel)

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

    def show(self) -> None:
        print('###### World ######')

        self.world.spawn_item()

        for y in range(self.world.height):
            for x in range(self.world.width):
                print('{:^5}'.format(str(self.world.grid[y][x])), end='')
            print()
        print('###################')

        print(self.world.get_agents())

    def respawn_apple(self):
        raise NotImplementedError


from components.world import World, Position
from components.resource import Resource
import components.item as items