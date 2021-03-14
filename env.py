from components.agent import Agent
# from components.block import Block
from components.world import World
from components.resource import Resource

import numpy as np
import cv2 as cv

from utils.image import put_rgba_to_image

class TOCEnv(object):

    def __init__(self,
                 render=False,
                 apple_respawn_rate=1
                 ):
        self.world = World()

        self.pixel_per_block = 32

        self.apple_respawn_rate = apple_respawn_rate

    def step(self, actions: np.array):
        assert actions.shape is (self.world.num_agents, 1)

        for agent, action in zip(self.world.agents, actions):
            agent.act(action)

    def reset(self):
        del self.world
        self.world = World()
        self.render()

    def render(self) -> np.array:
        image_size = (self.world.height * self.pixel_per_block, self.world.width * self.pixel_per_block, 3)

        layer_ground = np.zeros(shape=image_size)

        layer_field = np.zeros(shape=image_size)

        # Draw respawn fields

        for field in self.world.fruits_fields:
            cv.rectangle(layer_field, pt1=(field.p1 * self.pixel_per_block).to_tuple(), \
                         pt2=(field.p2 * self.pixel_per_block).to_tuple(), \
                         color=(200, 200, 200), \
                         thickness=-1 \
                         )

        layer_actors = np.zeros(shape=image_size)

        # Draw agents
        for iter_agent in self.world.agents:
            # resized_agent = cv.resize(Resource.Agent, dsize=(self.pixel_per_block, self.pixel_per_block))
            resized_agent = cv.resize(Resource.Agent, dsize=(self.pixel_per_block, self.pixel_per_block))

            pos_x, pos_y = (iter_agent.position * self.pixel_per_block).to_tuple()
            put_rgba_to_image(resized_agent, layer_actors, pos_x, pos_y)

        gray_layer_actors = cv.cvtColor(layer_actors.astype(np.uint8), cv.COLOR_BGR2GRAY)

        ret, mask = cv.threshold(gray_layer_actors, 1, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)

        masked_layer_field = cv.bitwise_and(layer_field, layer_field, mask=mask_inv)
        masked_layer_actors = cv.bitwise_and(layer_actors, layer_actors, mask=mask)

        output_layer = cv.add(masked_layer_field, masked_layer_actors)

        cv.imshow('window', output_layer / 255.)
        cv.waitKey(0)

        output = layer_field
        return output

    # TODO Move this method to util file

    def get_full_state(self):
        return self.world.grid

    def show(self) -> None:
        print('###### World ######')
        for y in range(self.world.height):
            for x in range(self.world.width):
                print('{:^5}'.format(str(self.world.grid[y][x])), end='')
            print()
        print('###################')

        print(self.world.get_agents())

    def respawn_apple(self):
        raise NotImplementedError
