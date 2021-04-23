import random


class BFS(object):
    pass


from components.position import Position
from components.world import World, Field


class BFS(object):

    def __init__(self, world: World):
        self.world = world

        self.current_pos = None
        self.candidate_fields = []

    def _recursive_search(self):
        pass

    def get_candidates(self, pos: Position, radius: int, distance: int) -> [Position]:
        candidates = []

        selections = []

        gap = radius + distance

        for x in range(pos.x - gap, pos.x + gap + 1, distance + radius * 2):
            for y in range(pos.y - gap, pos.y + gap + 1, distance + radius * 2):
                if not self.world.map_contains(pos): continue

                pos = Position(x=x, y=y)
                selections.append(pos)

        random.shuffle(selections)

        for pos in selections:

            new_field = Field.create_from_parameter(world=self.world, pos=pos, radius=radius)

            flag_collapse = False

            for field in self.world.fruits_fields:
                if field.is_overlap(field=new_field):
                    flag_collapse = True
                    break

            for field in self.candidate_fields:
                if field.is_overlap(field=new_field):
                    flag_collapse = True
                    break

            if flag_collapse:
                continue

            candidates.append(pos)
            self.candidate_fields.append(new_field)

        return candidates

    def search(self, pos: Position, radius: int, distance: int, k: int) -> [Position]:
        searched_positions = []
        self.candidate_fields.clear()

        self.current_pos = pos

        while len(searched_positions) < k:
            candidates = self.get_candidates(pos=pos, radius=radius, distance=distance)

            for candidate in candidates:
                if len(searched_positions) < k:
                    searched_positions.append(candidate)
                    self.world.env.draw_line(self.current_pos, candidate)
        return searched_positions

