import random

from components.agent import Color


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

        print('Candidate Start', pos)
        for x in range(pos.x - gap, pos.x + gap + 1, 1):
            for y in range(pos.y - gap, pos.y + gap + 1, 1):

                candidate = Position(x=x, y=y)
                dist = abs(pos.x - x) + abs(pos.y - y)

                if not self.world.map_contains(candidate): continue
                if not dist == gap: continue

                self.world.env.draw_line(self.current_pos, candidate, Color.Orange)
                selections.append(candidate)

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
        cur_idx = 0

        self.candidate_fields.clear()

        self.current_pos = pos


        while len(searched_positions) < k:

            candidates = self.get_candidates(pos=self.current_pos, radius=radius, distance=distance)

            for candidate in candidates:
                if len(searched_positions) < k:

                    # Check map contains this field
                    field = Field.create_from_parameter(self.world, candidate, radius)
                    if not self.world.contains_field(field): continue
                    # Check is this field collapse with others
                    if self.world.collapsed_field_exist(field): continue

                    flag_overlap = False
                    for search_pos in searched_positions:
                        iter_field = Field.create_from_parameter(self.world, search_pos, radius)
                        if field.is_overlap(iter_field):
                            flag_overlap = True
                            break

                    if flag_overlap:
                        continue

                    searched_positions.append(candidate)
                    print('Add Candidate', candidate)
                    print(len(searched_positions), candidate)
                    self.world.env.draw_line(self.current_pos, candidate, Color.Green)

            if cur_idx >= len(searched_positions): break

            self.world.env.draw_line(self.current_pos, searched_positions[cur_idx], Color.Blue)
            self.current_pos = searched_positions[cur_idx]
            cur_idx += 1

        return searched_positions

