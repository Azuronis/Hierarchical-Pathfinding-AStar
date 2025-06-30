from data import Node
from utility import GraphAStar, get_node, NodeAStar, raycast
from settings import TILE_SIZE


class Entity:
    def __init__(self, node_x: int, node_y: int, speed: float = 30):
        self.x: float = node_x * TILE_SIZE
        self.y: float = node_y * TILE_SIZE
        self.speed = speed

        self.level_2_path = []  # Mega Chunk path (Layer 2)
        self.level_1_path = []  # Chunk path (Layer 1)
        self.level_0_path = []  # Tile path (Layer 0)
        self.target_node = None
        self.goal_node = None

    @property
    def current_node(self):
        node_x = round(self.x / TILE_SIZE)
        node_y = round(self.y / TILE_SIZE)
        return get_node((node_x, node_y))

    def pathfind(self, end_node: Node) -> bool:
        self.goal_node = end_node
        raycast_valid = raycast(
            (self.current_node.x, self.current_node.y), (end_node.x, end_node.y)
        )
        if raycast_valid:
            self.target_node = self.goal_node
            self.level_0_path.clear()
            self.level_1_path.clear()
            self.level_2_path.clear()
            return True

        self.level_2_path.clear()
        self.level_1_path.clear()
        self.level_0_path.clear()

        self.level_2_path = GraphAStar(self.current_node, end_node, layer=2)

        if not self.level_2_path:
            return False

        self.compute_next_level_1_path()

        return True

    def compute_next_level_1_path(self):

        next_graph_node = self.level_2_path.pop(0)
        node = get_node((next_graph_node.x, next_graph_node.y))

        self.level_1_path = GraphAStar(self.current_node, node, layer=1)

        if self.level_1_path:
            self.compute_next_level_0_path()

    def compute_next_level_0_path(self):
        next_node = self.level_1_path.pop(0)
        node = get_node((next_node.x, next_node.y))

        self.level_0_path = NodeAStar(self.current_node, node)

        if self.level_0_path:
            self.target_node = self.level_0_path.pop(0)

    def update(self, dt: float):
        if self.target_node:
            self.move_toward_target(dt)
        else:
            if self.level_1_path:
                self.compute_next_level_0_path()
            else:
                if self.level_2_path:
                    self.compute_next_level_1_path()

    def move_toward_target(self, dt: float):
        if not self.target_node:
            return

        target_x, target_y = (
            self.target_node.x * TILE_SIZE,
            self.target_node.y * TILE_SIZE,
        )
        dx = target_x - self.x
        dy = target_y - self.y
        distance = (dx**2 + dy**2) ** 0.5

        if distance < 0.5:
            if self.target_node == self.goal_node:
                return
            if self.level_0_path:
                self.target_node = self.level_0_path.pop(0)
            else:
                self.target_node = None
        else:
            self.x += (dx / distance) * self.speed * dt
            self.y += (dy / distance) * self.speed * dt
