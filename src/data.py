from typing import Optional, Tuple, Dict, Set
import random
import pygame
import settings

Pos = tuple[int, int]
Color = tuple[int, int, int]
Matrix_2D = list[list[int]]

COLOR_PALETTE = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(256)
]


class Abstraction:
    __slots__ = ("entrances", "color_index", "cluster")

    def __init__(self):
        self.entrances: Dict[Pos, "GraphNode"] = {}
        self.color_index: int = random.randint(0, 255)
        self.cluster: Optional["Cluster"] = None

    @property
    def color(self) -> Color:
        return COLOR_PALETTE[self.color_index]


class Cluster:
    __slots__ = ("entrances", "color_index", "abstractions")

    def __init__(self) -> None:
        self.entrances: Dict[Pos, "GraphNode"] = {}
        self.color_index: int = random.randint(0, 255)
        self.abstractions: Set[Abstraction] = set()

    @property
    def color(self) -> Color:
        return COLOR_PALETTE[self.color_index]


class Chunk:
    __slots__ = ("x", "y", "nodes", "abstractions", "color_index", "surface")

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.nodes: Dict[Pos, "Node"] = {}
        self.abstractions: Set[Abstraction] = set()
        self.color_index: int = random.randint(0, 255)
        self.surface = pygame.Surface(
            (settings.CHUNK_PIXEL_SIZE, settings.CHUNK_PIXEL_SIZE), pygame.SRCALPHA
        )

    @property
    def color(self) -> Color:
        return COLOR_PALETTE[self.color_index]

    def update_surface(self) -> None:
        self.surface.fill((0, 0, 0, 0))

        for node in self.nodes.values():
            local_x = (node.x % settings.MAP_CHUNK_SIZE) * settings.TILE_SIZE
            local_y = (node.y % settings.MAP_CHUNK_SIZE) * settings.TILE_SIZE

            rect = pygame.Rect(local_x, local_y, settings.TILE_SIZE, settings.TILE_SIZE)
            pygame.draw.rect(self.surface, (33, 33, 33), rect, 0)


class MegaChunk:
    __slots__ = ("x", "y", "chunks", "clusters", "color_index")

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.chunks: Dict[Pos, Chunk] = {}
        self.clusters: Set[Cluster] = set()
        self.color_index: int = random.randint(0, 255)

    @property
    def color(self) -> Color:
        return COLOR_PALETTE[self.color_index]


class Node:
    __slots__ = ("x", "y", "connections", "abstraction")

    def __init__(self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y
        self.connections: Dict[Pos, "Node"] = {}
        self.abstraction: Optional[Abstraction] = None


class GraphNode:
    __slots__ = (
        "x",
        "y",
        "parent",
        "chunk_connections",
        "mega_chunk_connections",
    )

    def __init__(self, pos: Pos, parent: Abstraction) -> None:
        self.x, self.y = pos
        self.parent: Abstraction = parent
        self.chunk_connections: Dict[Pos, "GraphNode"] = {}
        self.mega_chunk_connections: Dict[Pos, "GraphNode"] = {}

chunks: Dict[Pos, Chunk] = {}
mega_chunks: Dict[Pos, MegaChunk] = {}
