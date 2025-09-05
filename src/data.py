import numpy as np
from enum import Enum
from settings import MAP_CHUNK_SIZE, CHUNK_SCALE_FACTOR, CHUNK_PIXEL_SIZE, TILE_SIZE
import pygame

class NodeType(Enum):
    PASSABLE = 0
    IMPASSABLE = 1

Pos = tuple[int, int]  # (x, y) coordinate tuple
Matrix_2D = np.ndarray  # 2D matrix of integers

class Chunk:
    def draw_nodes_surface(self, tile_size, node_color=(33,33,33,255), bg_color=(0,0,0,0)):
        """Draw the chunk's nodes onto its surface (for layer 0 chunks)."""
        if self.depth != 0 or not hasattr(self, 'nodes') or self.nodes is None:
            return
        import pygame
        h, w = self.nodes.shape
        self.surface = pygame.Surface((w * tile_size, h * tile_size), flags=pygame.SRCALPHA)
        self.surface.fill(bg_color)
        for y in range(h):
            for x in range(w):
                if self.nodes[y, x]:
                    rect = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
                    self.surface.fill(node_color, rect)

    def __init__(self, x: int, y: int, depth: int, nodes: Matrix_2D | None) -> None:
        self.x: int = x
        self.y: int = y
        self.depth = depth
        self.abstractions: set[Abstraction] = set()
        if self.depth == 0 and isinstance(nodes, Matrix_2D):
            self.nodes: Matrix_2D = nodes
        self.parent: Chunk | None = None
        self.children: dict[Pos, Chunk] = {}
        self.surface = None  # For layer 0 chunk rendering

    def tile_count(self) -> int:
        if self.depth == 0:
            if self.nodes is not None:
                return int(self.nodes.sum())
            return 0
        # recurse if children exist
        if self.children:
            return sum(child.tile_count() for child in self.children.values())
        return 0
    
class Abstraction:
    def __init__(self, chunk: Chunk):
        self.chunk = chunk
        self.parent: Abstraction | None = None
        self.entrances: dict[Pos, GraphNode] = {}
        
class GraphNode:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.abstraction: None | Abstraction = None
        self.connections: dict[Pos, GraphNode] = {}  # Direct connections to neighbors

chunks: dict[int, dict[Pos, Chunk]] = {}