from typing import Optional, Tuple, Dict, Set
import random

Pos = Tuple[int, int]
Color = Tuple[int, int, int]

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
    __slots__ = ("entrances", "color_index", "meta_cluster", "abstractions")

    def __init__(self) -> None:
        self.entrances: Dict[Pos, "GraphNode"] = {}
        self.color_index: int = random.randint(0, 255)
        self.meta_cluster: Optional["MetaCluster"] = None
        self.abstractions: Set[Abstraction] = set()

    @property
    def color(self) -> Color:
        return COLOR_PALETTE[self.color_index]


class MetaCluster:
    __slots__ = ("entrances", "color_index", "clusters")

    def __init__(self) -> None:
        self.entrances: Dict[Pos, "GraphNode"] = {}
        self.color_index: int = random.randint(0, 255)
        self.clusters: Set[Cluster] = set()

    @property
    def color(self) -> Color:
        return COLOR_PALETTE[self.color_index]


class Chunk:
    __slots__ = ("x", "y", "nodes", "abstractions", "color_index")

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.nodes: Dict[Pos, "Node"] = {}
        self.abstractions: Set[Abstraction] = set()
        self.color_index: int = random.randint(0, 255)

    @property
    def color(self) -> Color:
        return COLOR_PALETTE[self.color_index]


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


class GigaChunk:
    __slots__ = ("x", "y", "mega_chunks", "meta_clusters", "color_index")

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.mega_chunks: Dict[Pos, MegaChunk] = {}
        self.meta_clusters: Set[MetaCluster] = set()
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
        "giga_chunk_connections",
    )

    def __init__(self, pos: Pos, parent: Abstraction) -> None:
        self.x, self.y = pos
        self.parent: Abstraction = parent
        self.chunk_connections: Dict[Pos, "GraphNode"] = {}
        self.mega_chunk_connections: Dict[Pos, "GraphNode"] = {}
        self.giga_chunk_connections: Dict[Pos, "GraphNode"] = {}


# Global dictionaries for chunk storage
chunks: Dict[Pos, Chunk] = {}
mega_chunks: Dict[Pos, MegaChunk] = {}
giga_chunks: Dict[Pos, GigaChunk] = {}
