"""
Data Structures for Hierarchical Pathfinding System

This module defines the core data structures used in the hierarchical pathfinding system.
The system uses a three-level hierarchy to efficiently handle pathfinding in large maps:

Level 0: Individual Nodes
- Basic grid nodes with direct connections to neighbors
- Used for fine-grained pathfinding within small areas

Level 1: Abstractions and Chunks  
- Chunks group nodes into manageable regions
- Abstractions represent connected regions within chunks
- GraphNodes serve as entrance points between chunks

Level 2: Clusters and Mega-Chunks
- Mega-chunks group chunks into larger regions
- Clusters represent connected abstractions across multiple chunks
- Provides high-level pathfinding for long-distance navigation

The hierarchy allows the system to quickly find paths at the appropriate level of detail,
starting with high-level cluster navigation and refining to specific node paths as needed.
"""

import random
import pygame
import settings

# Type aliases for better code readability
Pos = tuple[int, int]  # (x, y) coordinate tuple
Color = tuple[int, int, int]  # RGB color tuple  
Matrix_2D = list[list[int]]  # 2D matrix of integers

# Generate a random color palette for visual distinction between different regions
# Each abstraction, cluster, chunk, and mega-chunk gets a unique color for visualization
COLOR_PALETTE = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(256)
]


class Abstraction:
    """
    Represents a connected region within a chunk for level 1 pathfinding.
    
    An abstraction groups together nodes that are connected within a chunk.
    It serves as an intermediate level between individual nodes and clusters,
    allowing for efficient pathfinding within chunk boundaries.
    
    Attributes:
        entrances: Dictionary mapping positions to GraphNode entrance points
        color_index: Index into the color palette for visualization
        cluster: Reference to the cluster this abstraction belongs to (level 2)
    """
    __slots__ = ("entrances", "color_index", "cluster")

    def __init__(self):
        """Initialize a new abstraction with empty entrances and random color."""
        self.entrances: dict[Pos, GraphNode] = {}
        self.color_index: int = random.randint(0, 255)
        self.cluster: Cluster | None = None

    @property
    def color(self) -> Color:
        """Get the RGB color for this abstraction from the color palette."""
        return COLOR_PALETTE[self.color_index]


class Cluster:
    """
    Represents a connected region spanning multiple chunks for level 2 pathfinding.
    
    A cluster groups together abstractions from different chunks that are connected
    through entrance points. This creates the highest level of abstraction in the
    hierarchical pathfinding system, allowing for efficient long-distance navigation.
    
    Attributes:
        entrances: Dictionary mapping positions to GraphNode entrance points
        color_index: Index into the color palette for visualization
        abstractions: Set of abstractions that belong to this cluster
    """
    __slots__ = ("entrances", "color_index", "abstractions")

    def __init__(self) -> None:
        """Initialize a new cluster with empty entrances and abstractions."""
        self.entrances: dict[Pos, GraphNode] = {}
        self.color_index: int = random.randint(0, 255)
        self.abstractions: set[Abstraction] = set()

    @property
    def color(self) -> Color:
        """Get the RGB color for this cluster from the color palette."""
        return COLOR_PALETTE[self.color_index]


class Chunk:
    """
    Represents a rectangular region of the map containing multiple nodes.
    
    A chunk is a level 1 container that groups together nodes into manageable
    regions. Each chunk contains nodes, abstractions (connected regions within
    the chunk), and a pygame surface for efficient rendering.
    
    Attributes:
        x: X coordinate of the chunk in chunk space
        y: Y coordinate of the chunk in chunk space
        nodes: Dictionary mapping positions to Node objects within this chunk
        abstractions: Set of abstractions (connected regions) within this chunk
        color_index: Index into the color palette for visualization
        surface: Pygame surface for rendering this chunk's nodes
    """
    __slots__ = ("x", "y", "nodes", "abstractions", "color_index", "surface")

    def __init__(self, x: int, y: int):
        """
        Initialize a new chunk at the specified coordinates.
        
        Args:
            x: X coordinate of the chunk in chunk space
            y: Y coordinate of the chunk in chunk space
        """
        self.x: int = x
        self.y: int = y
        self.nodes: dict[Pos, Node] = {}
        self.abstractions: set[Abstraction] = set()
        self.color_index: int = random.randint(0, 255)
        # Create a pygame surface for efficient rendering of this chunk
        self.surface = pygame.Surface(
            (settings.CHUNK_PIXEL_SIZE, settings.CHUNK_PIXEL_SIZE), pygame.SRCALPHA
        )

    @property
    def color(self) -> Color:
        """Get the RGB color for this chunk from the color palette."""
        return COLOR_PALETTE[self.color_index]

    def update_surface(self) -> None:
        """
        Update the chunk's rendering surface with all its nodes.
        
        This method clears the surface and redraws all nodes within the chunk.
        It's called when nodes are added or removed to keep the visualization
        up to date.
        """
        # Clear the surface with transparent background
        self.surface.fill((0, 0, 0, 0))

        # Draw each node as a rectangle on the surface
        for node in self.nodes.values():
            # Convert world coordinates to local chunk coordinates
            local_x = (node.x % settings.MAP_CHUNK_SIZE) * settings.TILE_SIZE
            local_y = (node.y % settings.MAP_CHUNK_SIZE) * settings.TILE_SIZE

            rect = pygame.Rect(local_x, local_y, settings.TILE_SIZE, settings.TILE_SIZE)
            pygame.draw.rect(self.surface, (33, 33, 33), rect, 0)


class MegaChunk:
    """
    Represents a large rectangular region containing multiple chunks.
    
    A mega-chunk is a level 2 container that groups together chunks into even
    larger regions. This creates the highest level of spatial organization in
    the hierarchical pathfinding system, allowing for efficient long-distance
    pathfinding across large areas of the map.
    
    Attributes:
        x: X coordinate of the mega-chunk in mega-chunk space
        y: Y coordinate of the mega-chunk in mega-chunk space
        chunks: Dictionary mapping chunk positions to Chunk objects
        clusters: Set of clusters (connected regions) within this mega-chunk
        color_index: Index into the color palette for visualization
    """
    __slots__ = ("x", "y", "chunks", "clusters", "color_index")

    def __init__(self, x: int, y: int):
        """
        Initialize a new mega-chunk at the specified coordinates.
        
        Args:
            x: X coordinate of the mega-chunk in mega-chunk space
            y: Y coordinate of the mega-chunk in mega-chunk space
        """
        self.x: int = x
        self.y: int = y
        self.chunks: dict[Pos, Chunk] = {}
        self.clusters: set[Cluster] = set()
        self.color_index: int = random.randint(0, 255)

    @property
    def color(self) -> Color:
        """Get the RGB color for this mega-chunk from the color palette."""
        return COLOR_PALETTE[self.color_index]


class Node:
    """
    Represents a single walkable position in the pathfinding grid.
    
    A node is the most basic unit in the hierarchical pathfinding system (level 0).
    Each node represents a walkable tile position and maintains connections to
    neighboring nodes. Nodes are grouped into abstractions for higher-level
    pathfinding.
    
    Attributes:
        x: X coordinate of the node in world space
        y: Y coordinate of the node in world space
        connections: Dictionary mapping neighbor positions to connected Node objects
        abstraction: Reference to the abstraction this node belongs to (level 1)
    """
    __slots__ = ("x", "y", "connections", "abstraction")

    def __init__(self, x: int, y: int) -> None:
        """
        Initialize a new node at the specified world coordinates.
        
        Args:
            x: X coordinate of the node in world space
            y: Y coordinate of the node in world space
        """
        self.x: int = x
        self.y: int = y
        self.connections: dict[Pos, Node] = {}
        self.abstraction: Abstraction | None = None


class GraphNode:
    """
    Represents an entrance point between different regions in the hierarchy.
    
    A GraphNode serves as a connection point between abstractions, chunks, and
    clusters. It's used for pathfinding at higher levels of the hierarchy,
    allowing entities to navigate between different regions efficiently.
    
    GraphNodes can have two types of connections:
    - chunk_connections: Links to other GraphNodes within the same abstraction
    - mega_chunk_connections: Links to GraphNodes in other abstractions/clusters
    
    Attributes:
        x: X coordinate of the entrance point in world space
        y: Y coordinate of the entrance point in world space
        parent: Reference to the abstraction this GraphNode belongs to
        chunk_connections: Dictionary mapping positions to connected GraphNodes within the same abstraction
        mega_chunk_connections: Dictionary mapping positions to connected GraphNodes in other abstractions
    """
    __slots__ = (
        "x",
        "y",
        "parent",
        "chunk_connections",
        "mega_chunk_connections",
    )

    def __init__(self, pos: Pos, parent: Abstraction) -> None:
        """
        Initialize a new GraphNode at the specified position.
        
        Args:
            pos: (x, y) tuple of the entrance point coordinates in world space
            parent: The abstraction this GraphNode belongs to
        """
        self.x, self.y = pos
        self.parent: Abstraction = parent
        self.chunk_connections: dict[Pos, GraphNode] = {}
        self.mega_chunk_connections: dict[Pos, GraphNode] = {}

# Global data structures for storing the hierarchical map organization
# These dictionaries serve as the main containers for all map data

# Dictionary mapping chunk coordinates to Chunk objects
# Key: (chunk_x, chunk_y) tuple in chunk space
# Value: Chunk object containing nodes and abstractions
chunks: dict[Pos, Chunk] = {}

# Dictionary mapping mega-chunk coordinates to MegaChunk objects  
# Key: (mega_chunk_x, mega_chunk_y) tuple in mega-chunk space
# Value: MegaChunk object containing chunks and clusters
mega_chunks: dict[Pos, MegaChunk] = {}
