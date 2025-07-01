"""
Utility Functions for Hierarchical Pathfinding A*

This module provides essential utility functions for the hierarchical pathfinding system,
including map generation, pathfinding algorithms, coordinate conversions, and geometric
operations. The module supports both low-level node-based pathfinding and high-level
graph-based pathfinding across different abstraction layers.

Key Features:
- Perlin noise map generation for procedural terrain
- A* pathfinding algorithms for both Node and GraphNode types
- Coordinate conversion and neighbor calculation utilities
- Line of sight (LOS) checking using raycast algorithm
- Graph node creation and management for hierarchical pathfinding
- Bresenham's line algorithm for drawing and path filling

The module operates at three levels:
- Level 0: Direct node-to-node connections within chunks
- Level 1: Chunk-level abstractions using entrance nodes
- Level 2: Mega-chunk level abstractions using cluster entrances

Usage:
    This module is primarily used by the main application and entity classes
    to perform pathfinding operations and manage the hierarchical graph structure.
"""

import noise
from settings import MAP_TILE_WIDTH, MAP_TILE_HEIGHT, MAP_CHUNK_SIZE
from data import Node, GraphNode, Cluster, Pos, chunks, Matrix_2D
import math
import heapq


def index_1d(x: int, y: int, width: int) -> int:
    """
    Converts 2D coordinates to 1D array index.
    
    This function performs a standard row-major order conversion from 2D coordinates
    to a 1D array index. This is commonly used for storing 2D grid data in 1D arrays
    for memory efficiency.
    
    Args:
        x (int): X coordinate (column)
        y (int): Y coordinate (row)
        width (int): Width of the 2D grid
        
    Returns:
        int: 1D array index corresponding to the 2D coordinates
        
    Example:
        >>> index_1d(2, 1, 5)
        7  # (1 * 5) + 2 = 7
    """
    return y * width + x


def get_neighbors(x: int, y: int) -> list[Pos]:
    """
    Gets all valid neighboring positions for a given coordinate.
    
    This function returns the four cardinal directions (up, down, left, right)
    that are within the bounds of the map. Diagonal movement is not supported
    in this implementation.
    
    Args:
        x (int): X coordinate of the center position
        y (int): Y coordinate of the center position
        
    Returns:
        list[Pos]: List of valid neighboring (x, y) coordinate tuples
        
    Example:
        >>> get_neighbors(5, 5)
        [(6, 5), (4, 5), (5, 6), (5, 4)]
    """
    neighbors = []
    for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
        if 0 <= nx < MAP_TILE_WIDTH and 0 <= ny < MAP_TILE_HEIGHT:
            neighbors.append((nx, ny))
    return neighbors


def generate_perlin_noise_map(
    width: int,
    height: int,
    noise_scale: float = 10.0,
    threshold: float = 0.0,
    octaves: int = 3,
) -> Matrix_2D:
    """
    Generates a 2D map using Perlin noise for procedural terrain generation.
    
    This function creates a height map using Perlin noise, which provides natural-looking
    terrain patterns. The noise values are thresholded to create binary walkable/impassable
    areas. Lower noise values become walkable (0), while higher values become impassable (1).
    
    Args:
        width (int): Width of the map in tiles
        height (int): Height of the map in tiles
        noise_scale (float): Scale factor for noise sampling (default: 10.0)
                            Higher values create smoother, more gradual terrain
        threshold (float): Threshold value for determining walkable areas (default: 0.0)
                          Values below threshold become walkable (0), above become impassable (1)
        octaves (int): Number of noise octaves for detail (default: 3)
                      More octaves add finer detail but increase computation time
        
    Returns:
        Matrix_2D: 2D list where 0 = walkable, 1 = impassable
        
    Example:
        >>> map_data = generate_perlin_noise_map(100, 100, noise_scale=15.0, threshold=0.2)
        >>> len(map_data)  # Height
        100
        >>> len(map_data[0])  # Width
        100
    """
    map_data: Matrix_2D = []

    for y in range(height):
        row = []
        for x in range(width):
            perlin_value = noise.pnoise2(
                x / noise_scale, y / noise_scale, octaves=octaves, persistence=0.9
            )
            if perlin_value < threshold:
                row.append(0)  # Walkable
            else:
                row.append(1)  # Impassable
        map_data.append(row)

    return map_data


def get_node(pos: Pos) -> Node | None:
    """
    Retrieves a Node object from the hierarchical chunk structure.
    
    This function navigates the chunk hierarchy to find a Node at the specified
    position. It first determines which chunk contains the position, then looks
    up the node within that chunk's node collection.
    
    Args:
        pos (Pos): (x, y) coordinate tuple of the node to retrieve
        
    Returns:
        Node | None: The Node object if found, None if the position is not walkable
                    or outside the map bounds
        
    Example:
        >>> node = get_node((10, 15))
        >>> if node:
        ...     print(f"Found node at ({node.x}, {node.y})")
    """
    chunk_x, chunk_y = pos[0] // MAP_CHUNK_SIZE, pos[1] // MAP_CHUNK_SIZE
    chunk = chunks.get((chunk_x, chunk_y))
    if chunk:
        return chunk.nodes.get(pos, None)
    return None


def heuristic_node(start: Node, end: Node) -> float:
    """
    Calculates the squared Euclidean distance between two nodes.
    
    This heuristic function is used in A* pathfinding to estimate the cost
    from one node to another. Using squared distance avoids the computational
    cost of square root while maintaining the same relative ordering for
    pathfinding decisions.
    
    Args:
        start (Node): Starting node
        end (Node): Target node
        
    Returns:
        float: Squared Euclidean distance between the nodes
        
    Example:
        >>> start = Node(0, 0)
        >>> end = Node(3, 4)
        >>> heuristic_node(start, end)
        25.0  # 3² + 4² = 9 + 16 = 25
    """
    return (start.x - end.x) ** 2 + (start.y - end.y) ** 2


def heuristic_graph_node(start: GraphNode, end: GraphNode) -> float:
    """
    Calculates the Euclidean distance between two graph nodes.
    
    This heuristic function is used in high-level A* pathfinding between
    GraphNode objects (entrance nodes). Unlike the node heuristic, this
    uses actual Euclidean distance since GraphNodes represent larger-scale
    pathfinding and the computational cost is acceptable.
    
    Args:
        start (GraphNode): Starting graph node
        end (GraphNode): Target graph node
        
    Returns:
        float: Euclidean distance between the graph nodes
        
    Example:
        >>> start = GraphNode((0, 0), abstraction)
        >>> end = GraphNode((3, 4), abstraction)
        >>> heuristic_graph_node(start, end)
        5.0  # √(3² + 4²) = √(9 + 16) = √25 = 5
    """
    return math.sqrt((start.x - end.x) ** 2 + (start.y - end.y) ** 2)


def reconstruct_graph_path(
    came_from: dict[GraphNode, GraphNode], end_node: GraphNode
) -> list[GraphNode]:
    """
    Reconstructs a path from the A* algorithm's came_from dictionary.
    
    This function takes the came_from dictionary generated during A* pathfinding
    and reconstructs the complete path from start to end by following the
    parent pointers backwards from the end node.
    
    Args:
        came_from (dict[GraphNode, GraphNode]): Dictionary mapping each node to its parent
        end_node (GraphNode): The final node in the path
        
    Returns:
        list[GraphNode]: Complete path from start to end node
        
    Example:
        >>> path = reconstruct_graph_path(came_from_dict, end_node)
        >>> print(f"Path length: {len(path)}")
    """
    path: list[GraphNode] = [end_node]
    current_node = end_node

    while current_node in came_from:
        current_node = came_from[current_node]
        path.append(current_node)

    path.reverse()
    return path


def reconstruct_node_path(came_from: dict[Node, Node], end_node: Node) -> list[Node]:
    """
    Reconstructs a path from the A* algorithm's came_from dictionary for Node objects.
    
    This function works identically to reconstruct_graph_path but operates on
    Node objects instead of GraphNode objects. It's used for low-level pathfinding
    within chunks.
    
    Args:
        came_from (dict[Node, Node]): Dictionary mapping each node to its parent
        end_node (Node): The final node in the path
        
    Returns:
        list[Node]: Complete path from start to end node
        
    Example:
        >>> path = reconstruct_node_path(came_from_dict, end_node)
        >>> print(f"Path length: {len(path)}")
    """
    path: list[Node] = [end_node]
    current_node = end_node

    while current_node in came_from:
        current_node = came_from[current_node]
        path.append(current_node)

    path.reverse()
    return path


def NodeAStar(start_node: Node, end_node: Node) -> list[Node]:
    """
    Performs A* pathfinding between two Node objects.
    
    This function implements the A* pathfinding algorithm for low-level navigation
    within chunks. It uses direct node connections and the squared Euclidean distance
    heuristic. The algorithm finds the shortest path between two nodes while avoiding
    impassable areas.
    
    Args:
        start_node (Node): Starting node for pathfinding
        end_node (Node): Target node for pathfinding
        
    Returns:
        list[Node]: List of nodes representing the path from start to end,
                   empty list if no path exists
        
    Example:
        >>> path = NodeAStar(start_node, end_node)
        >>> if path:
        ...     print(f"Found path with {len(path)} nodes")
        ... else:
        ...     print("No path found")
    """
    if not start_node or not end_node:
        return []
    if (end_node.x, end_node.y) in get_neighbors(start_node.x, start_node.y):
        return [start_node, end_node]

    open_list = []
    heapq.heappush(
        open_list, (0, id(start_node), start_node)
    )

    closed_list: set[Node] = set()

    g_scores: dict[Node, float] = {start_node: 0.0}
    h_scores: dict[Node, float] = {start_node: heuristic_node(start_node, end_node)}
    f_scores: dict[Node, float] = {
        start_node: g_scores[start_node] + h_scores[start_node]
    }
    came_from: dict[Node, Node] = {}

    while open_list:
        _, _, current_node = heapq.heappop(open_list)

        if current_node == end_node:
            return reconstruct_node_path(came_from, current_node)

        closed_list.add(current_node)

        for neighbor in current_node.connections.values():
            if neighbor in closed_list:
                continue

            tentative_g_score = g_scores[current_node] + heuristic_node(
                current_node, neighbor
            )

            if tentative_g_score < g_scores.get(neighbor, float("inf")):
                came_from[neighbor] = current_node
                g_scores[neighbor] = tentative_g_score
                h_scores[neighbor] = heuristic_node(neighbor, end_node)
                f_scores[neighbor] = g_scores[neighbor] + h_scores[neighbor]

                heapq.heappush(
                    open_list, (f_scores[neighbor], id(neighbor), neighbor)
                )

    return []


def create_graph_node(node: Node, layer: int) -> GraphNode | None:
    """
    Creates a GraphNode for hierarchical pathfinding at the specified layer.
    
    This function creates a GraphNode (entrance node) from a regular Node and
    establishes connections with other entrance nodes at the same layer. The
    GraphNode serves as a high-level abstraction for pathfinding between
    different regions of the map.
    
    Args:
        node (Node): The base node to convert to a GraphNode
        layer (int): The abstraction layer (1 for chunk level, 2 for mega-chunk level)
        
    Returns:
        GraphNode: The created GraphNode with established connections, or None if invalid
        
    Example:
        >>> graph_node = create_graph_node(node, 1)  # Chunk level
        >>> if graph_node:
        ...     print(f"Created entrance at ({graph_node.x}, {graph_node.y})")
    """

    abstraction = node.abstraction
    if not abstraction:
            return
    graph_node = GraphNode((node.x, node.y), abstraction)

    if layer == 1:
        abstraction.entrances[(graph_node.x, graph_node.y)] = graph_node

        for entrance in abstraction.entrances.values():
            if entrance != graph_node:
                graph_node.chunk_connections[(entrance.x, entrance.y)] = entrance
                entrance.chunk_connections[(graph_node.x, graph_node.y)] = graph_node

        return graph_node

    elif layer == 2:
        cluster = abstraction.cluster
        if not cluster:
            return
        cluster.entrances[(graph_node.x, graph_node.y)] = graph_node

        for entrance in cluster.entrances.values():
            if entrance != graph_node:
                graph_node.mega_chunk_connections[(entrance.x, entrance.y)] = entrance
                entrance.mega_chunk_connections[(graph_node.x, graph_node.y)] = (
                    graph_node
                )
        return graph_node


def delete_graph_node(graph_node: GraphNode, layer: int) -> None:
    """
    Removes a GraphNode and cleans up its connections.
    
    This function removes a GraphNode from the hierarchical structure and
    cleans up all connections to and from other GraphNodes. This is typically
    used when temporary GraphNodes are created for pathfinding and need to
    be cleaned up afterward.
    
    Args:
        graph_node (GraphNode): The GraphNode to remove
        layer (int): The abstraction layer (1 for chunk level, 2 for mega-chunk level)
        
    Example:
        >>> delete_graph_node(temp_graph_node, 1)  # Remove chunk-level entrance
    """
    if layer == 1:
        abstraction = graph_node.parent
        del abstraction.entrances[(graph_node.x, graph_node.y)]

        for entrance in abstraction.entrances.values():
            if entrance != graph_node:
                del entrance.chunk_connections[(graph_node.x, graph_node.y)]
                del graph_node.chunk_connections[(entrance.x, entrance.y)]
    elif layer == 2:
        abstraction = graph_node.parent
        if not abstraction.cluster:
            return
        
        cluster: Cluster = abstraction.cluster
        del cluster.entrances[(graph_node.x, graph_node.y)]

        for entrance in cluster.entrances.values():
            if entrance != graph_node:
                del entrance.mega_chunk_connections[(graph_node.x, graph_node.y)]
                del graph_node.mega_chunk_connections[(entrance.x, entrance.y)]


def GraphNodeAStar(
    start_graph_node: GraphNode, end_graph_node: GraphNode, layer: int
) -> list[GraphNode]:
    """
    Performs A* pathfinding between two GraphNode objects.
    
    This function implements A* pathfinding for high-level navigation between
    entrance nodes (GraphNodes). It operates at either the chunk level (layer 1)
    or mega-chunk level (layer 2) depending on the specified layer parameter.
    
    Args:
        start_graph_node (GraphNode): Starting entrance node
        end_graph_node (GraphNode): Target entrance node
        layer (int): The abstraction layer (1 for chunk level, 2 for mega-chunk level)
        
    Returns:
        list[GraphNode]: List of GraphNodes representing the high-level path,
                        empty list if no path exists
        
    Example:
        >>> path = GraphNodeAStar(start_entrance, end_entrance, 2)  # Mega-chunk level
        >>> print(f"High-level path has {len(path)} entrances")
    """
    open_list: list[GraphNode] = [start_graph_node]
    closed_list: set[GraphNode] = set()

    g_scores: dict[GraphNode, float] = {start_graph_node: 0.0}
    h_scores: dict[GraphNode, float] = {
        start_graph_node: heuristic_graph_node(start_graph_node, end_graph_node)
    }
    f_scores: dict[GraphNode, float] = {
        start_graph_node: g_scores[start_graph_node] + h_scores[start_graph_node]
    }

    came_from: dict[GraphNode, GraphNode] = {}

    while open_list:
        current_node = min(
            open_list, key=lambda graph_node: f_scores.get(graph_node, float("inf"))
        )

        if current_node == end_graph_node:
            graph_path = reconstruct_graph_path(came_from, end_graph_node)
            return graph_path

        open_list.remove(current_node)
        closed_list.add(current_node)

        if layer == 1:
            connections = current_node.chunk_connections
        elif layer == 2:
            connections = current_node.mega_chunk_connections

        for neighbor in connections.values():
            if neighbor in closed_list:
                continue

            tentative_g_score = g_scores[current_node] + heuristic_graph_node(
                current_node, neighbor
            )
            if tentative_g_score < g_scores.get(neighbor, float("inf")):
                came_from[neighbor] = current_node
                g_scores[neighbor] = tentative_g_score
                h_scores[neighbor] = heuristic_graph_node(neighbor, end_graph_node)
                f_scores[neighbor] = g_scores[neighbor] + h_scores[neighbor]

                if neighbor not in open_list:
                    open_list.append(neighbor)

    return []


def GraphAStar(start_node: Node, end_node: Node, layer: int) -> list[GraphNode]:
    """
    Performs hierarchical A* pathfinding between two nodes.
    
    This function provides a high-level interface for pathfinding that automatically
    handles the creation and cleanup of temporary GraphNodes. It first checks if
    the start and end nodes already have corresponding GraphNodes at the specified
    layer, and creates temporary ones if needed.
    
    Args:
        start_node (Node): Starting node for pathfinding
        end_node (Node): Target node for pathfinding
        layer (int): The abstraction layer (1 for chunk level, 2 for mega-chunk level)
        
    Returns:
        list[GraphNode]: List of GraphNodes representing the high-level path,
                        empty list if no path exists
        
    Example:
        >>> high_level_path = GraphAStar(start_node, end_node, 2)
        >>> print(f"Found {len(high_level_path)} high-level waypoints")
    """
    if not start_node or not end_node:
        return []
    start_node_abstract = start_node.abstraction
    end_node_abstract = end_node.abstraction

    if not start_node_abstract or not end_node_abstract:
        return []

    start_node_cluster = start_node_abstract.cluster
    end_node_cluster = end_node_abstract.cluster

    if not start_node_cluster or not end_node_cluster:
        return []

    created_start = False
    created_end = False

    start_graph_node = None
    if layer == 1:
        start_graph_node = start_node_abstract.entrances.get(
            (start_node.x, start_node.y)
        )
    elif layer == 2:
        start_graph_node = start_node_cluster.entrances.get(
            (start_node.x, start_node.y)
        )

    if not start_graph_node:
        start_graph_node = create_graph_node(start_node, layer)
        created_start = True

    end_graph_node = None
    if layer == 1:
        end_graph_node = end_node_abstract.entrances.get((end_node.x, end_node.y))
    elif layer == 2:
        end_graph_node = end_node_cluster.entrances.get((end_node.x, end_node.y))

    if not end_graph_node:
        end_graph_node = create_graph_node(end_node, layer)
        created_end = True

    if not start_graph_node or not end_graph_node:
        return []

    path = GraphNodeAStar(start_graph_node, end_graph_node, layer)
    if created_start:
        delete_graph_node(start_graph_node, layer)
    if created_end:
        delete_graph_node(end_graph_node, layer)

    return path


def raycast(start: Pos, end: Pos) -> bool:
    """
    Performs line of sight (LOS) checking using Bresenham's line algorithm.
    
    This function checks if there is a clear line of sight between two positions
    by tracing a line and checking if all positions along the line are walkable
    (contain valid nodes). It uses Bresenham's line algorithm for efficient
    pixel-perfect line tracing.
    
    Args:
        start (Pos): Starting position (x, y)
        end (Pos): Ending position (x, y)
        
    Returns:
        bool: True if there is a clear line of sight, False if any position
              along the line is impassable or outside the map bounds
        
    Example:
        >>> has_los = raycast((0, 0), (5, 5))
        >>> if has_los:
        ...     print("Direct path available")
        ... else:
        ...     print("Obstacle in the way")
    """
    x0, y0 = start
    x1, y1 = end

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        node = get_node((x0, y0))
        if not node:
            return False
        if (x0, y0) == (x1, y1):
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return True


def fill_line_with_action(start_pos: Pos, end_pos: Pos, action_func):
    """
    Applies an action function to all positions along a line.
    
    This function uses Bresenham's line algorithm to trace a line between two
    positions and applies the specified action function to each position along
    the line. This is commonly used for drawing operations like adding or
    removing nodes in a continuous line.
    
    Args:
        start_pos (Pos): Starting position (x, y)
        end_pos (Pos): Ending position (x, y)
        action_func: Function to apply to each position along the line.
                    Should accept a Pos tuple as its only argument.
        
    Example:
        >>> def add_node_at(pos):
        ...     add_node(pos)
        >>> fill_line_with_action((0, 0), (5, 5), add_node_at)
    """
    x0, y0 = start_pos
    x1, y1 = end_pos

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        action_func((x0, y0))

        if (x0, y0) == (x1, y1):
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
