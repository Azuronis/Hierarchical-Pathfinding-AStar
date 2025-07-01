"""
Hierarchical Pathfinding A* Visualization

This module implements a hierarchical pathfinding system with three levels of abstraction:
- Level 0: Individual nodes and their direct connections
- Level 1: Chunk-based abstractions with entrances and connections
- Level 2: Mega-chunk clusters with high-level pathfinding

The system uses a chunked approach where the world is divided into chunks (smaller regions)
and mega-chunks (larger regions). Each level provides different granularity for pathfinding,
allowing for navigation through large maps.

Key Features:
- Procedural map generation using Perlin noise
- Real-time pathfinding visualization
- Interactive map editing (add/remove nodes)
- Multi-level pathfinding with visual feedback
- Camera panning and zoom controls

Usage:
    Run this module directly to start the visualization. Use mouse clicks to set start/end
    points, WASD to pan the camera, and Shift+drag to add/remove nodes.
"""

from settings import (
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    TILE_SIZE,
    BACKGROUND_COLOR,
    NODE_COLOR,
    CONNECTION_COLOR,
    ENTRANCE_COLOR,
    GRID_COLOR,
    START_NODE_COLOR,
    END_NODE_COLOR,
    ENTITY_COLOR,
    TARGET_NODE_COLOR,
    PATH_LEVEL_0_COLOR,
    PATH_LEVEL_1_COLOR,
    PATH_LEVEL_2_COLOR,
    ENTRANCE_INDICATOR_COLOR,
    MAP_CHUNK_SIZE,
    ENTRANCE_SPACING,
    CHUNK_PIXEL_SIZE,
    SURFACE_WIDTH,
    SURFACE_HEIGHT,
    MEGA_CHUNK_SIZE,
    MEGA_CHUNK_PIXEL_SIZE,
    SCREEN_SCALE_FACTOR,
    SURFACE_OFFSET,
    PAN_SPEED,
    FPS,
    FONT_SIZE,
    UI_TEXT_OFFSET,
    NOISE_SCALE,
    NOISE_THRESHOLD,
    NOISE_OCTAVES,
    PATH_LINE_WIDTH,
    PATH_CIRCLE_RADIUS,
)
from utility import (
    index_1d,
    get_neighbors,
    generate_perlin_noise_map,
    get_node,
    fill_line_with_action,
)
from data import (
    Node,
    GraphNode,
    Abstraction,
    Cluster,
    Chunk,
    MegaChunk,
    Pos,
    Color,
    PosPairs,
    chunks,
    mega_chunks,
)
from entity import Entity
import pygame
import time
from collections import deque
import ctypes


pygame.font.init()

user32 = ctypes.windll.user32
screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
WIDTH, HEIGHT = (
    screen_width // SCREEN_SCALE_FACTOR,
    screen_height // SCREEN_SCALE_FACTOR,
)

screen = pygame.display.set_mode((WIDTH, HEIGHT), vsync=True)

OFFSET = SURFACE_OFFSET
OFFSET_WIDTH = SURFACE_WIDTH + OFFSET
OFFSET_HEIGHT = SURFACE_HEIGHT + OFFSET

# Create separate surfaces for different rendering layers
connection_surface = pygame.Surface((OFFSET_WIDTH, OFFSET_HEIGHT), pygame.SRCALPHA)
abstraction_surface = pygame.Surface((OFFSET_WIDTH, OFFSET_HEIGHT), pygame.SRCALPHA)
chunk_line_surface = pygame.Surface((OFFSET_WIDTH, OFFSET_HEIGHT), pygame.SRCALPHA)
chunk_entrance_surface = pygame.Surface((OFFSET_WIDTH, OFFSET_HEIGHT), pygame.SRCALPHA)
cluster_color_surface = pygame.Surface((OFFSET_WIDTH, OFFSET_HEIGHT), pygame.SRCALPHA)
mega_chunk_entrance_surface = pygame.Surface(
    (OFFSET_WIDTH, OFFSET_HEIGHT), pygame.SRCALPHA
)
mega_connection_surface = pygame.Surface((OFFSET_WIDTH, OFFSET_HEIGHT), pygame.SRCALPHA)
mega_chunk_line_surface = pygame.Surface((OFFSET_WIDTH, OFFSET_HEIGHT), pygame.SRCALPHA)

# Generate the procedural map using Perlin noise
print("Generating map...")
total_time = time.time()

# Create a 2D noise map using Perlin noise algorithm
map_data = generate_perlin_noise_map(
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    noise_scale=NOISE_SCALE,
    threshold=NOISE_THRESHOLD,
    octaves=NOISE_OCTAVES,
)

# Flatten the 2D map into a 1D list for easier processing
map_data_1d = [tile for row in map_data for tile in row]

# Extract positions where the noise value is 0 (walkable areas)
# These will become the nodes in our pathfinding graph
zero_positions = {
    (x, y)
    for y in range(MAP_TILE_HEIGHT)
    for x in range(MAP_TILE_WIDTH)
    if map_data_1d[index_1d(x, y, MAP_TILE_WIDTH)] == 0
}



def generate_tiles(positions: set[Pos]):
    """
    This function takes all walkable positions from the generated map and assigns them
    to appropriate chunks and mega-chunks. Each position becomes a Node in the pathfinding
    graph, and chunks/mega-chunks are created as needed to contain these nodes.
    
    Args:
        positions (set[Pos]): Set of (x, y) coordinate tuples representing walkable positions
    """
    for x, y in positions:
        # Calculate which chunk this position belongs to
        chunk_pos = x // MAP_CHUNK_SIZE, y // MAP_CHUNK_SIZE

        # Get or create the chunk for this position
        chunk = chunks.get(chunk_pos)
        if not chunk:
            chunk = Chunk(chunk_pos[0], chunk_pos[1])
            chunks[chunk_pos] = chunk
        
        # Add the node to the chunk
        chunk.nodes[(x, y)] = Node(x, y)

        # Calculate which mega-chunk this chunk belongs to
        mega_chunk_pos = (
            chunk_pos[0] // MEGA_CHUNK_SIZE,
            chunk_pos[1] // MEGA_CHUNK_SIZE,
        )
        
        # Get or create the mega-chunk for this chunk
        mega_chunk = mega_chunks.get(mega_chunk_pos)
        if not mega_chunk:
            mega_chunk = MegaChunk(mega_chunk_pos[0], mega_chunk_pos[1])
            mega_chunks[mega_chunk_pos] = mega_chunk
        
        # Add the chunk to the mega-chunk if not already present
        if chunk_pos not in mega_chunk.chunks:
            mega_chunk.chunks[chunk_pos] = chunk



def flood_fill_chunk(chunk: Chunk):
    """
    Performs flood fill algorithm to identify connected regions within a chunk.
    
    This function uses a breadth-first flood fill to identify groups of connected nodes
    within a chunk. Each connected group becomes an "abstraction" - a higher-level
    representation that can be used for pathfinding. The flood fill also establishes
    direct connections between neighboring nodes.
    
    The process:
    1. Clears any existing abstractions and connections
    2. Iterates through all nodes in the chunk
    3. For each unvisited node, starts a new flood fill
    4. Creates an abstraction for each connected region
    5. Establishes bidirectional connections between adjacent nodes

    Args:
        chunk (Chunk): The chunk object to process
    """
    # Clear any existing abstractions and node connections
    chunk.abstractions.clear()
    
    for node in chunk.nodes.values():
        node.abstraction = None
        node.connections.clear()

    chunk_nodes = list(chunk.nodes.keys())
    visited = set()

    # Process each node in the chunk
    while chunk_nodes:
        start_pos = chunk_nodes.pop()
        if start_pos in visited:
            continue

        # Start a new flood fill from this position
        queue = deque([start_pos])
        abstraction = Abstraction()  # Create new abstraction for this connected region
        chunk.abstractions.add(abstraction)

        # Flood fill this connected region
        while queue:
            node_pos = queue.popleft()
            if node_pos in visited:
                continue

            visited.add(node_pos)
            node = chunk.nodes[node_pos]
            node.abstraction = abstraction  # Assign node to this abstraction

            # Check all neighboring positions
            for neighbor_pos in get_neighbors(node.x, node.y):
                if neighbor_pos in chunk.nodes and neighbor_pos not in visited:
                    queue.append(neighbor_pos)
                    neighbor_node = chunk.nodes.get(neighbor_pos)
                    if neighbor_node:
                        # Establish bidirectional connection between nodes
                        node.connections[neighbor_pos] = neighbor_node
                        neighbor_node.connections[node_pos] = node


def make_chunk_entrances(chunk_pos: Pos, direction: str, entrance_size: int = 1) -> None:
    """
    Creates entrance points between a chunk and its neighbor along a specified direction.
    
    This function identifies potential entrance points along one edge of a chunk where
    it connects to a neighboring chunk. Entrances are created at regular intervals
    along continuous walkable paths, allowing entities to move between chunks.
    
    The process:
    1. Determines the neighbor chunk position based on direction
    2. Generates candidate pairs of connected nodes along the edge
    3. Groups continuous walkable paths together
    4. Creates entrance points at regular intervals within each group
    5. Establishes GraphNode connections between the entrances

    Args:
        chunk_pos (Pos): (x, y) coordinates of the chunk
        direction (str): Direction to check - "top", "bottom", "left", or "right"
    """
    chunk = chunks[chunk_pos]
    edge_range = range(MAP_CHUNK_SIZE)  # Edge range is the size of the chunk

    # Generate candidate pairs for all possible entrances along the specified side
    if direction == "top":
        neighbor_pos = (chunk_pos[0], chunk_pos[1] - 1)
        start_x, start_y = chunk_pos[0] * MAP_CHUNK_SIZE, chunk_pos[1] * MAP_CHUNK_SIZE
        neighbor_y = start_y - 1
        candidate_pairs = [
            ((start_x + x, start_y), (start_x + x, neighbor_y)) for x in edge_range
        ]
    elif direction == "bottom":
        neighbor_pos = (chunk_pos[0], chunk_pos[1] + 1)
        start_x, start_y = (
            chunk_pos[0] * MAP_CHUNK_SIZE,
            (chunk_pos[1] + 1) * MAP_CHUNK_SIZE - 1,
        )
        neighbor_y = start_y + 1
        candidate_pairs = [
            ((start_x + x, start_y), (start_x + x, neighbor_y)) for x in edge_range
        ]
    elif direction == "left":
        neighbor_pos = (chunk_pos[0] - 1, chunk_pos[1])
        start_x, start_y = chunk_pos[0] * MAP_CHUNK_SIZE, chunk_pos[1] * MAP_CHUNK_SIZE
        neighbor_x = start_x - 1
        candidate_pairs = [
            ((start_x, start_y + y), (neighbor_x, start_y + y)) for y in edge_range
        ]
    elif direction == "right":
        neighbor_pos = (chunk_pos[0] + 1, chunk_pos[1])
        start_x, start_y = (chunk_pos[0] + 1) * MAP_CHUNK_SIZE - 1, chunk_pos[
            1
        ] * MAP_CHUNK_SIZE
        neighbor_x = start_x + 1
        candidate_pairs = [
            ((start_x, start_y + y), (neighbor_x, start_y + y)) for y in edge_range
        ]
    else:
        return

    # Check if neighbor chunk exists
    neighbor_chunk = chunks.get(neighbor_pos)
    if not neighbor_chunk:
        return

    groups: list[PosPairs] = []
    current_group: PosPairs = []

    # Filter candidate pairs to only include walkable connections
    # Group continuous walkable paths together
    for pair in candidate_pairs:
        chunk_node_pos, neighbor_node_pos = pair
        if chunk.nodes.get(chunk_node_pos) and neighbor_chunk.nodes.get(
            neighbor_node_pos
        ):
            current_group.append(pair)
        else:
            if current_group:
                groups.append(current_group)
                current_group = []
    if current_group:
        groups.append(current_group)

    half_entrance_size = entrance_size // 2 # 1//2 = 0 for entrance size = 1

    # Create entrance points at regular intervals within each continuous group
    for group in groups:

        # TODO : https://harablog.wordpress.com/2009/02/05/hierarchical-clearance-based-pathfinding/
        # Finish the clearance based entrance spacing for connecting entrances. 
        
        valid_entrances: PosPairs = [] # valid entrances for variable entrance size
        group_length = len(group)
        if group_length < entrance_size: # if the entrance size is greater than the group length just skip the group
            continue
        start_index = half_entrance_size
        end_index = group_length - half_entrance_size

        for i in range(start_index, end_index):
            valid_entrances.append(group[i])
        group = valid_entrances # re-assign the group candidates with the valid ones.

        entrances: PosPairs = []
        for i in range(0, len(group), ENTRANCE_SPACING):
            group_slice = group[i : i + ENTRANCE_SPACING]
            middle_index = len(group_slice) // 2  # Grab the middle node
            entrances.append(group_slice[middle_index])

        # Create GraphNode connections for each entrance pair
        for chunk_node_pos, neighbor_node_pos in entrances:
            chunk_node = chunk.nodes.get(chunk_node_pos)
            neighbor_node = neighbor_chunk.nodes.get(neighbor_node_pos)

            if not chunk_node or not neighbor_node:
                continue
            
            chunk_abstract = chunk_node.abstraction
            neighbor_abstract = neighbor_node.abstraction

            if not chunk_abstract or not neighbor_abstract:
                continue
            
            # Create or get GraphNode for chunk entrance
            chunk_graph_node = chunk_abstract.entrances.get(chunk_node_pos)
            if not chunk_graph_node:
                chunk_graph_node = GraphNode(chunk_node_pos, chunk_abstract)
                chunk_abstract.entrances[(chunk_graph_node.x, chunk_graph_node.y)] = (
                    chunk_graph_node
                )

            # Create or get GraphNode for neighbor entrance
            neighbor_graph_node = neighbor_abstract.entrances.get(neighbor_node_pos)
            if not neighbor_graph_node:
                neighbor_graph_node = GraphNode(neighbor_node_pos, neighbor_abstract)
                neighbor_abstract.entrances[
                    (neighbor_graph_node.x, neighbor_graph_node.y)
                ] = neighbor_graph_node

            # Establish bidirectional connections between entrances
            chunk_graph_node.chunk_connections[
                (neighbor_graph_node.x, neighbor_graph_node.y)
            ] = neighbor_graph_node
            neighbor_graph_node.chunk_connections[
                (chunk_graph_node.x, chunk_graph_node.y)
            ] = chunk_graph_node


def generate_entrances(chunk: Chunk) -> None:
    """
    Generates entrance points for a chunk on all four sides.
    
    This function clears any existing entrances and then creates new entrance points
    along all four edges of the chunk (top, right, bottom, left). Each side is processed
    to find connections with neighboring chunks and establish entrance points for
    inter-chunk pathfinding.

    Args:
        chunk (Chunk): The chunk object to generate entrances for
    """
    # Clear all existing abstraction entrances since new graph nodes will be created
    for abstraction in chunk.abstractions:
        abstraction.entrances.clear()

    # Generate entrances for all four sides of the chunk
    cx, cy = chunk.x, chunk.y
    make_chunk_entrances((cx, cy), "top")
    make_chunk_entrances((cx, cy), "right")
    make_chunk_entrances((cx, cy), "bottom")
    make_chunk_entrances((cx, cy), "left")


def make_connections(chunk: Chunk) -> None:
    """
    Establishes connections between all entrance nodes within the same abstraction.
    
    This function creates a fully connected graph of entrance nodes within each
    abstraction. Every entrance node is connected to every other entrance node
    in the same abstraction, allowing for efficient pathfinding within the chunk.

    Args:
        chunk (Chunk): The chunk object to process
    """
    for abstraction in chunk.abstractions:
        entrances_list = list(abstraction.entrances.values())
        # Connect each entrance to all other entrances in the same abstraction
        for i, entrance in enumerate(entrances_list):
            for j, other_entrance in enumerate(entrances_list):
                if i != j:
                    entrance.chunk_connections[(other_entrance.x, other_entrance.y)] = (
                        other_entrance
                    )

def generate_clusters(mega_chunk: MegaChunk) -> set[Cluster]:
    """
    Creates clusters by flood-filling connected abstractions within a mega-chunk.
    
    This function groups connected abstractions within a mega-chunk into clusters.
    A cluster is a higher-level abstraction that contains multiple connected
    abstractions from different chunks. This creates the second level of the
    hierarchical pathfinding system.
    
    The process:
    1. Collects all entrance nodes from all abstractions in the mega-chunk
    2. Handles abstractions with no entrances (isolated regions)
    3. Uses flood fill to group connected abstractions into clusters
    4. Each cluster represents a connected region spanning multiple chunks

    Args:
        mega_chunk (MegaChunk): The mega-chunk object to process

    Returns:
        set[Cluster]: Set of newly created Cluster objects
    """
    all_entrances: set[GraphNode] = set()
    clusters: set[Cluster] = set()

    # Collect all entrances and handle isolated abstractions
    for chunk in mega_chunk.chunks.values():
        for abstraction in chunk.abstractions:
            if len(abstraction.entrances) == 0:
                # Create a cluster for abstractions with no entrances (isolated regions)
                cluster = Cluster()
                abstraction.cluster = cluster
                cluster.abstractions.add(abstraction)
                clusters.add(cluster)
            else:
                all_entrances.update(abstraction.entrances.values())
    
    processed_entrances: set[GraphNode] = set()
     
    # Flood fill entrances to group connected abstractions into clusters
    while all_entrances:
        start_entrance = all_entrances.pop()
        queue = deque([start_entrance])
        cluster = Cluster()
        clusters.add(cluster)

        while queue:
            entrance: GraphNode = queue.popleft()
            abstraction: Abstraction = entrance.parent

            # Assign the abstraction to this cluster
            abstraction.cluster = cluster

            if entrance in processed_entrances:
                continue

            processed_entrances.add(entrance)
            cluster.abstractions.add(abstraction)

            # Add connected entrances to the flood fill queue
            for connected_entrance in entrance.chunk_connections.values():
                if (
                    connected_entrance not in processed_entrances
                    and connected_entrance in all_entrances
                ):
                    all_entrances.discard(connected_entrance)
                    queue.append(connected_entrance)

    return clusters


def make_mega_chunk_entrances(mega_chunk_pos: Pos, direction: str) -> None:
    """
    Same process as generating normal chunk entrances.
    Uses current chunk entrances , stores them all, and picks middle nodes out of continous lines.

    Args:
        mega_chunk_pos (Pos): (int, int)
        direction (str): "top", "bottom", "left", "right"
    """
    edge_range = range(MAP_CHUNK_SIZE * MEGA_CHUNK_SIZE) # chunk size * how many chunks in each mega chunk

    if direction == "top":
        start_x, start_y = (
            mega_chunk_pos[0] * MAP_CHUNK_SIZE * MEGA_CHUNK_SIZE,
            mega_chunk_pos[1] * MAP_CHUNK_SIZE * MEGA_CHUNK_SIZE,
        )
        neighbor_y = start_y - 1
        candidate_pairs = [
            ((start_x + x, start_y), (start_x + x, neighbor_y)) for x in edge_range
        ]
    elif direction == "bottom":
        start_x, start_y = (
            mega_chunk_pos[0] * MAP_CHUNK_SIZE * MEGA_CHUNK_SIZE,
            (mega_chunk_pos[1] + 1) * MAP_CHUNK_SIZE * MEGA_CHUNK_SIZE - 1,
        )
        neighbor_y = start_y + 1
        candidate_pairs = [
            ((start_x + x, start_y), (start_x + x, neighbor_y)) for x in edge_range
        ]
    elif direction == "left":
        start_x, start_y = (
            mega_chunk_pos[0] * MAP_CHUNK_SIZE * MEGA_CHUNK_SIZE,
            mega_chunk_pos[1] * MAP_CHUNK_SIZE * MEGA_CHUNK_SIZE,
        )
        neighbor_x = start_x - 1
        candidate_pairs = [
            ((start_x, start_y + y), (neighbor_x, start_y + y)) for y in edge_range
        ]
    elif direction == "right":
        start_x, start_y = (
            (mega_chunk_pos[0] + 1) * MAP_CHUNK_SIZE * MEGA_CHUNK_SIZE - 1,
            mega_chunk_pos[1] * MAP_CHUNK_SIZE * MEGA_CHUNK_SIZE,
        )
        neighbor_x = start_x + 1
        candidate_pairs = [
            ((start_x, start_y + y), (neighbor_x, start_y + y)) for y in edge_range
        ]
    else:
        return
    
    # store all of the mega chunks, chunk entrances
    pairs = []
    for pair in candidate_pairs:
        node_pos, neighbor_pos = pair
        if node_pos in zero_positions and neighbor_pos in zero_positions:
            pairs.append(pair)

    groups = []
    current_group = []
    sorted_pairs = sorted(pairs, key=lambda x: (x[0][0], x[0][1]))

    for i in range(len(sorted_pairs)):
        current_pair = sorted_pairs[i]
        if not current_group:
            current_group.append(current_pair)
            continue

        last_pair = current_group[-1]
        if (
            abs(current_pair[0][0] - last_pair[1][0]) <= 1
            and abs(current_pair[0][1] - last_pair[1][1]) <= 1
        ):
            current_group.append(current_pair)
        else:
            groups.append(current_group)
            current_group = [current_pair]

    if current_group:
        groups.append(current_group)

    for group in groups:
        entrance_nodes = []
        for pair in group:
            node_pos, neighbor_pos = pair
            node = get_node(node_pos)
            neighbor = get_node(neighbor_pos)

            if node and neighbor:
                abstract = node.abstraction
                neighbor_abstract = neighbor.abstraction

                if (
                    abstract
                    and neighbor_abstract
                    and node_pos in abstract.entrances
                    and neighbor_pos in neighbor_abstract.entrances
                ):

                    entrance = abstract.entrances[node_pos]
                    neighbor_entrance = neighbor_abstract.entrances[neighbor_pos]
                    entrance_nodes.append((entrance, neighbor_entrance))

        if not entrance_nodes:
            continue

        graph_nodes = [node[0] for node in entrance_nodes]
        neighbor_nodes = [node[1] for node in entrance_nodes]

        avg_x = sum(entrance.x for entrance in graph_nodes) / len(graph_nodes)
        avg_y = sum(entrance.y for entrance in graph_nodes) / len(graph_nodes)
        centroid_entrance = min(
            graph_nodes, key=lambda e: ((e.x - avg_x) ** 2 + (e.y - avg_y) ** 2)
        )

        neighbor_entrance = neighbor_nodes[graph_nodes.index(centroid_entrance)]

        centroid_entrance.mega_chunk_connections[
            (neighbor_entrance.x, neighbor_entrance.y)
        ] = neighbor_entrance
        neighbor_entrance.mega_chunk_connections[
            (centroid_entrance.x, centroid_entrance.y)
        ] = centroid_entrance

        if centroid_entrance.parent.cluster:
            centroid_entrance.parent.cluster.entrances[
                (centroid_entrance.x, centroid_entrance.y)
            ] = centroid_entrance


def generate_mega_entrances(mega_chunk: MegaChunk) -> None:
    for cluster in mega_chunk.clusters:
        cluster.entrances.clear()

    cx, cy = mega_chunk.x, mega_chunk.y
    make_mega_chunk_entrances((cx, cy), "top")
    make_mega_chunk_entrances((cx, cy), "right")
    make_mega_chunk_entrances((cx, cy), "bottom")
    make_mega_chunk_entrances((cx, cy), "left")


def update_mega_chunk_connections(mega_chunk: MegaChunk) -> None:
    all_entrances: list[GraphNode] = []
    for cluster in mega_chunk.clusters:
        all_entrances.extend(cluster.entrances.values())

    for entrance in all_entrances:
        external_connections = {}
        for pos, connected in entrance.mega_chunk_connections.items():
            connected_mega_pos = (
                connected.x // (MAP_CHUNK_SIZE * MEGA_CHUNK_SIZE),
                connected.y // (MAP_CHUNK_SIZE * MEGA_CHUNK_SIZE),
            )
            if connected_mega_pos != (mega_chunk.x, mega_chunk.y):
                external_connections[pos] = connected
        entrance.mega_chunk_connections = external_connections

    for cluster in mega_chunk.clusters:
        cluster_entrances = list(cluster.entrances.values())

        for start_entrance in cluster_entrances:
            visited = set()
            queue = deque([start_entrance])

            while queue:
                current_entrance = queue.popleft()

                if current_entrance in visited:
                    continue

                visited.add(current_entrance)
                
                for other_entrance in cluster_entrances:
                    if other_entrance != current_entrance:
                        current_entrance.mega_chunk_connections[
                            (other_entrance.x, other_entrance.y)
                        ] = other_entrance

                for connected in current_entrance.chunk_connections.values():
                    if connected in cluster_entrances and connected not in visited:
                        queue.append(connected)

def add_node(tile_pos: Pos) -> bool:
    existing_node = get_node(tile_pos)
    if existing_node:
        return False

    zero_positions.add(tile_pos)

    chunk_pos = (tile_pos[0] // MAP_CHUNK_SIZE, tile_pos[1] // MAP_CHUNK_SIZE)
    chunk = chunks.get(chunk_pos)

    if not chunk:
        chunk = Chunk(chunk_pos[0], chunk_pos[1])
        chunks[chunk_pos] = chunk

    mega_chunk_pos = chunk_pos[0] // MEGA_CHUNK_SIZE, chunk_pos[1] // MEGA_CHUNK_SIZE
    mega_chunk = mega_chunks.get(mega_chunk_pos)

    if not mega_chunk:
        mega_chunk = MegaChunk(mega_chunk_pos[0], mega_chunk_pos[1])
        mega_chunks[mega_chunk_pos] = mega_chunk

    if chunk_pos not in mega_chunk.chunks:
        mega_chunk.chunks[chunk_pos] = chunk

    affected_mega_chunks = set()
    affected_mega_chunks.add(mega_chunk)

    new_node = Node(tile_pos[0], tile_pos[1])
    chunk.nodes[tile_pos] = new_node

    flood_fill_chunk(chunk)
    generate_entrances(chunk)
    make_connections(chunk)

    neighbor_chunks_to_update = set()
    for neighbor_pos in get_neighbors(tile_pos[0], tile_pos[1]):
        neighbor_chunk_pos = (
            neighbor_pos[0] // MAP_CHUNK_SIZE,
            neighbor_pos[1] // MAP_CHUNK_SIZE,
        )
        if neighbor_chunk_pos != chunk_pos:
            neighbor_chunks_to_update.add(neighbor_chunk_pos)

    for neighbor_chunk_pos in neighbor_chunks_to_update:
        neighbor_chunk = chunks.get(neighbor_chunk_pos)
        if neighbor_chunk:
            flood_fill_chunk(neighbor_chunk)
            generate_entrances(neighbor_chunk)
            make_connections(neighbor_chunk)

            neighbor_mega_chunk_pos = (
                neighbor_chunk_pos[0] // MEGA_CHUNK_SIZE,
                neighbor_chunk_pos[1] // MEGA_CHUNK_SIZE,
            )
            neighbor_mega_chunk = mega_chunks.get(neighbor_mega_chunk_pos)
            if neighbor_mega_chunk and neighbor_mega_chunk not in affected_mega_chunks:
                affected_mega_chunks.add(neighbor_mega_chunk)

    for affected_mega_chunk in affected_mega_chunks:
        clusters = generate_clusters(affected_mega_chunk)
        affected_mega_chunk.clusters = clusters
        generate_mega_entrances(affected_mega_chunk)
        update_mega_chunk_connections(affected_mega_chunk)

    tile_rect = pygame.Rect(
        (tile_pos[0] % MAP_CHUNK_SIZE) * TILE_SIZE,
        (tile_pos[1] % MAP_CHUNK_SIZE) * TILE_SIZE,
        TILE_SIZE,
        TILE_SIZE,
    )
    pygame.draw.rect(chunk.surface, NODE_COLOR, tile_rect, 0)

    world_tile_rect = pygame.Rect(
        tile_pos[0] * TILE_SIZE, tile_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE
    )
    pygame.draw.rect(cluster_color_surface, NODE_COLOR, world_tile_rect, 0)

    existing_chunks = [chunk_pos] + [
        pos for pos in neighbor_chunks_to_update if chunks.get(pos)
    ]
    redraw_affected_areas(existing_chunks, affected_mega_chunks)

    for affected_chunk_pos in existing_chunks:
        affected_chunk = chunks.get(affected_chunk_pos)
        if affected_chunk:
            affected_chunk.update_surface()
            abstraction_surface.blit(
                affected_chunk.surface,
                (
                    affected_chunk.x * CHUNK_PIXEL_SIZE,
                    affected_chunk.y * CHUNK_PIXEL_SIZE,
                ),
            )

    return True


# === NODE REMOVAL ===
def remove_node(tile_pos: Pos) -> bool:
    node = get_node(tile_pos)
    if not node:
        return False

    zero_positions.discard(tile_pos)

    chunk_pos = (node.x // MAP_CHUNK_SIZE, node.y // MAP_CHUNK_SIZE)
    chunk = chunks.get(chunk_pos)

    if not chunk:
        print(f"Warning: Node exists but chunk doesn't at {chunk_pos}")
        return False

    mega_chunk_pos = chunk_pos[0] // MEGA_CHUNK_SIZE, chunk_pos[1] // MEGA_CHUNK_SIZE
    mega_chunk = mega_chunks.get(mega_chunk_pos)

    if not mega_chunk:
        print(f"Warning: Chunk exists but mega chunk doesn't at {mega_chunk_pos}")
        return False

    affected_mega_chunks = set()
    affected_mega_chunks.add(mega_chunk)

    removed_node = chunk.nodes.pop(tile_pos, None)
    if not removed_node:
        return False

    flood_fill_chunk(chunk)
    generate_entrances(chunk)
    make_connections(chunk)

    neighbor_chunks_to_update = set()
    for neighbor_pos in get_neighbors(node.x, node.y):
        neighbor_chunk_pos = (
            neighbor_pos[0] // MAP_CHUNK_SIZE,
            neighbor_pos[1] // MAP_CHUNK_SIZE,
        )
        if neighbor_chunk_pos != chunk_pos:
            neighbor_chunks_to_update.add(neighbor_chunk_pos)

    for neighbor_chunk_pos in neighbor_chunks_to_update:
        neighbor_chunk = chunks.get(neighbor_chunk_pos)
        if neighbor_chunk:
            flood_fill_chunk(neighbor_chunk)
            generate_entrances(neighbor_chunk)
            make_connections(neighbor_chunk)

            neighbor_mega_chunk_pos = (
                neighbor_chunk_pos[0] // MEGA_CHUNK_SIZE,
                neighbor_chunk_pos[1] // MEGA_CHUNK_SIZE,
            )
            neighbor_mega_chunk = mega_chunks.get(neighbor_mega_chunk_pos)
            if neighbor_mega_chunk and neighbor_mega_chunk not in affected_mega_chunks:
                affected_mega_chunks.add(neighbor_mega_chunk)

    for affected_mega_chunk in affected_mega_chunks:
        clusters = generate_clusters(affected_mega_chunk)
        affected_mega_chunk.clusters = clusters
        generate_mega_entrances(affected_mega_chunk)
        update_mega_chunk_connections(affected_mega_chunk)

    tile_rect = pygame.Rect(
        (tile_pos[0] % MAP_CHUNK_SIZE) * TILE_SIZE,
        (tile_pos[1] % MAP_CHUNK_SIZE) * TILE_SIZE,
        TILE_SIZE,
        TILE_SIZE,
    )
    chunk.surface.fill(BACKGROUND_COLOR, tile_rect)

    world_tile_rect = pygame.Rect(
        tile_pos[0] * TILE_SIZE, tile_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE
    )
    abstraction_surface.fill(BACKGROUND_COLOR, world_tile_rect)
    cluster_color_surface.fill(BACKGROUND_COLOR, world_tile_rect)

    existing_chunks = [chunk_pos] + [
        pos for pos in neighbor_chunks_to_update if chunks.get(pos)
    ]
    redraw_affected_areas(existing_chunks, affected_mega_chunks)

    for affected_chunk_pos in existing_chunks:
        affected_chunk = chunks.get(affected_chunk_pos)
        if affected_chunk:
            affected_chunk.update_surface()
            abstraction_surface.blit(
                affected_chunk.surface,
                (
                    affected_chunk.x * CHUNK_PIXEL_SIZE,
                    affected_chunk.y * CHUNK_PIXEL_SIZE,
                ),
            )
    return True


def redraw_affected_areas(affected_chunk_positions: list, affected_mega_chunks: set):
    for chunk_pos in affected_chunk_positions:
        chunk = chunks.get(chunk_pos)
        if not chunk:
            continue

        chunk_rect = pygame.Rect(
            chunk.x * CHUNK_PIXEL_SIZE,
            chunk.y * CHUNK_PIXEL_SIZE,
            CHUNK_PIXEL_SIZE,
            CHUNK_PIXEL_SIZE,
        )
        chunk_entrance_surface.fill(BACKGROUND_COLOR, chunk_rect)
        connection_surface.fill(BACKGROUND_COLOR, chunk_rect)

        for abstraction in chunk.abstractions:
            for entrance in abstraction.entrances.values():
                rect = pygame.Rect(
                    entrance.x * TILE_SIZE, entrance.y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                )
                pygame.draw.rect(chunk_entrance_surface, ENTRANCE_COLOR, rect, 1)

                for connected_entrance in entrance.chunk_connections.values():
                    e_center = (
                        entrance.x * TILE_SIZE + TILE_SIZE // 2,
                        entrance.y * TILE_SIZE + TILE_SIZE // 2,
                    )
                    c_center = (
                        connected_entrance.x * TILE_SIZE + TILE_SIZE // 2,
                        connected_entrance.y * TILE_SIZE + TILE_SIZE // 2,
                    )
                    pygame.draw.line(
                        connection_surface,
                        CONNECTION_COLOR,
                        e_center,
                        c_center,
                        PATH_LINE_WIDTH,
                    )

    for mega_chunk in affected_mega_chunks:
        mega_chunk_rect = pygame.Rect(
            mega_chunk.x * MEGA_CHUNK_PIXEL_SIZE,
            mega_chunk.y * MEGA_CHUNK_PIXEL_SIZE,
            MEGA_CHUNK_PIXEL_SIZE,
            MEGA_CHUNK_PIXEL_SIZE,
        )
        mega_chunk_entrance_surface.fill(BACKGROUND_COLOR, mega_chunk_rect)
        mega_connection_surface.fill(BACKGROUND_COLOR, mega_chunk_rect)

        drawn_connections = set()
        for cluster in mega_chunk.clusters:
            for entrance in cluster.entrances.values():
                rect = pygame.Rect(
                    entrance.x * TILE_SIZE, entrance.y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                )
                pygame.draw.rect(mega_chunk_entrance_surface, ENTRANCE_COLOR, rect, 1)

                e_center = (
                    entrance.x * TILE_SIZE + TILE_SIZE // 2,
                    entrance.y * TILE_SIZE + TILE_SIZE // 2,
                )

                for other_entrance in entrance.mega_chunk_connections.values():
                    connection_key = tuple(
                        sorted(
                            [
                                (entrance.x, entrance.y),
                                (other_entrance.x, other_entrance.y),
                            ]
                        )
                    )

                    if connection_key not in drawn_connections:
                        drawn_connections.add(connection_key)
                        c_center = (
                            other_entrance.x * TILE_SIZE + TILE_SIZE // 2,
                            other_entrance.y * TILE_SIZE + TILE_SIZE // 2,
                        )
                        pygame.draw.line(
                            mega_connection_surface,
                            CONNECTION_COLOR,
                            e_center,
                            c_center,
                            PATH_LINE_WIDTH,
                        )

                pygame.draw.circle(
                    mega_connection_surface,
                    ENTRANCE_INDICATOR_COLOR,
                    e_center,
                    PATH_CIRCLE_RADIUS,
                )

def draw_chunk_grid():
    for chunk_x in range(0, MAP_TILE_WIDTH // MAP_CHUNK_SIZE + 1):
        x = chunk_x * CHUNK_PIXEL_SIZE
        pygame.draw.line(
            chunk_line_surface, GRID_COLOR, (x, 0), (x, SURFACE_HEIGHT), PATH_LINE_WIDTH
        )
        x *= MEGA_CHUNK_SIZE
        pygame.draw.line(
            mega_chunk_line_surface,
            GRID_COLOR,
            (x, 0),
            (x, SURFACE_HEIGHT),
            PATH_LINE_WIDTH,
        )

    for chunk_y in range(0, MAP_TILE_HEIGHT // MAP_CHUNK_SIZE + 1):
        y = chunk_y * CHUNK_PIXEL_SIZE
        pygame.draw.line(
            chunk_line_surface, GRID_COLOR, (0, y), (SURFACE_WIDTH, y), PATH_LINE_WIDTH
        )
        y *= MEGA_CHUNK_SIZE
        pygame.draw.line(
            mega_chunk_line_surface,
            GRID_COLOR,
            (0, y),
            (SURFACE_WIDTH, y),
            PATH_LINE_WIDTH,
        )


def draw_chunks():
    for chunk in chunks.values():
        chunk.update_surface()

def draw_abstractions():
    for chunk in chunks.values():
        abstraction_surface.blit(
            chunk.surface, (chunk.x * CHUNK_PIXEL_SIZE, chunk.y * CHUNK_PIXEL_SIZE)
        )

def draw_chunk_entrances():
    chunk_entrance_surface.fill(BACKGROUND_COLOR)
    for chunk in chunks.values():
        for abstract in chunk.abstractions:
            for entrance in abstract.entrances.values():
                rect = pygame.Rect(
                    entrance.x * TILE_SIZE, entrance.y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                )
                pygame.draw.rect(chunk_entrance_surface, ENTRANCE_COLOR, rect, 1)


def draw_chunk_connections():
    connection_surface.fill(BACKGROUND_COLOR)
    for chunk in chunks.values():
        for abstraction in chunk.abstractions:
            for entrance in abstraction.entrances.values():
                for connected_entrance in entrance.chunk_connections.values():
                    e_center = (
                        entrance.x * TILE_SIZE + TILE_SIZE // 2,
                        entrance.y * TILE_SIZE + TILE_SIZE // 2,
                    )
                    c_center = (
                        connected_entrance.x * TILE_SIZE + TILE_SIZE // 2,
                        connected_entrance.y * TILE_SIZE + TILE_SIZE // 2,
                    )
                    pygame.draw.line(
                        connection_surface,
                        CONNECTION_COLOR,
                        e_center,
                        c_center,
                        PATH_LINE_WIDTH,
                    )


def draw_mega_entrances():
    mega_chunk_entrance_surface.fill(BACKGROUND_COLOR)
    for mega_chunk in mega_chunks.values():
        for cluster in mega_chunk.clusters:
            for entrance in cluster.entrances.values():
                rect = pygame.Rect(
                    entrance.x * TILE_SIZE, entrance.y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                )
                pygame.draw.rect(mega_chunk_entrance_surface, ENTRANCE_COLOR, rect, 1)

def draw_mega_connections():
    mega_connection_surface.fill(BACKGROUND_COLOR)
    drawn_connections = set()

    for mega_chunk in mega_chunks.values():
        for cluster in mega_chunk.clusters:
            for entrance in cluster.entrances.values():
                e_center = (
                    entrance.x * TILE_SIZE + TILE_SIZE // 2,
                    entrance.y * TILE_SIZE + TILE_SIZE // 2,
                )
                for other_entrance in entrance.mega_chunk_connections.values():
                    connection_key = tuple(
                        sorted(
                            [
                                (entrance.x, entrance.y),
                                (other_entrance.x, other_entrance.y),
                            ]
                        )
                    )

                    if connection_key not in drawn_connections:
                        drawn_connections.add(connection_key)
                        c_center = (
                            other_entrance.x * TILE_SIZE + TILE_SIZE // 2,
                            other_entrance.y * TILE_SIZE + TILE_SIZE // 2,
                        )
                        pygame.draw.line(
                            mega_connection_surface,
                            CONNECTION_COLOR,
                            e_center,
                            c_center,
                            PATH_LINE_WIDTH,
                        )
                pygame.draw.circle(
                    mega_connection_surface,
                    ENTRANCE_INDICATOR_COLOR,
                    e_center,
                    PATH_CIRCLE_RADIUS,
                )


def draw_path(
    path: list[Node] | list[GraphNode], color: Color = END_NODE_COLOR, rect_size=1, lines=False
) -> None:
    """
    Draws a path on the screen using rectangles and optional connecting lines.
    
    This function visualizes a pathfinding result by drawing rectangles at each
    node position and optionally connecting them with lines. The path can be
    drawn at different levels of the hierarchy (level 0, 1, or 2).
    
    Args:
        path: List of Node or GraphNode objects representing the path
        color: Color to use for drawing the path (default: END_NODE_COLOR)
        rect_size: Thickness of the rectangle border (default: 1)
        lines: Whether to draw connecting lines between nodes (default: False)
    """
    if not path:
        return

    prev_pos = None
    for node in path:
        # Draw a rectangle at each node position
        rect = pygame.Rect(
            node.x * TILE_SIZE + pan_offset[0],
            node.y * TILE_SIZE + pan_offset[1],
            TILE_SIZE,
            TILE_SIZE,
        )
        pygame.draw.rect(screen, color, rect, rect_size)

        # Draw connecting lines between consecutive nodes if requested
        if lines and prev_pos:
            start_pos = (
                prev_pos[0] * TILE_SIZE + TILE_SIZE // 2 + pan_offset[0],
                prev_pos[1] * TILE_SIZE + TILE_SIZE // 2 + pan_offset[1],
            )
            end_pos = (
                node.x * TILE_SIZE + TILE_SIZE // 2 + pan_offset[0],
                node.y * TILE_SIZE + TILE_SIZE // 2 + pan_offset[1],
            )
            pygame.draw.line(screen, color, start_pos, end_pos, 2)

        prev_pos = (node.x, node.y)


def redraw_all_surfaces():
    """
    Clears and redraws all visualization surfaces.
    
    This function is called when the map structure changes (e.g., when nodes
    are added or removed). It clears all the rendering surfaces and redraws
    the entrance points and connections at both chunk and mega-chunk levels.
    """
    # Clear all surfaces with background color
    chunk_entrance_surface.fill(BACKGROUND_COLOR)
    connection_surface.fill(BACKGROUND_COLOR)
    mega_chunk_entrance_surface.fill(BACKGROUND_COLOR)
    mega_connection_surface.fill(BACKGROUND_COLOR)

    # Redraw all visualization elements
    draw_chunk_entrances()
    draw_chunk_connections()
    draw_mega_entrances()
    draw_mega_connections()


def get_tile_from_mouse(pos: tuple[int, int]) -> Pos:
    """
    Converts mouse screen coordinates to tile coordinates.
    
    This function takes the mouse position in screen coordinates and converts
    it to the corresponding tile coordinates, taking into account the current
    camera pan offset.
    
    Args:
        pos: (x, y) tuple of mouse screen coordinates
        
    Returns:
        Pos: (x, y) tuple of tile coordinates
    """
    x, y = pos
    tile_x = (x - pan_offset[0]) // TILE_SIZE
    tile_y = (y - pan_offset[1]) // TILE_SIZE
    return tile_x, tile_y


# Step 1: Generate the hierarchical tile structure
t = time.time()
generate_tiles(zero_positions)
print(f"- Tiles generated. Took {time.time() - t:.2f} seconds")

# Draw the chunk grid for visualization
draw_chunk_grid()

# Step 2: Perform flood fill to identify connected regions within each chunk
t = time.time()
for chunk in chunks.values():
    flood_fill_chunk(chunk)
print(f"- Chunks flood filled. Took {time.time() - t:.2f} seconds")

# Step 3: Generate entrance points between chunks
t = time.time()
for chunk in chunks.values():
    generate_entrances(chunk)
print(f"- Chunk entrances generated. Took {time.time() - t:.2f} seconds")

# Step 4: Create connections between entrance nodes within each abstraction
t = time.time()
for chunk in chunks.values():
    make_connections(chunk)
print(f"- Connections generated. Took {time.time() - t:.2f} seconds")

# Step 5: Generate clusters by grouping connected abstractions within mega-chunks
t = time.time()
for mega_chunk in mega_chunks.values():
    clusters = generate_clusters(mega_chunk)
    mega_chunk.clusters = clusters
print(f"- Clusters generated. Took {time.time() - t:.2f} seconds")

# Step 6: Generate mega-chunk level entrances and connections
t = time.time()
for mega_chunk in mega_chunks.values():
    generate_mega_entrances(mega_chunk)
print(f"- Mega Chunk entrances generated. Took {time.time() - t:.2f} seconds")

# Update mega-chunk connections
for mega_chunk in mega_chunks.values():
    update_mega_chunk_connections(mega_chunk)

# Draw all nodes on the cluster color surface for level 2 visualization
for mega_chunk in mega_chunks.values():
    for chunk in mega_chunk.chunks.values():
        for node in chunk.nodes.values():
            rect = pygame.Rect(
                node.x * TILE_SIZE, node.y * TILE_SIZE, TILE_SIZE, TILE_SIZE
            )
            pygame.draw.rect(cluster_color_surface, NODE_COLOR, rect, 0)

# Step 7: Render all visualization layers
t = time.time()
draw_chunks()
draw_abstractions()
draw_chunk_entrances()
draw_chunk_connections()
draw_mega_entrances()
draw_mega_connections()
print(f"- Drawing completed. Took {time.time() - t:.2f} seconds")
print(f"- Total time. Took {time.time() - total_time:.2f} seconds")


# Initialize game state variables
layer = 1  # Current visualization layer (1 = chunk level, 2 = mega-chunk level)
selected_surface = abstraction_surface
selected_line_surface = chunk_line_surface
selected_entrance_surface = chunk_entrance_surface
selected_connection_surface = connection_surface

# Pathfinding state
start_node = None
end_node = None
pan_offset = [0, 0]  # Camera pan offset

# Debug visualization flags
debug_show_grid = True
debug_show_entrances = True
debug_show_connections = True
debug_show_paths = True

# Mouse interaction state
is_drag_deleting = False
last_deleted_pos = None
is_drag_adding = False
last_added_pos = None

# Create the pathfinding entity
entity = Entity(0, 0)

# Initialize pygame clock and game loop flag
clock = pygame.time.Clock()
running = True

# Main game loop
while running:
    dt = clock.tick(FPS) / 1000.0  # Delta time in seconds

    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            # Switch between visualization layers
            if event.key == pygame.K_1:
                # Switch to chunk level (level 1) visualization
                layer = 1
                selected_surface = abstraction_surface
                selected_line_surface = chunk_line_surface
                selected_entrance_surface = chunk_entrance_surface
                selected_connection_surface = connection_surface
            elif event.key == pygame.K_2:
                # Switch to mega-chunk level (level 2) visualization
                layer = 2
                selected_surface = cluster_color_surface
                selected_line_surface = mega_chunk_line_surface
                selected_entrance_surface = mega_chunk_entrance_surface
                selected_connection_surface = mega_connection_surface

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            tile_pos = get_tile_from_mouse(mouse_pos)

            if event.button == 1:  # Left mouse button
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                    # Shift + left click: Start adding nodes (drag mode)
                    is_drag_adding = True
                    last_added_pos = tile_pos
                    add_node(tile_pos)
                else:
                    # Left click: Set start node for pathfinding
                    node = get_node(tile_pos)
                    if node:
                        start_node = node
                        entity.x, entity.y = (
                            start_node.x * TILE_SIZE,
                            start_node.y * TILE_SIZE,
                        )

                        # If both start and end nodes are set, perform pathfinding
                        if start_node and end_node and start_node != end_node:
                            entity.pathfind(end_node)

            elif event.button == 3:  # Right mouse button
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                    # Shift + right click: Start removing nodes (drag mode)
                    is_drag_deleting = True
                    last_deleted_pos = tile_pos
                    remove_node(tile_pos)
                    
                else:
                    # Right click: Set end node for pathfinding
                    node = get_node(tile_pos)
                    if node:
                        end_node = node
                        # Clear previous paths
                        entity.level_0_path.clear()
                        entity.level_1_path.clear()
                        entity.level_2_path.clear()

                        # If both start and end nodes are set, perform pathfinding
                        if start_node and end_node and start_node != end_node:
                            entity.x, entity.y = (
                                start_node.x * TILE_SIZE,
                                start_node.y * TILE_SIZE,
                            )
                            entity.pathfind(end_node)

        elif event.type == pygame.MOUSEBUTTONUP:
            # End drag operations when mouse button is released
            if event.button == 1 or event.button == 3:
                is_drag_deleting = False
                last_deleted_pos = None
                is_drag_adding = False
                last_added_pos = None

        elif event.type == pygame.MOUSEMOTION:
            # Handle drag operations for adding/removing nodes
            if is_drag_deleting:
                # Drag to remove nodes
                mouse_pos = pygame.mouse.get_pos()
                tile_pos = get_tile_from_mouse(mouse_pos)

                if tile_pos != last_deleted_pos and last_deleted_pos is not None:
                    # Fill line between last position and current position with remove_node action
                    fill_line_with_action(last_deleted_pos, tile_pos, remove_node)
                elif last_deleted_pos is None:
                    remove_node(tile_pos)

                last_deleted_pos = tile_pos

            elif is_drag_adding:
                # Drag to add nodes
                mouse_pos = pygame.mouse.get_pos()
                tile_pos = get_tile_from_mouse(mouse_pos)

                if tile_pos != last_added_pos and last_added_pos is not None:
                    # Fill line between last position and current position with add_node action
                    fill_line_with_action(last_added_pos, tile_pos, add_node)
                elif last_added_pos is None:
                    add_node(tile_pos)

                last_added_pos = tile_pos

    # Handle continuous key presses for camera movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:  # Move camera up
        pan_offset[1] += PAN_SPEED
    if keys[pygame.K_s]:  # Move camera down
        pan_offset[1] -= PAN_SPEED
    if keys[pygame.K_a]:  # Move camera left
        pan_offset[0] += PAN_SPEED
    if keys[pygame.K_d]:  # Move camera right
        pan_offset[0] -= PAN_SPEED

    # Update entity movement and pathfinding
    entity.update(dt)

    # Clear screen and draw the main surface
    screen.fill(BACKGROUND_COLOR)
    screen.blit(selected_surface, pan_offset)

    # Draw debug visualization layers based on flags
    if debug_show_grid:
        screen.blit(selected_line_surface, pan_offset)
    if debug_show_entrances:
        screen.blit(selected_entrance_surface, pan_offset)
    if debug_show_connections:
        screen.blit(selected_connection_surface, pan_offset)

    # Draw mouse cursor highlight and tile coordinates
    mouse_pos = pygame.mouse.get_pos()
    tile_pos = get_tile_from_mouse(mouse_pos)
    rect = pygame.Rect(
        tile_pos[0] * TILE_SIZE + pan_offset[0],
        tile_pos[1] * TILE_SIZE + pan_offset[1],
        TILE_SIZE,
        TILE_SIZE,
    )
    pygame.draw.rect(screen, ENTRANCE_COLOR, rect, 1)

    # Display current tile coordinates
    font = pygame.font.Font(None, FONT_SIZE)
    tile_text = font.render(f"Tile: {tile_pos[0]}, {tile_pos[1]}", True, ENTRANCE_COLOR)
    screen.blit(
        tile_text, (mouse_pos[0] + UI_TEXT_OFFSET, mouse_pos[1] + UI_TEXT_OFFSET)
    )

    # Draw start and end nodes for pathfinding
    if start_node:
        rect = pygame.Rect(
            start_node.x * TILE_SIZE + pan_offset[0],
            start_node.y * TILE_SIZE + pan_offset[1],
            TILE_SIZE,
            TILE_SIZE,
        )
        pygame.draw.rect(screen, START_NODE_COLOR, rect, 0)

    if end_node:
        rect = pygame.Rect(
            end_node.x * TILE_SIZE + pan_offset[0],
            end_node.y * TILE_SIZE + pan_offset[1],
            TILE_SIZE,
            TILE_SIZE,
        )
        pygame.draw.rect(screen, END_NODE_COLOR, rect, 0)

    # Draw pathfinding paths at different levels
    if debug_show_paths:
        if entity.level_0_path:
            draw_path(entity.level_0_path, color=PATH_LEVEL_0_COLOR)
        if entity.level_1_path:
            draw_path(entity.level_1_path, color=PATH_LEVEL_1_COLOR, lines=True)
        if entity.level_2_path:
            draw_path(entity.level_2_path, color=PATH_LEVEL_2_COLOR, lines=True)

    # Draw the moving entity
    rect = pygame.Rect(
        entity.x + pan_offset[0] , entity.y + pan_offset[1], TILE_SIZE, TILE_SIZE
    )
    pygame.draw.rect(screen, ENTITY_COLOR, rect, 0)

    # Draw the entity's current target node
    if entity.target_node:
        rect = pygame.Rect(
            entity.target_node.x * TILE_SIZE + pan_offset[0],
            entity.target_node.y * TILE_SIZE + pan_offset[1],
            TILE_SIZE,
            TILE_SIZE,
        )
        pygame.draw.rect(screen, TARGET_NODE_COLOR, rect, 1)

    # Update display with FPS counter
    pygame.display.set_caption(f"FPS: {clock.get_fps():.2f}")
    pygame.display.update()

pygame.quit()
