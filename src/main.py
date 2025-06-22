from settings import (
    WIDTH,
    HEIGHT,
    SURFACE_WIDTH,
    SURFACE_HEIGHT,
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    MAP_CHUNK_SIZE,
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    TILE_SIZE,
    CHUNK_PIXEL_SIZE,
    ENTRANCE_SPACING,
)
from utility import (
    index_1d,
    get_neighbors,
    generate_perlin_noise_map,
    get_node,
    NodeAStar,
    GraphAStar,
)
from data import (
    Node,
    GraphNode,
    Abstraction,
    Cluster,
    Chunk,
    MegaChunk,
    GigaChunk,
    Pos,
    Color,
    chunks,
    mega_chunks,
    giga_chunks,
)
from entity import Entity
import pygame
from typing import Tuple, Set, List
import time
from collections import deque
import pygame_gui
import ctypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


pygame.font.init()

total = time.time()

user32 = ctypes.windll.user32
screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
WIDTH, HEIGHT = screen_width // 1.2, screen_height // 1.2

screen = pygame.display.set_mode((WIDTH, HEIGHT))
manager = pygame_gui.UIManager((WIDTH, HEIGHT), theme_path=r"theme.json")

OFFSET = 1
OFFSET_WIDTH = SURFACE_WIDTH + OFFSET
OFFSET_HEIGHT = SURFACE_HEIGHT + OFFSET

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

map_data = generate_perlin_noise_map(
    MAP_TILE_WIDTH, MAP_TILE_HEIGHT, noise_scale=200, threshold=0.05, octaves=5
)
map_data_1d = [tile for row in map_data for tile in row]
zero_positions = {
    (x, y)
    for y in range(MAP_TILE_HEIGHT)
    for x in range(MAP_TILE_WIDTH)
    if map_data_1d[index_1d(x, y, MAP_TILE_WIDTH)] == 0
}

t = time.time()


def generate_tiles(positions: Set[Pos]):
    for x, y in positions:

        chunk_pos = x // MAP_CHUNK_SIZE, y // MAP_CHUNK_SIZE

        chunk = chunks.get(chunk_pos)
        if not chunk:
            chunk = Chunk(chunk_pos[0], chunk_pos[1])
            chunks[chunk_pos] = chunk
        chunk.nodes[(x, y)] = Node(x, y)

        mega_chunk_pos = chunk_pos[0] // 3, chunk_pos[1] // 3
        mega_chunk = mega_chunks.get(mega_chunk_pos)
        if not mega_chunk:
            mega_chunk = MegaChunk(mega_chunk_pos[0], mega_chunk_pos[1])
            mega_chunks[mega_chunk_pos] = mega_chunk
        if chunk_pos not in mega_chunk.chunks:
            mega_chunk.chunks[chunk_pos] = chunk

        giga_chunk_pos = mega_chunk_pos[0] // 3, mega_chunk_pos[1] // 3
        giga_chunk = giga_chunks.get(giga_chunk_pos)
        if not giga_chunk:
            giga_chunk = GigaChunk(giga_chunk_pos[0], giga_chunk_pos[1])
            giga_chunks[giga_chunk_pos] = giga_chunk
        if mega_chunk_pos not in giga_chunk.mega_chunks:
            giga_chunk.mega_chunks[mega_chunk_pos] = mega_chunk


generate_tiles(zero_positions)
dt = time.time() - t
print(f"- Tiles generated. Took {dt} seconds")


def draw_chunk_grid():
    # Draw vertical lines
    for chunk_x in range(0, MAP_TILE_WIDTH // MAP_CHUNK_SIZE + 1):
        x = chunk_x * CHUNK_PIXEL_SIZE
        pygame.draw.line(
            chunk_line_surface, (66, 66, 66), (x, 0), (x, SURFACE_HEIGHT), 1
        )
        x *= 3
        pygame.draw.line(
            mega_chunk_line_surface, (66, 66, 66), (x, 0), (x, SURFACE_HEIGHT), 1
        )

    # Draw horizontal lines
    for chunk_y in range(0, MAP_TILE_HEIGHT // MAP_CHUNK_SIZE + 1):
        y = chunk_y * CHUNK_PIXEL_SIZE
        pygame.draw.line(
            chunk_line_surface, (66, 66, 66), (0, y), (SURFACE_WIDTH, y), 1
        )
        y *= 3
        pygame.draw.line(
            mega_chunk_line_surface, (66, 66, 66), (0, y), (SURFACE_WIDTH, y), 1
        )


draw_chunk_grid()

t = time.time()

class ThreadedFloodFill:
    def __init__(self, chunks, get_neighbors_func):
        self.chunks = chunks
        self.get_neighbors = get_neighbors_func

    def flood_fill_chunk_thread(self, chunk):
        """Thread-safe flood fill for a single chunk"""

        chunk_nodes = list(chunk.nodes.keys())
        visited = set()

        while chunk_nodes:
            start_pos = chunk_nodes.pop()

            if start_pos in visited:
                continue

            queue = deque([start_pos])
            abstraction = Abstraction()
            chunk.abstractions.add(abstraction)

            while queue:
                node_pos = queue.popleft()

                if node_pos in visited:
                    continue

                visited.add(node_pos)
                node = chunk.nodes[node_pos]
                node.abstraction = abstraction

                for neighbor_pos in self.get_neighbors(node.x, node.y):
                    if neighbor_pos in chunk.nodes and neighbor_pos not in visited:
                        queue.append(neighbor_pos)
                        neighbor_node = chunk.nodes.get(neighbor_pos)
                        if neighbor_node:
                            # Bidirectional connections
                            node.connections[neighbor_pos] = neighbor_node
                            neighbor_node.connections[node_pos] = node

    def parallel_flood_fill(self, max_workers=None):
        """Perform flood fill on all chunks using threading"""
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 8)

        chunk_list = list(self.chunks.values())
        total_chunks = len(chunk_list)

        print(f"Processing {total_chunks} chunks with {max_workers} threads...")

        start_time = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self.flood_fill_chunk_thread, chunk): chunk
                for chunk in chunk_list
            }
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    future.result()
                    completed += 1

                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        remaining = (total_chunks - completed) / rate
                        print(
                            f"Progress: {completed}/{total_chunks} chunks "
                            f"({completed/total_chunks*100:.1f}%) - "
                            f"ETA: {remaining:.1f}s"
                        )

                except Exception as e:
                    print(f"Error processing chunk at {chunk.x}, {chunk.y}: {e}")

        elapsed = time.time() - start_time
        print(
            f"Completed flood fill for {total_chunks} chunks in {elapsed:.2f} seconds"
        )


threaded_fill = ThreadedFloodFill(chunks, get_neighbors)
threaded_fill.parallel_flood_fill()

dt = time.time() - t
print(f"- Chunks flood filled. Took {dt} seconds")


t = time.time()


def make_chunk_entrances(chunk_pos: Pos, direction: str) -> None:

    chunk = chunks[chunk_pos]

    edge_range = range(MAP_CHUNK_SIZE)

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

    neighbor_chunk = chunks.get(neighbor_pos)
    if not neighbor_chunk:
        return

    groups: List[List[Tuple[Pos, Pos]]] = []
    current_group: List[Tuple[Pos, Pos]] = []

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

    for group in groups:
        entrances: List[Tuple[Pos, Pos]] = []
        for i in range(0, len(group), ENTRANCE_SPACING):
            group_slice = group[i : i + ENTRANCE_SPACING]
            middle_index = len(group_slice) // 2
            entrances.append(group_slice[middle_index])

        for chunk_node_pos, neighbor_node_pos in entrances:
            chunk_node = chunk.nodes.get(chunk_node_pos)
            neighbor_node = neighbor_chunk.nodes.get(neighbor_node_pos)

            if chunk_node is None or neighbor_node is None:
                continue

            chunk_abstract = chunk_node.abstraction
            neighbor_abstract = neighbor_node.abstraction

            assert chunk_abstract is not None
            assert neighbor_abstract is not None

            # Check if a GraphNode already exists at the chunk_node_pos
            chunk_graph_node = chunk_abstract.entrances.get(chunk_node_pos)
            if not chunk_graph_node:
                chunk_graph_node = GraphNode(chunk_node_pos, chunk_abstract)
                chunk_abstract.entrances[(chunk_graph_node.x, chunk_graph_node.y)] = (
                    chunk_graph_node
                )

            # Check if a GraphNode already exists at the neighbor_node_pos
            neighbor_graph_node = neighbor_abstract.entrances.get(neighbor_node_pos)
            if not neighbor_graph_node:
                neighbor_graph_node = GraphNode(neighbor_node_pos, neighbor_abstract)
                neighbor_abstract.entrances[
                    (neighbor_graph_node.x, neighbor_graph_node.y)
                ] = neighbor_graph_node

            # Add connections between the graph nodes
            chunk_graph_node.chunk_connections[
                (neighbor_graph_node.x, neighbor_graph_node.y)
            ] = neighbor_graph_node
            neighbor_graph_node.chunk_connections[
                (chunk_graph_node.x, chunk_graph_node.y)
            ] = chunk_graph_node


def generate_entrances() -> None:
    for cx, cy in chunks.keys():
        make_chunk_entrances((cx, cy), "top")
        make_chunk_entrances((cx, cy), "right")
        make_chunk_entrances((cx, cy), "bottom")
        make_chunk_entrances((cx, cy), "left")


generate_entrances()

dt = time.time() - t
print(f"- Chunk entrances generated. Took {dt} seconds")

t = time.time()


def make_connections(chunk: Chunk) -> None:
    for abstraction in chunk.abstractions:
        for entrance in abstraction.entrances.values():
            e_x, e_y = entrance.x, entrance.y

            for other_entrance in abstraction.entrances.values():
                o_x, o_y = other_entrance.x, other_entrance.y
                if entrance == other_entrance:
                    continue
                other_entrance.chunk_connections[(e_x, e_y)] = entrance
                entrance.chunk_connections[(o_x, o_y)] = other_entrance


for chunk in chunks.values():
    make_connections(chunk)
dt = time.time() - t
print(f"- Connections generated. Took {dt} seconds")


t = time.time()


def generate_clusters(mega_chunk: MegaChunk) -> Set[Cluster]:
    all_entrances: Set[GraphNode] = set()
    clusters: set[Cluster] = set()

    for chunk in mega_chunk.chunks.values():
        for abstraction in chunk.abstractions:
            if len(abstraction.entrances) == 0:
                cluster = Cluster()
                abstraction.cluster = cluster
                cluster.abstractions.add(abstraction)
                clusters.add(cluster)
            else:
                all_entrances.update(abstraction.entrances.values())

    processed_entrances: set[GraphNode] = set()

    while all_entrances:
        start_entrance = all_entrances.pop()
        queue = deque([start_entrance])
        cluster = Cluster()
        clusters.add(cluster)

        while queue:
            entrance: GraphNode = queue.popleft()
            abstraction: Abstraction = entrance.parent

            abstraction.cluster = cluster

            if entrance in processed_entrances:
                continue

            processed_entrances.add(entrance)

            cluster.abstractions.add(abstraction)
            for connected_entrance in entrance.chunk_connections.values():
                if (
                    connected_entrance not in processed_entrances
                    and connected_entrance in all_entrances
                ):
                    all_entrances.discard(connected_entrance)
                    queue.append(connected_entrance)
    return clusters


for mega_chunk in mega_chunks.values():
    clusters: Set[Cluster] = generate_clusters(mega_chunk)
    mega_chunk.clusters = clusters
dt = time.time() - t
print(f"- Clusters generated. Took {dt} seconds")


def make_mega_chunk_entrances(mega_chunk_pos: Pos, direction: str) -> None:
    edge_range = range(MAP_CHUNK_SIZE * 3)

    # generate possible chunk candidates
    if direction == "top":
        start_x, start_y = (
            mega_chunk_pos[0] * 3 * MAP_CHUNK_SIZE,
            mega_chunk_pos[1] * 3 * MAP_CHUNK_SIZE,
        )
        neighbor_y = start_y - 1
        candidate_pairs = [
            ((start_x + x, start_y), (start_x + x, neighbor_y)) for x in edge_range
        ]
    elif direction == "bottom":
        start_x, start_y = (
            mega_chunk_pos[0] * 3 * MAP_CHUNK_SIZE,
            (mega_chunk_pos[1] + 1) * 3 * MAP_CHUNK_SIZE - 1,
        )
        neighbor_y = start_y + 1
        candidate_pairs = [
            ((start_x + x, start_y), (start_x + x, neighbor_y)) for x in edge_range
        ]
    elif direction == "left":
        start_x, start_y = (
            mega_chunk_pos[0] * 3 * MAP_CHUNK_SIZE,
            mega_chunk_pos[1] * 3 * MAP_CHUNK_SIZE,
        )
        neighbor_x = start_x - 1
        candidate_pairs = [
            ((start_x, start_y + y), (neighbor_x, start_y + y)) for y in edge_range
        ]
    elif direction == "right":
        start_x, start_y = (
            mega_chunk_pos[0] + 1
        ) * 3 * MAP_CHUNK_SIZE - 1, mega_chunk_pos[1] * 3 * MAP_CHUNK_SIZE
        neighbor_x = start_x + 1
        candidate_pairs = [
            ((start_x, start_y + y), (neighbor_x, start_y + y)) for y in edge_range
        ]

    pairs = []
    for pair in candidate_pairs:
        node_pos, neighbor_pos = pair
        if node_pos in zero_positions and neighbor_pos in zero_positions:
            pairs.append(pair)

    groups = []
    current_group = []
    sorted_pairs = sorted(
        pairs, key=lambda x: (x[0][0], x[0][1])
    )  # Sort by x,y coordinates

    for i in range(len(sorted_pairs)):
        current_pair = sorted_pairs[i]
        if not current_group:
            current_group.append(current_pair)
            continue

        # Check if current pair is adjacent to last pair in current group
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
    entrance_groups = []
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

                    graph_node: GraphNode = abstract.entrances[node_pos]
                    neighbor_graph_node: GraphNode = neighbor_abstract.entrances[
                        neighbor_pos
                    ]

                    graph_node.mega_chunk_connections[
                        (neighbor_graph_node.x, neighbor_graph_node.y)
                    ] = neighbor_graph_node
                    neighbor_graph_node.mega_chunk_connections[
                        (graph_node.x, graph_node.y)
                    ] = graph_node

        entrance_groups.append(entrance_nodes)
    filtered_entrance_groups = []

    for entrance_group in entrance_groups:
        graph_nodes = [node[0] for node in entrance_group]
        neighor_nodes = [node[1] for node in entrance_group]

        avg_x = sum(entrance.x for entrance in graph_nodes) / len(graph_nodes)
        avg_y = sum(entrance.y for entrance in graph_nodes) / len(graph_nodes)
        centroid_entrance = min(
            graph_nodes, key=lambda e: ((e.x - avg_x) ** 2 + (e.y - avg_y) ** 2)
        )

        neighbor_entrance = neighor_nodes[graph_nodes.index(centroid_entrance)]
        filtered_entrance_groups.append((centroid_entrance, neighbor_entrance))

    entrance_groups = filtered_entrance_groups

    # Draw centroids in white
    for entrance_pair in entrance_groups:
        entrance, neighor_entrance = entrance_pair

        entrance.mega_chunk_connections[(neighor_entrance.x, neighor_entrance.y)] = (
            neighor_entrance
        )
        neighor_entrance.mega_chunk_connections[(entrance.x, entrance.y)] = entrance

        abstraction = entrance.parent
        cluster = abstraction.cluster
        if cluster:
            cluster.entrances[(entrance.x, entrance.y)] = entrance
            rect = pygame.Rect(
                entrance.x * TILE_SIZE, entrance.y * TILE_SIZE, TILE_SIZE, TILE_SIZE
            )
            pygame.draw.rect(mega_connection_surface, (255, 255, 255), rect, 1)

        line = (
            entrance.x * TILE_SIZE + TILE_SIZE // 2,
            entrance.y * TILE_SIZE + TILE_SIZE // 2,
        ), (
            neighor_entrance.x * TILE_SIZE + TILE_SIZE // 2,
            neighor_entrance.y * TILE_SIZE + TILE_SIZE // 2,
        )
        pygame.draw.line(mega_connection_surface, (255, 255, 255), line[0], line[1], 1)


def generate_mega_entrances() -> None:
    for cx, cy in mega_chunks.keys():
        make_mega_chunk_entrances((cx, cy), "top")
        make_mega_chunk_entrances((cx, cy), "right")
        make_mega_chunk_entrances((cx, cy), "bottom")
        make_mega_chunk_entrances((cx, cy), "left")


generate_mega_entrances()

dt = time.time() - t
print(f"- Mega Chunk entrances generated. Took {dt} seconds")


def make_mega_chunk_connections() -> None:
    for mega_chunk in mega_chunks.values():
        for cluster in mega_chunk.clusters:
            for entrance in cluster.entrances.values():
                for other_entrance in cluster.entrances.values():
                    if entrance == other_entrance:
                        continue

                    entrance.mega_chunk_connections[
                        (other_entrance.x, other_entrance.y)
                    ] = other_entrance
                    other_entrance.mega_chunk_connections[(entrance.x, entrance.y)] = (
                        entrance
                    )


make_mega_chunk_connections()


for mega_chunk in mega_chunks.values():
    for chunk in mega_chunk.chunks.values():
        for node in chunk.nodes.values():
            rect = pygame.Rect(
                node.x * TILE_SIZE, node.y * TILE_SIZE, TILE_SIZE, TILE_SIZE
            )
            assert node.abstraction != None
            assert node.abstraction.cluster != None
            # pygame.draw.rect(cluster_color_surface, node.abstraction.cluster.color, rect, 0)
            pygame.draw.rect(cluster_color_surface, (33, 33, 33), rect, 0)


t = time.time()


def draw_mega_entrances():
    for mega_chunk in mega_chunks.values():
        for cluster in mega_chunk.clusters:
            for entrance in cluster.entrances.values():
                rect = pygame.Rect(
                    entrance.x * TILE_SIZE, entrance.y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                )
                pygame.draw.rect(mega_chunk_entrance_surface, (255, 255, 255), rect, 1)


def draw_chunk_connections():
    for chunk in chunks.values():
        for abstraction in chunk.abstractions:
            for entrance in abstraction.entrances.values():
                for connected_entrance in entrance.chunk_connections.values():
                    e_x, e_y = entrance.x, entrance.y
                    c_x, c_y = connected_entrance.x, connected_entrance.y
                    e_center = (
                        e_x * TILE_SIZE + TILE_SIZE // 2,
                        e_y * TILE_SIZE + TILE_SIZE // 2,
                    )
                    c_center = (
                        c_x * TILE_SIZE + TILE_SIZE // 2,
                        c_y * TILE_SIZE + TILE_SIZE // 2,
                    )
                    pygame.draw.line(
                        connection_surface, (0, 255, 0), e_center, c_center, 1
                    )


def draw_mega_connections():
    for mega_chunk in mega_chunks.values():
        for cluster in mega_chunk.clusters:
            for entrance in cluster.entrances.values():
                for other_entrance in entrance.mega_chunk_connections.values():
                    e_x, e_y = entrance.x, entrance.y
                    c_x, c_y = other_entrance.x, other_entrance.y
                    e_center = (
                        e_x * TILE_SIZE + TILE_SIZE // 2,
                        e_y * TILE_SIZE + TILE_SIZE // 2,
                    )
                    c_center = (
                        c_x * TILE_SIZE + TILE_SIZE // 2,
                        c_y * TILE_SIZE + TILE_SIZE // 2,
                    )
                    pygame.draw.line(
                        mega_connection_surface, (0, 255, 0), e_center, c_center, 1
                    )


def draw_abstractions():
    for chunk in chunks.values():
        for node in chunk.nodes.values():
            assert node.abstraction != None
            abstraction: Abstraction = node.abstraction
            rect = pygame.Rect(
                node.x * TILE_SIZE, node.y * TILE_SIZE, TILE_SIZE, TILE_SIZE
            )
            # pygame.draw.rect(abstraction_surface, abstraction.color, rect, 0)
            pygame.draw.rect(abstraction_surface, (33, 33, 33), rect, 0)


def draw_chunk_entrances():
    for chunk in chunks.values():
        for abstract in chunk.abstractions:
            for entrance in abstract.entrances.values():
                rect = pygame.Rect(
                    entrance.x * TILE_SIZE, entrance.y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                )
                pygame.draw.rect(chunk_entrance_surface, (255, 255, 255), rect, 1)


def draw_path(
    path: List[Node], color: Color = (255, 0, 0), rect_size=1, lines=False
) -> None:
    if not path:
        return

    prev_pos = None

    for node in path:
        # Draw node as a rectangle
        rect = pygame.Rect(
            node.x * TILE_SIZE + pan_offset[0],
            node.y * TILE_SIZE + pan_offset[1],
            TILE_SIZE,
            TILE_SIZE,
        )
        pygame.draw.rect(screen, color, rect, rect_size)

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


draw_abstractions()
draw_chunk_entrances()
draw_chunk_connections()

draw_mega_entrances()
draw_mega_connections()


dt = time.time() - t
print(f"- Drawing completed. Took {dt} seconds")


dt = time.time() - total
print(f"- Total time. Took {dt} seconds")


# debug_panel = pygame_gui.elements.ui_panel.UIPanel(relative_rect=pygame.Rect((350, 275), (200, 70)), manager=manager, starting_height=2)
# hello_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((350, 275), (100, 50)),
#                                              text='Say Hello',
#                                              manager=manager, parent_element=debug_panel, container=debug_panel, starting_height=1)


def get_tile_from_mouse(pos: Tuple[int, int]) -> Pos:
    """
    Converts a mouse position (x, y) to a tile position (x, y).
    """
    x, y = pos
    tile_x = (x - pan_offset[0]) // TILE_SIZE
    tile_y = (y - pan_offset[1]) // TILE_SIZE
    return tile_x, tile_y


layer = 1
selected_surface = abstraction_surface
selected_line_surface = chunk_line_surface
selected_entrance_surface = chunk_entrance_surface
selected_connection_surface = connection_surface
start_node, end_node = None, None

node_path = None
graph_path = None
cluster_path = None


pan_offset = [0, 0]
PAN_SPEED = 10

clock = pygame.time.Clock()
running = True


entity = Entity(0, 0)

while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                layer = 1
                selected_surface = abstraction_surface
                selected_line_surface = chunk_line_surface
                selected_entrance_surface = chunk_entrance_surface
                selected_connection_surface = connection_surface
            elif event.key == pygame.K_2:
                layer = 2
                selected_surface = cluster_color_surface
                selected_line_surface = mega_chunk_line_surface
                selected_entrance_surface = mega_chunk_entrance_surface
                selected_connection_surface = mega_connection_surface
                
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                tile_pos = get_tile_from_mouse(mouse_pos)
                node = get_node(tile_pos)
                if node:
                    start_node = node
                    entity.x, entity.y = (
                        start_node.x * TILE_SIZE,
                        start_node.y * TILE_SIZE,
                    )

                if start_node and end_node:
                    if start_node == end_node:
                        cluster_path = None
                        node_path = None
                        graph_path = None
                        continue

                    cluster_path = GraphAStar(start_node, end_node, 2)
                    cluster_start = cluster_path[0]
                    cluster_next = cluster_path[1]
                    graph_path = GraphAStar(
                        get_node((cluster_start.x, cluster_start.y)),
                        get_node((cluster_next.x, cluster_next.y)),
                        1,
                    )
                    node_path = NodeAStar(
                        get_node((graph_path[0].x, graph_path[0].y)),
                        get_node((graph_path[1].x, graph_path[1].y)),
                    )

                    entity.pathfind(end_node)

            elif event.button == 3:
                entity.level_0_path.clear()
                entity.level_1_path.clear()
                entity.level_2_path.clear()
                
                mouse_pos = pygame.mouse.get_pos()
                tile_pos = get_tile_from_mouse(mouse_pos)
                node = get_node(tile_pos)
                if node:
                    end_node = node

                if start_node and end_node:
                    if start_node == end_node:
                        cluster_path = None
                        node_path = None
                        graph_path = None
                        continue
                    entity.x, entity.y = (
                        start_node.x * TILE_SIZE,
                        start_node.y * TILE_SIZE,
                    )
                    cluster_path = GraphAStar(start_node, end_node, 2)
                    cluster_start = cluster_path[0]
                    cluster_next = cluster_path[1]
                    graph_path = GraphAStar(
                        get_node((cluster_start.x, cluster_start.y)),
                        get_node((cluster_next.x, cluster_next.y)),
                        1,
                    )
                    node_path = NodeAStar(
                        get_node((graph_path[0].x, graph_path[0].y)),
                        get_node((graph_path[1].x, graph_path[1].y)),
                    )
                    entity.pathfind(end_node)

        manager.process_events(event)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        pan_offset[1] += PAN_SPEED
    if keys[pygame.K_s]:
        pan_offset[1] -= PAN_SPEED
    if keys[pygame.K_a]:
        pan_offset[0] += PAN_SPEED
    if keys[pygame.K_d]:
        pan_offset[0] -= PAN_SPEED

    manager.update(dt)

    screen.fill((0, 0, 0))
    screen.blit(selected_surface, pan_offset)
    screen.blit(selected_line_surface, pan_offset)
    screen.blit(selected_entrance_surface, pan_offset)
    screen.blit(selected_connection_surface, pan_offset)

    if start_node:
        rect = pygame.Rect(
            start_node.x * TILE_SIZE + pan_offset[0],
            start_node.y * TILE_SIZE + pan_offset[1],
            TILE_SIZE,
            TILE_SIZE,
        )
        pygame.draw.rect(screen, (0, 0, 255), rect, 0)
    if end_node:
        rect = pygame.Rect(
            end_node.x * TILE_SIZE + pan_offset[0],
            end_node.y * TILE_SIZE + pan_offset[1],
            TILE_SIZE,
            TILE_SIZE,
        )
        pygame.draw.rect(screen, (255, 0, 0), rect, 0)
    if entity.level_0_path:
        draw_path(entity.level_0_path, color=(0, 255, 0))
    if entity.level_1_path:
        draw_path(entity.level_1_path, color=(255, 255, 0), lines=True)
    if entity.level_2_path:
        draw_path(entity.level_2_path, color=(255, 0, 255), lines=True)

    manager.draw_ui(screen)
    rect = pygame.Rect(
        entity.x + pan_offset[0], entity.y + pan_offset[1], TILE_SIZE, TILE_SIZE
    )
    pygame.draw.rect(screen, (255, 255, 255), rect, 0)

    if entity.target_node:
        rect = pygame.Rect(
            entity.target_node.x * TILE_SIZE + pan_offset[0],
            entity.target_node.y * TILE_SIZE + pan_offset[1],
            TILE_SIZE,
            TILE_SIZE,
        )
        pygame.draw.rect(screen, (255, 0, 0), rect, 1)

    entity.update(dt)

    pygame.display.set_caption(f"FPS: {clock.get_fps():.2f}")
    pygame.display.update()
