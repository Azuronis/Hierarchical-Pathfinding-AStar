from typing import Optional, Tuple, Dict, Set, List, Union, Iterable
import noise
from settings import MAP_TILE_WIDTH, MAP_TILE_HEIGHT, MAP_CHUNK_SIZE, TILE_SIZE
from typing import Mapping, Optional, Tuple, Dict, Set, List, Union, Iterable
from data import Node, GraphNode, Abstraction, Cluster, MetaCluster, Chunk, MegaChunk, GigaChunk, Pos, Color, chunks, mega_chunks, giga_chunks
import math


Pos = Tuple[int, int]
Color = Tuple[int, int, int]


def index_1d(x: int, y: int, width: int) -> int:
    return y * width + x

def get_neighbors(x: int, y: int) -> Iterable[Pos]:
    for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
        if 0 <= nx < MAP_TILE_WIDTH and 0 <= ny < MAP_TILE_HEIGHT:
            yield nx, ny

def generate_perlin_noise_map(
    width: int, height: int, noise_scale: float = 10.0, threshold: float = 0.0, octaves: int = 3) -> List[List[int]]:
    map_data: List[List[int]] = []

    for y in range(height):
        row = []
        for x in range(width):
            perlin_value = noise.pnoise2(x / noise_scale, y / noise_scale, octaves=octaves, persistence=0.9)
            if perlin_value < threshold:
                row.append(0)
            else:
                row.append(1)
        map_data.append(row)

    return map_data


def get_node(pos: Pos) -> Union[Node, None]:
    chunk_x, chunk_y = pos[0]//MAP_CHUNK_SIZE, pos[1]//MAP_CHUNK_SIZE
    chunk = chunks.get((chunk_x, chunk_y))
    if chunk:
        return chunk.nodes.get(pos, None)
    return None


def heuristic_node(start: Node, end: Node) -> float:
    return ((start.x - end.x) ** 2 + (start.y - end.y) ** 2)

def heuristic_graph_node(start: GraphNode, end: GraphNode) -> float:
    return math.sqrt((start.x - end.x) ** 2 + (start.y - end.y) ** 2)


def reconstruct_graph_path(came_from: Dict[GraphNode, GraphNode], end_node: GraphNode) -> List[GraphNode]:

    path: List[GraphNode] = [end_node]
    current_node = end_node

    while current_node in came_from:
        current_node = came_from[current_node]
        path.append(current_node)

    path.reverse()
    return path

def reconstruct_node_path(came_from: Dict[Node, Node], end_node: Node) -> List[Node]:

    path: List[Node] = [end_node]
    current_node = end_node

    while current_node in came_from:
        current_node = came_from[current_node]
        path.append(current_node)

    path.reverse()
    return path

import heapq
from typing import Tuple, Optional, List, Dict, Set

def NodeAStar(start_node: Node, end_node: Node) -> Tuple[Optional[List[Node]], List[Node]]:
    if (end_node.x, end_node.y) in get_neighbors(start_node.x, start_node.y):
        return [start_node, end_node]

    open_list = []
    heapq.heappush(open_list, (0, id(start_node), start_node))  # Include id() to prevent comparison issues
    
    closed_list: Set[Node] = set()

    g_scores: Dict[Node, float] = {start_node: 0.0}
    h_scores: Dict[Node, float] = {start_node: heuristic_node(start_node, end_node)}
    f_scores: Dict[Node, float] = {start_node: g_scores[start_node] + h_scores[start_node]}
    came_from: Dict[Node, Node] = {}

    while open_list:
        _, _, current_node = heapq.heappop(open_list)  # Get lowest f-score node

        if current_node == end_node:
            return reconstruct_node_path(came_from, current_node)

        closed_list.add(current_node)

        for neighbor in current_node.connections.values():
            if neighbor in closed_list:
                continue

            tentative_g_score = g_scores[current_node] + heuristic_node(current_node, neighbor)

            if tentative_g_score < g_scores.get(neighbor, float('inf')):
                came_from[neighbor] = current_node
                g_scores[neighbor] = tentative_g_score
                h_scores[neighbor] = heuristic_node(neighbor, end_node)
                f_scores[neighbor] = g_scores[neighbor] + h_scores[neighbor]

                heapq.heappush(open_list, (f_scores[neighbor], id(neighbor), neighbor))  # Use id() for tie-breaking

    return None

def create_graph_node(node: Node, layer: int) -> GraphNode:
    if layer == 1:
        abstraction = node.abstraction
        graph_node = GraphNode((node.x, node.y), abstraction)
        abstraction.entrances[(graph_node.x, graph_node.y)] = graph_node

        for entrance in abstraction.entrances.values():
            if entrance != graph_node:
                graph_node.chunk_connections[(entrance.x, entrance.y)] = entrance
                entrance.chunk_connections[(graph_node.x, graph_node.y)] = graph_node

        return graph_node
    
    elif layer == 2:
        abstraction = node.abstraction
        cluster = abstraction.cluster
        graph_node = GraphNode((node.x, node.y), abstraction)
        cluster.entrances[(graph_node.x, graph_node.y)] = graph_node

        for entrance in cluster.entrances.values():
            if entrance != graph_node:
                graph_node.mega_chunk_connections[(entrance.x, entrance.y)] = entrance
                entrance.mega_chunk_connections[(graph_node.x, graph_node.y)] = graph_node
        return graph_node
    return None

def delete_graph_node(graph_node: GraphNode, layer: int) -> None:
    if layer == 1:
        abstraction = graph_node.parent
        del abstraction.entrances[(graph_node.x, graph_node.y)]

        for entrance in abstraction.entrances.values():
            if entrance != graph_node:
                del entrance.chunk_connections[(graph_node.x, graph_node.y)]
                del graph_node.chunk_connections[(entrance.x, entrance.y)]
    elif layer == 2:
        abstraction = graph_node.parent
        cluster: Cluster = abstraction.cluster
        del cluster.entrances[(graph_node.x, graph_node.y)]

        for entrance in cluster.entrances.values():
            if entrance != graph_node:
                del entrance.mega_chunk_connections[(graph_node.x, graph_node.y)]
                del graph_node.mega_chunk_connections[(entrance.x, entrance.y)]

def GraphNodeAStar(start_graph_node: GraphNode, end_graph_node: GraphNode, layer: int) -> Optional[List[GraphNode]]:
    
    open_list: List[GraphNode] = [start_graph_node]
    closed_list: Set[GraphNode] = set()

    g_scores: Dict[GraphNode, float] = {start_graph_node: 0.0}
    h_scores: Dict[GraphNode, float] = {start_graph_node: heuristic_graph_node(start_graph_node, end_graph_node)}
    f_scores: Dict[GraphNode, float] = {start_graph_node: g_scores[start_graph_node] + h_scores[start_graph_node]}

    came_from: Dict[GraphNode, GraphNode] = {}

    # Standard A* search
    while open_list:
        current_node = min(open_list, key=lambda graph_node: f_scores.get(graph_node, float('inf')))

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

            tentative_g_score = g_scores[current_node] + heuristic_graph_node(current_node, neighbor)
            if tentative_g_score < g_scores.get(neighbor, float('inf')):
                came_from[neighbor] = current_node
                g_scores[neighbor] = tentative_g_score
                h_scores[neighbor] = heuristic_graph_node(neighbor, end_graph_node)
                f_scores[neighbor] = g_scores[neighbor] + h_scores[neighbor]

                if neighbor not in open_list:
                    open_list.append(neighbor)
                    
    return None

def GraphAStar(start_node: Node, end_node: Node, layer: int) -> Optional[List[GraphNode]]:

    start_node_abstract = start_node.abstraction
    start_node_cluster = start_node_abstract.cluster
    end_node_abstract = end_node.abstraction
    end_node_cluster = end_node_abstract.cluster
    
    created_start = False
    created_end = False
    
    start_graph_node = None
    if layer == 1:
        start_graph_node = start_node_abstract.entrances.get((start_node.x, start_node.y))
    elif layer == 2:
        start_graph_node = start_node_cluster.entrances.get((start_node.x, start_node.y))
        
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
        return None

    path = GraphNodeAStar(start_graph_node, end_graph_node, layer)

    if created_start:
        delete_graph_node(start_graph_node, layer)
    if created_end:
        delete_graph_node(end_graph_node, layer)

    return path


def raycast(start: Pos, end: Pos) -> bool:
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