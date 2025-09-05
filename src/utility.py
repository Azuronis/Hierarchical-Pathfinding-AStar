from settings import MAP_TILE_WIDTH, MAP_TILE_HEIGHT, MAP_CHUNK_SIZE, CHUNK_SCALE_FACTOR
from data import GraphNode, Pos, Matrix_2D, NodeType, Chunk, chunks
import noise
import numpy as np
import random
import os, json, math

def get_tile(tile_x, tile_y) -> bool:
    layer0 = chunks.get(0, {})
    cx, cy = tile_x // MAP_CHUNK_SIZE, tile_y // MAP_CHUNK_SIZE
    chunk = layer0.get((cx, cy))
    if chunk is None or chunk.nodes is None:
        return False
    local_x, local_y = tile_x % MAP_CHUNK_SIZE, tile_y % MAP_CHUNK_SIZE
    if local_y >= chunk.nodes.shape[0] or local_x >= chunk.nodes.shape[1]:
        return False
    return bool(chunk.nodes[local_y, local_x])

def get_neighbors(x: int, y: int) -> list[Pos]:
    neighbors = []
    for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
        if 0 <= nx < MAP_TILE_WIDTH and 0 <= ny < MAP_TILE_HEIGHT:
            neighbors.append((nx, ny))
    return neighbors

def generate_map(
    width: int,
    height: int,
    noise_scale: float = 10.0,
    threshold: float = 0.0,
    octaves: int = 3,
    world_seed: int = random.randint(1, 1000)
) -> np.ndarray:

    x_coords = np.arange(width)
    y_coords = np.arange(height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    world_x_grid = x_grid / noise_scale
    world_y_grid = y_grid / noise_scale

    data = np.zeros((height, width), dtype=bool)

    for y in range(height):
        for x in range(width):
            perlin_value = noise.pnoise2(
                world_x_grid[y, x], world_y_grid[y, x],
                octaves=octaves, persistence=0.9, base=int(world_seed)
            )
            data[y, x] = perlin_value < threshold

    return data

def is_passable(x, y):
    # Translate world coordinates (x,y) to the owning chunk and local indices
    chunk_x, chunk_y = x // MAP_CHUNK_SIZE, y // MAP_CHUNK_SIZE
    layer_0_chunks = chunks.get(0, {})
    chunk = layer_0_chunks.get((chunk_x, chunk_y))
    if chunk is None or chunk.nodes is None:
        return False

    # Compute local indices within this chunk
    local_x = x - chunk_x * MAP_CHUNK_SIZE
    local_y = y - chunk_y * MAP_CHUNK_SIZE

    # Bounds check against the actual chunk tile array shape
    h, w = chunk.nodes.shape
    if local_x < 0 or local_y < 0 or local_x >= w or local_y >= h:
        return False

    return bool(chunk.nodes[local_y, local_x])


def calculate_hpa_depth(map_width: int, map_height: int,
                        chunk_size: int = MAP_CHUNK_SIZE,
                        upper_chunk_size: int = CHUNK_SCALE_FACTOR) -> int:
    chunks_x = math.ceil(map_width  / chunk_size)
    chunks_y = math.ceil(map_height / chunk_size)
    depth = 1
    while chunks_x > 2 or chunks_y > 2:
        chunks_x = math.ceil(chunks_x / upper_chunk_size)
        chunks_y = math.ceil(chunks_y / upper_chunk_size)
        depth += 1
    return depth

def save_map(path_base: str, world: np.ndarray, meta: dict | None = None) -> None:
    # Ensure the *parent directory* exists
    folder = os.path.dirname(path_base)
    if folder:
        os.makedirs(folder, exist_ok=True)

    # Save binary map data
    np.savez_compressed(path_base + ".npz", world=world.astype(np.bool_))

    # Save metadata JSON
    info = {
        "version": 1,
        "dtype": "bool",
        "height": int(world.shape[0]),
        "width": int(world.shape[1]),
        "map_chunk_size": int(MAP_CHUNK_SIZE),
    }
    if meta:
        info.update(meta)

    with open(path_base + ".json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def load_map(path_base: str) -> tuple[np.ndarray, dict]:
    with np.load(path_base + ".npz") as z:
        world = z["world"].astype(np.bool_)
    meta = {}
    json_path = path_base + ".json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return world, meta

def print_chunk_structure(chunk: Chunk, max_depth):
    indent = max_depth - chunk.depth - 1
    char = ''
    if indent != 0:
        char = '└'
    print(f'{'     '*indent}{char}({chunk.depth}) Chunk {chunk}')
    if chunk.children:
        for chunk_child in chunk.children.values():
            print_chunk_structure(chunk_child, max_depth)

def _tile_span(depth: int) -> int:
    # how many world tiles a chunk spans per side at this depth
    return MAP_CHUNK_SIZE * (CHUNK_SCALE_FACTOR ** depth)

def populate_chunks_from_map(world: Matrix_2D, max_depth: int):
    H, W = world.shape

    # init/clear layers we will build
    for d in range(max_depth):
        chunks[d] = {}

    # ---- build TOP layer (no nodes) ----
    top_depth = max_depth - 1
    print(f"initializing chunk layer {top_depth}")
    top_span = _tile_span(top_depth)
    top_cx = math.ceil(W / top_span)
    top_cy = math.ceil(H / top_span)

    for cy in range(top_cy):
        for cx in range(top_cx):
            chunk = Chunk(cx, cy, top_depth, nodes=None)
            chunks[top_depth][(cx, cy)] = chunk  # <-- fix: use top_depth

    # ---- descend layers down to depth 0 ----
    for depth in range(top_depth - 1, -1, -1):
        span = _tile_span(depth)
        cx_count = math.ceil(W / span)
        cy_count = math.ceil(H / span)
        print(f"initializing chunk layer {depth}")

        for cy in range(cy_count):
            for cx in range(cx_count):
                # find parent at depth+1
                p_cx = cx // CHUNK_SCALE_FACTOR
                p_cy = cy // CHUNK_SCALE_FACTOR
                parent = chunks[depth + 1].get((p_cx, p_cy))

                # nodes only for depth 0
                if depth == 0:
                    x0, y0 = cx * MAP_CHUNK_SIZE, cy * MAP_CHUNK_SIZE
                    x1, y1 = min(x0 + MAP_CHUNK_SIZE, W), min(y0 + MAP_CHUNK_SIZE, H)
                    nodes = world[y0:y1, x0:x1].copy()
                else:
                    nodes = None

                child = Chunk(cx, cy, depth, nodes=nodes)
                child.parent = parent
                chunks[depth][(cx, cy)] = child

                if parent is not None:
                    # key children by their (cx, cy) at the child depth
                    parent.children[(cx, cy)] = child


def stitch_map_from_chunks() -> np.ndarray:
    layer0 = chunks.get(0, {})
    if not layer0:
        return np.zeros((0, 0), dtype=np.bool_)
    max_cx = max(cx for (cx, _cy) in layer0.keys())
    max_cy = max(cy for (_cx, cy) in layer0.keys())
    H = (max_cy + 1) * MAP_CHUNK_SIZE
    W = (max_cx + 1) * MAP_CHUNK_SIZE
    full = np.zeros((H, W), dtype=np.bool_)
    for (cx, cy), ch in layer0.items():
        if ch.nodes is None:
            continue
        y0, x0 = cy * MAP_CHUNK_SIZE, cx * MAP_CHUNK_SIZE
        h, w = ch.nodes.shape
        full[y0:y0+h, x0:x0+w] = ch.nodes
    return full

def generate_chunk_noise(
    chunk_x: int, 
    chunk_y: int, 
    chunk_size: int,
    noise_scale: float = 10.0,
    threshold: float = 0.0,
    octaves: int = 3,
    world_seed: int = random.randint(1, 1000)
) -> np.ndarray:
    
    world_start_x = chunk_x * chunk_size
    world_start_y = chunk_y * chunk_size
    
    local_x = np.arange(chunk_size)
    local_y = np.arange(chunk_size)
    x_grid, y_grid = np.meshgrid(local_x, local_y)
    
    world_x_grid = (world_start_x + x_grid) / noise_scale
    world_y_grid = (world_start_y + y_grid) / noise_scale
    
    chunk_data = np.zeros((chunk_size, chunk_size), dtype=bool)
    
    for y in range(chunk_size):
        for x in range(chunk_size):
            perlin_value = noise.pnoise2(
                world_x_grid[y, x], world_y_grid[y, x], 
                octaves=octaves, persistence=0.9, base=int(world_seed)
            )
            chunk_data[y, x] = perlin_value < threshold
    
    return chunk_data
