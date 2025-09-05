# main_v2.py
# v1.7.1

import pygame
from collections import deque
from settings import (
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    SURFACE_WIDTH,
    SURFACE_HEIGHT,
    SCREEN_SCALE_FACTOR,
    SURFACE_OFFSET,
    MAP_CHUNK_SIZE,
    TILE_SIZE,
    CHUNK_SCALE_FACTOR,
    DEPTH_COLORS,
    MAP_FILE_PATH
)

from utility import (
    load_map,
    populate_chunks_from_map,
    get_neighbors,
    get_tile,
    calculate_hpa_depth,
    is_passable
)

from data import (
    Pos, chunks, Chunk, GraphNode, Abstraction
)


pygame.init()
pygame.font.init()

info = pygame.display.Info()
screen_width, screen_height = info.current_w, info.current_h
WIDTH  = screen_width  // SCREEN_SCALE_FACTOR
HEIGHT = screen_height // SCREEN_SCALE_FACTOR
screen = pygame.display.set_mode((WIDTH, HEIGHT), vsync=True)

OFFSET = SURFACE_OFFSET
OFFSET_WIDTH  = SURFACE_WIDTH  + OFFSET
OFFSET_HEIGHT = SURFACE_HEIGHT + OFFSET

max_depth = calculate_hpa_depth(MAP_TILE_WIDTH, MAP_TILE_HEIGHT)
world_data, meta = load_map(MAP_FILE_PATH)
WORLD_H, WORLD_W = world_data.shape
WORLD_PX_W = WORLD_W * TILE_SIZE
WORLD_PX_H = WORLD_H * TILE_SIZE

populate_chunks_from_map(world_data, max_depth)
del world_data  # Free memory


def flood_fill_chunk_layer_0(chunk: Chunk, entrances: dict[Pos, GraphNode]):
    chunk.abstractions.clear()
    h, w = chunk.nodes.shape
    visited = set()
    
    # Convert global entrance coordinates to local chunk coordinates
    local_entrances = {}
    for pos, graph_node in entrances.items():
        global_x, global_y = pos
        local_x = global_x - (chunk.x * MAP_CHUNK_SIZE)
        local_y = global_y - (chunk.y * MAP_CHUNK_SIZE)

        local_entrances[(local_x, local_y)] = graph_node
    
    for y in range(h):
        for x in range(w):
            if not chunk.nodes[y, x]:  # Skip impassable tiles
                continue
                
            pos = (x, y)
            if pos in visited:
                continue
                
            # Start new component
            queue = deque([pos])
            abstraction = Abstraction(chunk)
            
            chunk.abstractions.add(abstraction)
            
            while queue:
                current_pos = queue.popleft()
                if current_pos in visited:
                    continue
                
                # Now use local coordinates to look up entrances
                graph_node = local_entrances.get(current_pos, None)
                if graph_node:
                    abstraction.entrances[current_pos] = graph_node
                    graph_node.abstraction = abstraction

                visited.add(current_pos)
                
                cx, cy = current_pos
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    neighbor_pos = (nx, ny)

                    if (0 <= nx < w and 0 <= ny < h and 
                        chunk.nodes[ny, nx] and 
                        neighbor_pos not in visited):
                        queue.append(neighbor_pos)
                        
def flood_fill_chunk(chunk: Chunk):
    layer = chunk.depth
    if layer == 0: # Layer 0 special flood fill.
        entrances = generate_entrances_layer_0(chunk)
        flood_fill_chunk_layer_0(chunk, entrances)
        
    # Else fill other chunks through entrance connections. 
    else:
        abstractions = chunk.abstractions
        # queue = deque([pos])
        # abstraction = Abstraction(chunk)
        
        # chunk.abstractions.add(abstraction)
        
        # while queue:
        # pass
        # TODO. Finish this
        
        
def make_intra_connections(chunk: Chunk):
    for abstraction in chunk.abstractions:
        # Get all entrances for this abstraction
        entrance_items = list(abstraction.entrances.items())  # (local_pos, graph_node) pairs
        
        for i, (local_pos1, graph_node1) in enumerate(entrance_items):
            for j, (local_pos2, graph_node2) in enumerate(entrance_items[i+1:], i+1):
                if graph_node1 == graph_node2:
                    continue
                    
                # Store connections using the graph nodes themselves
                graph_node1.connections[local_pos2] = graph_node2
                graph_node2.connections[local_pos1] = graph_node1


def generate_entrances_layer_0(chunk: Chunk):
    
    entrances: dict[Pos, GraphNode] = {}
    
    chunk_depth = chunk.depth
    chunk_node_size = MAP_CHUNK_SIZE * (CHUNK_SCALE_FACTOR ** chunk_depth)
    
    base_x = chunk.x * MAP_CHUNK_SIZE
    base_y = chunk.y * MAP_CHUNK_SIZE

    layer_0_chunks = chunks.get(0, {})

    for wall_direction in [0, 1, 2, 3]:  # top, bottom, left, right
        if wall_direction == 0:  # top
            y_values = [base_y]
            x_values = range(base_x, base_x + chunk_node_size)
            neighbor_key = (chunk.x, chunk.y - 1)
            neighbor_offset = (0, -1)
        elif wall_direction == 1:  # bottom
            y_values = [base_y + chunk_node_size - 1]
            x_values = range(base_x, base_x + chunk_node_size)
            neighbor_key = (chunk.x, chunk.y + 1)
            neighbor_offset = (0, 1)
        elif wall_direction == 2:  # left
            y_values = range(base_y, base_y + chunk_node_size)
            x_values = [base_x]
            neighbor_key = (chunk.x - 1, chunk.y)
            neighbor_offset = (-1, 0)
        elif wall_direction == 3:  # right
            y_values = range(base_y, base_y + chunk_node_size)
            x_values = [base_x + chunk_node_size - 1]
            neighbor_key = (chunk.x + 1, chunk.y)
            neighbor_offset = (1, 0)

        border_groups = []
        wall_nodes = []
        for y in y_values:
            for x in x_values:
                if is_passable(x, y) and is_passable(x + neighbor_offset[0], y + neighbor_offset[1]):
                    wall_nodes.append((x, y))  # world coords
                else:
                    if wall_nodes:
                        border_groups.append(wall_nodes.copy())
                        wall_nodes = []
        if wall_nodes:
            border_groups.append(wall_nodes.copy())
        
        neighbor_chunk = layer_0_chunks.get(neighbor_key)

        # process each group
        for group in border_groups:
            found_neighbor = False
            if neighbor_chunk:
                for gx, gy in group:
                    ng_x, ng_y = gx + neighbor_offset[0], gy + neighbor_offset[1]
                    # search entrances in neighbor abstractions
                    for neighbor_abstraction in neighbor_chunk.abstractions:
                        for (lx, ly), neighbor_node in neighbor_abstraction.entrances.items():
                            glob_nx = neighbor_chunk.x * MAP_CHUNK_SIZE + lx
                            glob_ny = neighbor_chunk.y * MAP_CHUNK_SIZE + ly
                            if (glob_nx, glob_ny) == (ng_x, ng_y):
                                # neighbor already has an entrance here → make one on this side too
                                graph_node = entrances.get((gx, gy))
                                if not graph_node:
                                    graph_node = GraphNode(gx, gy)
                                    entrances[(gx, gy)] = graph_node
                                # link both ways
                                neighbor_node.connections[(gx, gy)] = graph_node
                                graph_node.connections[(glob_nx, glob_ny)] = neighbor_node
                                found_neighbor = True
                                break
                        if found_neighbor:
                            break
                    if found_neighbor:
                        break
            
            # if no neighbor entrance matched → put one in the middle
            if not found_neighbor:
                gx, gy = group[len(group) // 2]
                if (gx, gy) not in entrances:
                    entrances[(gx, gy)] = GraphNode(gx, gy)

    return entrances


def generate_all_chunk_entrances():
    for chunk_data in chunks.values():
        for chunk in chunk_data.values():
            flood_fill_chunk(chunk)
    

def link_neighbor_entrances():
    layer_0 = chunks.get(0, {})
    
    # Build a lookup: world position -> GraphNode
    world_to_node = {}
    for chunk in layer_0.values():
        for abstraction in chunk.abstractions:
            for (lx, ly), node in abstraction.entrances.items():
                gx = chunk.x * MAP_CHUNK_SIZE + lx
                gy = chunk.y * MAP_CHUNK_SIZE + ly
                world_to_node[(gx, gy)] = node

    # Now connect any entrances that are adjacent across chunk borders
    for (gx, gy), node in world_to_node.items():
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            ng = (gx + dx, gy + dy)
            neighbor = world_to_node.get(ng)
            if neighbor is None:
                continue
            # Only connect if they're from different chunks
            if (gx // MAP_CHUNK_SIZE, gy // MAP_CHUNK_SIZE) != (ng[0] // MAP_CHUNK_SIZE, ng[1] // MAP_CHUNK_SIZE):
                node.connections[ng] = neighbor
                neighbor.connections[(gx, gy)] = node

def generate_all_intra_connections():
    for chunk_data in chunks.values():
        for chunk in chunk_data.values():
            make_intra_connections(chunk)
            
            
generate_all_chunk_entrances()
generate_all_intra_connections()
link_neighbor_entrances()

node_surface  = pygame.Surface((WORLD_PX_W, WORLD_PX_H), flags=pygame.SRCALPHA)
node_grid     = pygame.Surface((WORLD_PX_W, WORLD_PX_H), flags=pygame.SRCALPHA)
grid_surfaces = [pygame.Surface((WORLD_PX_W, WORLD_PX_H), flags=pygame.SRCALPHA)
                 for _ in range(max_depth)]
entrance_surfaces = [pygame.Surface((WORLD_PX_W, WORLD_PX_H), flags=pygame.SRCALPHA)
                 for _ in range(max_depth)]
inter_connection_surfaces = [pygame.Surface((WORLD_PX_W, WORLD_PX_H), flags=pygame.SRCALPHA)
                 for _ in range(max_depth)]
intra_connection_surfaces = [pygame.Surface((WORLD_PX_W, WORLD_PX_H), flags=pygame.SRCALPHA)
                 for _ in range(max_depth)]


def draw_nodes(surface: pygame.Surface):
    rect = pygame.Rect(0, 0, TILE_SIZE, TILE_SIZE)
    for ch in chunks.get(0, {}).values():
        nodes = ch.nodes
        if nodes is None:
            continue
        h, w = nodes.shape
        wx0 = ch.x * MAP_CHUNK_SIZE
        wy0 = ch.y * MAP_CHUNK_SIZE

        for ly in range(h):
            py = (wy0 + ly) * TILE_SIZE
            rect.y = py
            for lx in range(w):
                if not nodes[ly, lx]:
                    continue
                px = (wx0 + lx) * TILE_SIZE
                rect.x = px
                pygame.draw.rect(surface, (200, 200, 200), rect)

def draw_grid(color: tuple[int, int, int, int], tile_size: int = TILE_SIZE):
    for x in range(0, OFFSET_WIDTH, tile_size):
        pygame.draw.line(node_grid, color, (x, 0), (x, OFFSET_HEIGHT))
    for y in range(0, OFFSET_HEIGHT, tile_size):
        pygame.draw.line(node_grid, color, (0, y), (OFFSET_WIDTH, y))
        
def draw_tile_grid(surface: pygame.Surface, color=(66,66,66,60)):
    surface.fill((0,0,0,0))
    for x in range(0, WORLD_PX_W, TILE_SIZE):
        pygame.draw.line(surface, color, (x, 0), (x, WORLD_PX_H))
    for y in range(0, WORLD_PX_H, TILE_SIZE):
        pygame.draw.line(surface, color, (0, y), (WORLD_PX_W, y))

def draw_chunk_overlay_surface(surface: pygame.Surface, depth: int,
                               fill_color=None,
                               line_color=(255,255,255,160),
                               line_width=1,
                               show_labels: bool = True,
                               label_color: tuple[int,int,int] = (255,255,255),
                               label_font_size: int = 16):
    surface.fill((0, 0, 0, 0))
    depth_chunks = chunks.get(depth, {})
    if not depth_chunks:
        return

    if fill_color is None:
        fill_color = DEPTH_COLORS[depth % len(DEPTH_COLORS)]

    step_tiles = MAP_CHUNK_SIZE * (CHUNK_SCALE_FACTOR ** depth)
    font = pygame.font.Font(None, label_font_size) if show_labels else None

    for ch in depth_chunks.values():
        # clamp chunk rect at world edges
        x0_tiles = ch.x * step_tiles
        y0_tiles = ch.y * step_tiles
        w_tiles  = max(0, min(step_tiles, WORLD_W - x0_tiles))
        h_tiles  = max(0, min(step_tiles, WORLD_H - y0_tiles))
        if w_tiles == 0 or h_tiles == 0:
            continue

        # tiles → pixels
        x0 = x0_tiles * TILE_SIZE
        y0 = y0_tiles * TILE_SIZE
        w  = w_tiles  * TILE_SIZE
        h  = h_tiles  * TILE_SIZE

        # same visual treatment as before
        pygame.draw.rect(surface, fill_color, (x0, y0, w, h))
        pygame.draw.rect(surface, line_color, (x0, y0, w, h), width=line_width)

        # label
        if font:
            label = font.render(f"({ch.x},{ch.y}) d{ch.depth} n({ch.tile_count()})", True, label_color)
            surface.blit(label, (x0 + 3, y0 + 2))

def draw_entrances(surface: pygame.Surface, depth: int):
    """Draw entrance nodes for the specified depth level"""
    surface.fill((0, 0, 0, 0))  # Clear with transparency
    
    # Get chunks for this depth level
    depth_chunks = chunks.get(depth, {})
    
    for chunk in depth_chunks.values():
        for abstraction in chunk.abstractions:
            # Draw each entrance node
            for entrance_pos, entrance_node in abstraction.entrances.items():
                # Convert local chunk coordinates to global world coordinates
                px = entrance_node.x * TILE_SIZE
                py = entrance_node.y * TILE_SIZE

                
                # Draw entrance as a colored circle
                pygame.draw.circle(surface, (255, 0, 0, 255), (px + TILE_SIZE//2, py + TILE_SIZE//2), TILE_SIZE//3)
                
                # Draw a border around the entrance
                pygame.draw.circle(surface, (255, 255, 255, 255), (px + TILE_SIZE//2, py + TILE_SIZE//2), TILE_SIZE//3, 2)

def draw_intra_chunk_connections(surface: pygame.Surface, depth: int):
    surface.fill((0, 0, 0, 0))
    depth_chunks = chunks.get(depth, {})
    
    for chunk in depth_chunks.values():
        for abstraction in chunk.abstractions:
            entrances = list(abstraction.entrances.values())
            for i, entrance1 in enumerate(entrances):
                for j, entrance2 in enumerate(entrances[i+1:], i+1):
                    global_x1 = entrance1.x
                    global_y1 = entrance1.y
                    global_x2 = entrance2.x
                    global_y2 = entrance2.y
                    
                    px1 = global_x1 * TILE_SIZE + TILE_SIZE//2
                    py1 = global_y1 * TILE_SIZE + TILE_SIZE//2
                    px2 = global_x2 * TILE_SIZE + TILE_SIZE//2
                    py2 = global_y2 * TILE_SIZE + TILE_SIZE//2
                    
                    pygame.draw.line(surface, (0, 0, 255, 180), (px1, py1), (px2, py2), 2)

def draw_inter_chunk_connections(surface: pygame.Surface, depth: int):
    surface.fill((0, 0, 0, 0))
    depth_chunks = chunks.get(depth, {})
    for chunk in depth_chunks.values():
        for abstraction in chunk.abstractions:
            entrances = list(abstraction.entrances.values())
            for entrance in entrances:
                for neighbor_pos, neighbor_node in entrance.connections.items():
                    if neighbor_node.abstraction == entrance.abstraction:
                        continue
                    if neighbor_node.abstraction is None:
                        continue
                    global_x1 = entrance.x
                    global_y1 = entrance.y
                    global_x2 = neighbor_node.x
                    global_y2 = neighbor_node.y
                    px1 = global_x1 * TILE_SIZE + TILE_SIZE//2
                    py1 = global_y1 * TILE_SIZE + TILE_SIZE//2
                    px2 = global_x2 * TILE_SIZE + TILE_SIZE//2
                    py2 = global_y2 * TILE_SIZE + TILE_SIZE//2
                    
                    pygame.draw.line(surface, (0, 0, 255, 180), (px1, py1), (px2, py2), 2)
                    
draw_nodes(node_surface)
draw_tile_grid(node_grid)

# Create entrance surfaces for each depth
for i in range(max_depth):
    draw_chunk_overlay_surface(
        grid_surfaces[i],
        depth=i,
        fill_color=DEPTH_COLORS[i % len(DEPTH_COLORS)],
        line_color=(255,255,255,255),
        line_width=1,
        show_labels=True
    )
    # Create entrance surface for this depth
    draw_entrances(entrance_surfaces[i], i)
    draw_intra_chunk_connections(intra_connection_surfaces[i], i)
    draw_inter_chunk_connections(inter_connection_surfaces[i], i)
    
draw_grid((66, 66, 66, 60), tile_size=TILE_SIZE)
draw_nodes(node_surface)



clock = pygame.time.Clock()
running = True
FPS = 60
PAN_SPEED = 50
pan_offset = [0, 0]
current_depth = 0

while running:
    dt = clock.tick(FPS) / 1000.0
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                current_depth = 0
            elif event.key == pygame.K_2 and max_depth > 1:
                current_depth = 1
            elif event.key == pygame.K_3 and max_depth > 2:
                current_depth = 2
            elif event.key == pygame.K_4 and max_depth > 3:
                current_depth = 3
            elif event.key == pygame.K_5 and max_depth > 4:
                current_depth = 4

    keys = pygame.key.get_pressed()
    moved = False
    if keys[pygame.K_w]:
        pan_offset[1] += PAN_SPEED  # up
        moved = True
    if keys[pygame.K_s]:
        pan_offset[1] -= PAN_SPEED  # down
        moved = True
    if keys[pygame.K_a]:
        pan_offset[0] += PAN_SPEED  # left
        moved = True
    if keys[pygame.K_d]:
        pan_offset[0] -= PAN_SPEED  # right
        moved = True
    
    screen.fill((0, 0, 0))
    
    
    pan = (pan_offset[0], pan_offset[1])
    
    mouse_x, mouse_y = pygame.mouse.get_pos()
    mouse_x, mouse_y = mouse_x - pan[0], mouse_y - pan[1]
    tile_x, tile_y = mouse_x // TILE_SIZE, mouse_y // TILE_SIZE
    is_tile = get_tile(tile_x, tile_y)

    
    screen.blit(node_surface, (pan_offset[0], pan_offset[1]))
    screen.blit(node_grid, (pan_offset[0], pan_offset[1]))
    screen.blit(grid_surfaces[current_depth], (pan_offset[0], pan_offset[1]))
    screen.blit(intra_connection_surfaces[current_depth], (pan_offset[0], pan_offset[1])) # Blit connection surface
    screen.blit(inter_connection_surfaces[current_depth], (pan_offset[0], pan_offset[1])) # Blit connection surface
    screen.blit(entrance_surfaces[current_depth], (pan_offset[0], pan_offset[1])) # Blit entrance surface

    # Node outline highlight (mouse hover) ---------------------------
    rect =  pygame.Rect(
            (tile_x * TILE_SIZE) + pan_offset[0],
            (tile_y * TILE_SIZE) + pan_offset[1],
            TILE_SIZE, TILE_SIZE
        )
    outline_color = (255, 255, 255)
    if is_tile:
        outline_color = (255, 255, 0)
    pygame.draw.rect(screen, outline_color, rect, width=2)
    # --------------------------------------------------------


    pygame.display.set_caption(f"Depth: {current_depth} | FPS: {clock.get_fps():.2f} | Tile Pos: ({str(tile_x).zfill(4)}, {str(tile_y).zfill(4)}) Passable: {is_tile}")
    pygame.display.update()
