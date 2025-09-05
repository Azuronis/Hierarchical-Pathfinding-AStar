from pathlib import Path
import numpy as np
import json
import os

MAP_NAME = "TEST_MAP_(500x500)"

TILE_SIZE = 10

DEPTH_COLORS = [
    (255,  64,  64, 40),
    ( 64, 255,  64, 40),
    ( 64, 128, 255, 40),
    (255, 200,  64, 40),
    (255,  64, 200, 40),
    ( 64, 255, 255, 40),
]

BACKGROUND_COLOR = (0, 0, 0, 0)
NODE_COLOR = (33, 33, 33, 255)
CONNECTION_COLOR = (0, 255, 0, 100)
ENTRANCE_COLOR = (255, 255, 255, 255)
GRID_COLOR = (66, 66, 66, 255)

START_NODE_COLOR = (0, 0, 255)
END_NODE_COLOR = (255, 0, 0)
ENTITY_COLOR = (255, 255, 255)
TARGET_NODE_COLOR = (255, 0, 0)
PATH_LEVEL_0_COLOR = (0, 255, 0)
PATH_LEVEL_1_COLOR = (255, 255, 0)
PATH_LEVEL_2_COLOR = (255, 0, 255)
ENTRANCE_INDICATOR_COLOR = (255, 0, 0)


MAP_CHUNK_SIZE = 11 # number of tiles per chunk
ENTRANCE_SPACING = MAP_CHUNK_SIZE
CHUNK_PIXEL_SIZE = TILE_SIZE * MAP_CHUNK_SIZE

CHUNK_SCALE_FACTOR = 3  # number of chunks per mega chunk

SCREEN_SCALE_FACTOR = 1.2  # factor to scale screen size
SURFACE_OFFSET = 1
PAN_SPEED = 5  # speed of camera panning
FPS = 120
FONT_SIZE = 24 
UI_TEXT_OFFSET = 10

NOISE_SCALE = 100
NOISE_THRESHOLD = 0.05
NOISE_OCTAVES = 5

PATH_LINE_WIDTH = 1
PATH_CIRCLE_RADIUS = 2




HERE = Path(__file__).resolve().parent
MAP_FILE_PATH = str(HERE / "maps" / MAP_NAME / MAP_NAME)
json_path = MAP_FILE_PATH + ".json"

meta = None
height, width = 1000, 1000
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
if meta:
    height, width = meta['height'], meta['width']

MAP_TILE_WIDTH = width
MAP_TILE_HEIGHT = height

SURFACE_WIDTH = TILE_SIZE * MAP_TILE_WIDTH
SURFACE_HEIGHT = TILE_SIZE * MAP_TILE_HEIGHT
