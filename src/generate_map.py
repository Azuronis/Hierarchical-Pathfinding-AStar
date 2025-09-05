import sys
from pathlib import Path
from utility import save_map, generate_map


MAP_WIDTH, MAP_HEIGHT = 500, 500
NOISE_SCALE = 10
THRESHOLD = 0
OCTAVES = 3
MAP_NAME = f'TEST_MAP_({MAP_WIDTH}x{MAP_HEIGHT})'



ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
maps_dir = Path(__file__).resolve().parent / "maps"

path_base = maps_dir / MAP_NAME / MAP_NAME
map_data = generate_map(MAP_WIDTH, MAP_HEIGHT, NOISE_SCALE, THRESHOLD, OCTAVES)
save_map(str(path_base), map_data)
