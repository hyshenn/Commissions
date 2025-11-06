from typing import Tuple, List, Dict, Optional
from system.lib import minescript
from raycast import raycast_block_subregions
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor

AIRLIKE: set[str] = {"minecraft:air", "minecraft:light"}
LIQUIDS: set[str] = {"minecraft:water", "minecraft:lava"}
MAX_VERTICAL_STEP: int = 1
EYE_HEIGHT: float = 1.62

class Detection:
    def __init__(self, target_blocks: List[str], radius: int, MAX_WORKERS: int):
        px, py, pz = minescript.player_position()
        self.origin: Tuple[float, float, float] = (px, py + EYE_HEIGHT, pz)
        self.radius: int = radius
        self.target_blocks: List[str] = target_blocks
        self.MAX_WORKERS: int = MAX_WORKERS

    def update_origin(self):
        px, py, pz = minescript.player_position()
        self.origin = (px, py + EYE_HEIGHT, pz)

    def calculate_radius(self) -> List[Tuple[int, int, int]]:
        self.update_origin()
        positions: List[Tuple[int,int,int]] = []
        px, py, pz = self.origin
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                for dz in range(-self.radius, self.radius + 1):
                    if math.sqrt(dx * dx + dy * dy + dz * dz) <= self.radius:
                        positions.append((int(px + dx), int(py + dy), int(pz + dz)))
        return positions

    def locate_blocks(self) -> Tuple[List[Tuple[int,int,int]], List[List[Tuple[float,float]]]]:
        positions: List[Tuple[int,int,int]] = self.calculate_radius()
        blocks: List[str] = minescript.get_block_list(positions)
        found: List[Tuple[int,int,int]] = []
        visible_angles: List[List[Tuple[float,float]]] = []

        def check_block(i: int) -> Optional[Tuple[Tuple[int,int,int], List[Tuple[float,float]]]]:
            if any(t in blocks[i] for t in self.target_blocks):
                angles: List[Tuple[float,float]] = raycast_block_subregions(self.origin, positions[i], rays_per_axis=2)
                if angles:
                    return positions[i], angles
            return None

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            for res in executor.map(check_block, range(len(positions))):
                if res:
                    pos, angles = res
                    found.append(pos)
                    visible_angles.append(angles)

        return found, visible_angles

    def is_walkable(self, pos: Tuple[int,int,int], block_cache: Dict[Tuple[int,int,int], str]) -> bool:
        nx, ny, nz = pos
        px, py, pz = minescript.player_position()
        dx, dy, dz = nx-px, ny-(py + EYE_HEIGHT), nz-pz
        if abs(dy) > MAX_VERTICAL_STEP:
            return False

        def get_block(x: int, y: int, z: int) -> str:
            key: Tuple[int,int,int] = (x,y,z)
            if key not in block_cache:
                block_cache[key] = minescript.getblock(x,y,z)
            return block_cache[key]

        below: str = get_block(nx, ny-1, nz)
        here: str = get_block(nx, ny, nz)
        above: str = get_block(nx, ny+1, nz)
        if below in AIRLIKE.union(LIQUIDS):
            return False
        if here not in AIRLIKE:
            return False
        if above not in AIRLIKE:
            return False
        if abs(dx)+abs(dz) == 2:
            a: str = get_block(nx+dx, ny, nz)
            b: str = get_block(nx, ny, nz+dz)
            if a not in AIRLIKE or b not in AIRLIKE:
                return False
        return True

    def find_nearest_safe(self, start_pos: Tuple[int,int,int], max_radius: int = 2) -> Optional[Tuple[int,int,int]]:
        block_cache: Dict[Tuple[int,int,int], str] = {}
        visited: set[Tuple[int,int,int]] = set()
        queue: deque[Tuple[int,int,int]] = deque([start_pos])

        while queue:
            x, y, z = queue.popleft()
            if (x, y, z) in visited:
                continue
            visited.add((x, y, z))
            if self.is_walkable((x, y, z), block_cache):
                return (x, y, z)
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,0,1), (0,0,-1), (0,1,0), (0,-1,0)]:
                nx, ny, nz = x+dx, y+dy, z+dz
                if abs(nx-start_pos[0]) + abs(ny-start_pos[1]) + abs(nz-start_pos[2]) <= max_radius:
                    queue.append((nx, ny, nz))
        return None
