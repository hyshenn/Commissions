from typing import Tuple, List, Dict, Optional
from system.lib import minescript
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor

class Ray:
    def __init__(self, origin):
        self.origin = origin

    def normalize(self, vector):
        vx, vy, vz = vector
        length = math.sqrt(vx*vx + vy*vy + vz*vz)
        if length == 0:
            return (0.0, 0.0, 0.0)
        return (vx/length, vy/length, vz/length)

    def sign_DDA(self, vector):
        return tuple(1 if c > 0 else -1 if c < 0 else 0 for c in vector)

    def raycast(self, direction, target_block, max_dist=20):
        ox, oy, oz = self.origin
        dir_norm = self.normalize(direction)
        bx, by, bz = map(math.floor, (ox, oy, oz))
        step = self.sign_DDA(dir_norm)

        tMax = [0.0, 0.0, 0.0]
        tDelta = [0.0, 0.0, 0.0]

        for i in range(3):
            if dir_norm[i] != 0:
                if step[i] > 0:
                    tMax[i] = (math.floor([ox, oy, oz][i] + 1) - [ox, oy, oz][i]) / dir_norm[i]
                else:
                    tMax[i] = ([ox, oy, oz][i] - math.floor([ox, oy, oz][i])) / -dir_norm[i]
                tDelta[i] = abs(1 / dir_norm[i])
            else:
                tMax[i] = float('inf')
                tDelta[i] = float('inf')

        t = 0
        while t <= max_dist:
            block = minescript.getblock(bx, by, bz)
            if block and block != "minecraft:air":
                if (bx, by, bz) == target_block:
                    hit_point = (ox + dir_norm[0]*t, oy + dir_norm[1]*t, oz + dir_norm[2]*t)
                    dx = hit_point[0] - ox
                    dy = hit_point[1] - oy
                    dz = hit_point[2] - oz
                    dist_xz = math.sqrt(dx*dx + dz*dz)
                    yaw = math.degrees(math.atan2(dz, dx)) - 90
                    pitch = -math.degrees(math.atan2(dy, dist_xz))
                    return (hit_point, yaw, pitch)
                else:
                    return None

            axis = tMax.index(min(tMax))
            t = tMax[axis]
            if axis == 0:
                bx += step[0]
                tMax[0] += tDelta[0]
            elif axis == 1:
                by += step[1]
                tMax[1] += tDelta[1]
            else:
                bz += step[2]
                tMax[2] += tDelta[2]

        return None

def raycast_block_subregions(origin, block_pos, max_dist=20, rays_per_axis=4):
    bx, by, bz = block_pos
    step = 1.0 / rays_per_axis
    center = (bx + 0.5, by + 0.5, bz + 0.5)
    hits = []

    def raycast_single(target_pos):
        dx = target_pos[0] - origin[0]
        dy = target_pos[1] - origin[1]
        dz = target_pos[2] - origin[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if distance > max_dist:
            return None
        ray = Ray(origin)
        hit = ray.raycast((dx, dy, dz), block_pos, max_dist)
        if hit:
            hit_point, yaw, pitch = hit
            dist_to_center = math.sqrt(
                (hit_point[0]-center[0])**2 +
                (hit_point[1]-center[1])**2 +
                (hit_point[2]-center[2])**2
            )
            return (dist_to_center, yaw, pitch)
        return None

    targets = [
        (bx + (i + 0.5) * step, by + (k + 0.5) * step, bz + (j + 0.5) * step)
        for i in range(rays_per_axis)
        for j in range(rays_per_axis)
        for k in range(rays_per_axis)
    ]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(raycast_single, targets))

    hits = [r for r in results if r is not None]
    if hits:
        hits.sort(key=lambda x: x[0])
        _, yaw, pitch = hits[0]
        return [(yaw, pitch)]
    return []


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

