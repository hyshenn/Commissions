from system.lib import minescript
import math
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
