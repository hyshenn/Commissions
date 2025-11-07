from system.lib import minescript
from system.lib.java import JavaClass
from detector import Detection
from rotation import look #I preferred this one since i like it better, your choice tho
from ThetaStar import path_walk_to, path_find
from blockRenderer import render_blocks, stop_rendering
from raycast import raycast_block_subregions
import threading, time, math, random

ClickType = JavaClass("net.minecraft.world.inventory.ClickType")
Minecraft = JavaClass("net.minecraft.client.Minecraft")

BLOCK_LIST = [
    "minecraft:cyan_terracotta",
    "minecraft:gray_wool",
    "minecraft:dark_prismarine",
    "minecraft:prismarine_bricks",
    "minecraft:prismarine",
    "minecraft:polished_diorite"
        ]

def path_helper(path):
    return [(x, y-1, z) for x, y, z in path]


#Get the commissions from the tablist
def get_commissions():
    connection = Minecraft.getInstance().getConnection()
    if not connection:
        return None
    players = connection.getOnlinePlayers()
    if not players:
        return None
    current_commissions = []
    try:
        for player in players.toArray():
            display = player.getTabListDisplayName()
            if display:
                text = display.getString()
                if ":" in text and "%" in text:
                    current_commissions.append(text)
    except:
        pass
    return current_commissions

#Travel to the commission based on the returned value of get_commissions()
def travel_to_commission():
    AREA_LIST = {
        (-88, 147, -14): "Rampart's Quarry",
        (-126, 172, -76): "Upper Mines",
        (52, 198, -25): "Lava Springs",
        (149, 150, 30): "Royal Mines",
        (25, 129, 33): "Cliffside Veins"
    }
    #Warp waypoints(It's most likely too fragile and exact might need to change it in the future)
    EATHER_WARPS = {
        "Lava Springs": [(31, 196, -10)],
        "Upper Mines": [(-33, 174, -31), (-66, 222, -50), (-114, 196, -30)]
    }

    commissions = get_commissions()
    area = next((a for c in commissions for a in AREA_LIST.values() if a in c.strip()), None)
    if not area:
        return
    #helpers
    def warp(area_name):
        positions = EATHER_WARPS.get(area_name, [])
        if not positions:
            return
        render_blocks(positions)

        for wx, wy, wz in positions:
            minescript.player_press_sneak(True)
            time.sleep(0.2)

            px, py, pz = minescript.player_position()
            rays = raycast_block_subregions((px, py+1.52, pz), (wx, wy, wz), max_dist=500, rays_per_axis=4)
            if rays:
                yaw, pitch = rays[0]
                look(yaw, pitch)
                print(f"Looked at warp {wx, wy, wz}")
            else:
                print(f"No ray hit for warp {wx, wy, wz}")

            minescript.player_press_use(True)
            time.sleep(0.1)
            minescript.player_press_use(False)
            minescript.player_press_sneak(False)
            time.sleep(0.5)
        stop_rendering()

    def go(area_name, pre_points=(), go_to_area=True):
        for point in pre_points:
            path = path_find(minescript.player_position(), point)
            render_blocks(path_helper(path))
            path_walk_to(path)
            stop_rendering()
            time.sleep(0.1)
        if go_to_area:
            for pos, name in AREA_LIST.items():
                if name == area_name:
                    path = path_find(minescript.player_position(), pos)
                    render_blocks(path_helper(path))
                    path_walk_to(path)
                    stop_rendering()
                    return


    if area == "Lava Springs":
        go(area, pre_points=[(4, 147, -29)], go_to_area=False)
        warp(area)
        go(area)

    elif area == "Upper Mines":
        go(area, pre_points=[(-2, 147, -19)], go_to_area=False)
        warp(area)
        go(area)

    elif area == "Cliffside Veins":
        go(area, pre_points=[(38, 136, 18)], final_dist=2)

    elif area == "Royal Mines":
        go(area, pre_points=[(47, 136, 19)], final_dist=2)

    elif area == "Rampart's Quarry":
        go(area)

    elif area not in AREA_LIST and "Mine" in area:
        go(None, pre_points=[(23, 145, -26)], go_to_area=False)

#claim commissions via shift clicking them
def claim_commissions():
    time.sleep(0.5)
    screen = Minecraft.getInstance().screen
    if screen is None:
        return
    container_menu = screen.getMenu()
    items = minescript.container_get_items()
    commission_slots = [item.slot for item in items if item.item == "minecraft:writable_book"]
    mouse_button = 0
    for slot in commission_slots:
        try:
            Minecraft.getInstance().gameMode.handleInventoryMouseClick(
                container_menu.containerId,
                slot,
                mouse_button,
                ClickType.QUICK_MOVE,
                Minecraft.getInstance().player
            )
            time.sleep(0.5)
        except:
            pass

#mob commissions
def kill_commission():
    #get the commissions
    commissions = get_commissions()
    target_type = next((t for c in commissions for t in ["Ice Walker", "Goblin"] if t in c.strip()), None)
    if not target_type:
        return
    #thread safety
    look_lock = threading.Lock()
    current_look = [None]
    look_thread = [None]
    look_thread_running = [False]
    #get the yaw, pitch for the mob
    def calc_look(px, py, pz, ex, ey, ez):
        py += 1.62
        dx, dy, dz = ex - px, ey - py, ez - pz
        dist_xz = math.sqrt(dx*dx + dz*dz)
        yaw = math.degrees(math.atan2(-dx, dz))
        pitch = math.degrees(-math.atan2(dy, dist_xz))
        return yaw, pitch
    #thread safety
    def thread_look():
        look_thread_running[0] = True
        while look_thread_running[0]:
            with look_lock:
                target = current_look[0]
            if target:
                yaw, pitch = target
                look(yaw, pitch)
            else:
                time.sleep(0.05)
            time.sleep(0.01)

    def start_look_thread():
        if look_thread[0] is None or not look_thread[0].is_alive():
            look_thread[0] = threading.Thread(target=thread_look, daemon=True)
            look_thread[0].start()

    def stop_look_thread():
        look_thread_running[0] = False

    start_look_thread()

    available_entities = []
    target_entities = minescript.entities(max_distance=100)

    for t_entity in target_entities:
        if target_type in t_entity.name:
            available_entities.append(t_entity)
    #get closest entity(not really working)
    px, py, pz = minescript.player_position()
    available_entities.sort(key=lambda e: (e.position[0]-px)**2 + (e.position[1]-py)**2 + (e.position[2]-pz)**2)

    for entity in available_entities:
        while True:
            entities = minescript.entities(max_distance=100)
            live = next((e for e in entities if e.uuid == entity.uuid), None)
            if not live or live.health <= 0:
                with look_lock:
                    current_look[0] = None
                break

            px, py, pz = minescript.player_position()
            ex, ey, ez = live.position
            dist_sq = (px - ex) ** 2 + (py - ey) ** 2 + (pz - ez) ** 2

            if dist_sq > 3:
                path_walk_to(live.position, distance=1.5)
                with look_lock:
                    current_look[0] = None
            else:
                yaw, pitch = calc_look(px, py, pz, ex, ey, ez)
                with look_lock:
                    current_look[0] = (yaw, pitch)

                minescript.player_press_attack(True)
                time.sleep(1 / random.uniform(8, 11))
                minescript.player_press_attack(False)

            time.sleep(0.05)
    with look_lock:
        current_look[0] = None
    stop_look_thread()

#mining loop
def mining():
    #make 2 block lists 1 always updating so that when we run out of blocks in the main one we have the other list to fall back on
    available_blocks, available_angles = [], []
    current_blocks, current_angles = [], []
    detector = Detection(BLOCK_LIST, 3, 25)
    def scan_blocks():
        nonlocal available_blocks, available_angles
        while True:
            px, py, pz = minescript.player_position()
            detector.origin = (px, py, pz)
            blocks, angles = detector.locate_blocks()
            if blocks: available_blocks[:] = blocks
            if angles: available_angles[:] = angles
            time.sleep(1.5)
    threading.Thread(target=scan_blocks, daemon=True).start()
    while True:
        if not current_blocks and available_blocks:
            current_blocks[:] = available_blocks[:]
            current_angles[:] = available_angles[:]
            render_blocks(current_blocks)
        while current_blocks:
            block = current_blocks.pop(0)
            angles_for_block = current_angles.pop(0)
            if minescript.get_block(*block) not in BLOCK_LIST:
                stop_rendering(remove_blocks=[block])
                continue
            yaw, pitch = angles_for_block[0]
            look(yaw, pitch)
            while minescript.get_block(*block) in BLOCK_LIST:
                minescript.player_press_attack(True)
                time.sleep(0.01)
            # Remove the block from the render immediately
            stop_rendering(remove_blocks=[block])
        time.sleep(0.05)
claim_commissions()
