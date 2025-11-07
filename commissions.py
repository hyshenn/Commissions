from system.lib import minescript
from system.lib.java import JavaClass
from detector import Detection
from rotation import look
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

AREA_LIST = {
    (-88, 147, -14): "Rampart's Quarry",
    (-126, 172, -76): "Upper Mines",
    (52, 198, -25): "Lava Springs",
    (149, 150, 30): "Royal Mines",
    (25, 129, 33): "Cliffside Veins",
    (-7, 128, 160): "glacite",
    (-135, 144, 141): "goblin"
}

EATHER_WARPS = {
    "Lava Springs": [(31, 196, -10)],
    "Upper Mines": [(-33, 174, -31), (-66, 222, -50), (-114, 196, -30)]
}

EMISSARY = (42, 135, 22)

commissions_list = []
stop_threads = threading.Event()

def commissions_helper_thread():
    global commissions_list
    while True:
        fetched = get_commissions()
        if fetched:
            commissions_list[:] = fetched
        time.sleep(2)

def path_helper(path):
    return [(x, y-1, z) for x, y, z in path]

def return_to_emissary():
    player_pos = tuple(minescript.player_position())
    path = path_find(player_pos, EMISSARY)
    render_blocks(path_helper(path))
    path_walk_to(path=path)
    stop_rendering()

def extract_percent(commission):
    if "%" in commission:
        try:
            return float(commission.strip().rstrip("%").split()[-1])
        except ValueError:
            return 0.0
    return 0.0

def get_commissions():
    connection = Minecraft.getInstance().getConnection()
    if not connection:
        return []
    players = connection.getOnlinePlayers()
    if not players:
        return []
    current_commissions = []
    try:
        for player in players.toArray():
            display = player.getTabListDisplayName()
            if display:
                text = display.getString()
                if (":" in text and "%" in text and "Raid" not in text) or "DONE" in text.upper():
                    current_commissions.append(text)
    except:
        pass
    return current_commissions

def travel_to_commission(area_name):
    if not area_name:
        return
    def warp(area_name):
        positions = EATHER_WARPS.get(area_name, [])
        if not positions:
            return
        render_blocks(positions)
        minescript.player_inventory_select_slot(8)
        for wx, wy, wz in positions:
            minescript.player_press_sneak(True)
            time.sleep(0.2)
            px, py, pz = minescript.player_position()
            rays = raycast_block_subregions((px, py+1.52, pz), (wx, wy, wz), max_dist=500, rays_per_axis=4)
            if rays:
                yaw, pitch = rays[0]
                look(yaw, pitch)
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
            path_walk_to(path=path)
            stop_rendering()
            time.sleep(1)
        if go_to_area:
            for pos, name in AREA_LIST.items():
                if name == area_name:
                    path = path_find(minescript.player_position(), pos)
                    render_blocks(path_helper(path))
                    path_walk_to(path=path)
                    stop_rendering()
                    return
    if area_name == "Lava Springs":
        go(area_name, pre_points=[(4, 147, -29)], go_to_area=False)
        warp(area_name)
        go(area_name)
    elif area_name == "Upper Mines":
        go(area_name, pre_points=[(-2, 147, -19)], go_to_area=False)
        warp(area_name)
        go(area_name)
    elif area_name == "Cliffside Veins":
        go(area_name, pre_points=[(38, 136, 18)])
    elif area_name == "Royal Mines":
        go(area_name, pre_points=[(47, 136, 19)])
    elif area_name == "Rampart's Quarry":
        go(area_name)
    elif area_name == "glacite":
        minescript.player_inventory_select_slot(4)
        go(area_name, pre_points=[(61, 135, 29), (0, 128, 66)], go_to_area=False)
        go(area_name)
    elif area_name == "goblin":
        minescript.player_inventory_select_slot(1)
        go(area_name, pre_points=[(61, 135, 29), (0, 128, 66), (-20, 128, 163)], go_to_area=False)
        go(area_name)

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

def kill_commission():
    commission_list_lower = [c.lower() for c in commissions_list]
    target_type = next(
        (t for c in commission_list_lower for t in ["glacite", "goblin"] if t in c.strip()),
        None
    )
    if not target_type:
        return
    if target_type == "glacite":
        target_type = "Glacite Walker"
    elif target_type == "goblin":
        target_type = "Goblin"
    look_lock = threading.Lock()
    current_look = [None]
    look_thread = [None]
    look_thread_running = [False]
    def calc_look(px, py, pz, ex, ey, ez):
        py += 1.62
        dx, dy, dz = ex - px, ey - py, ez - pz
        dist_xz = math.sqrt(dx*dx + dz*dz)
        yaw = math.degrees(math.atan2(-dx, dz))
        pitch = math.degrees(-math.atan2(dy, dist_xz))
        return yaw, pitch
    def thread_look():
        look_thread_running[0] = True
        while look_thread_running[0] and not stop_threads.is_set():
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
        if target_type.lower() in t_entity.name.lower():
            available_entities.append(t_entity)
    if not available_entities:
        return
    px, py, pz = minescript.player_position()
    available_entities.sort(key=lambda e: (e.position[0]-px)**2 + (e.position[1]-py)**2 + (e.position[2]-pz)**2)
    for entity in available_entities:
        while not stop_threads.is_set():
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

def mining():
    minescript.player_inventory_select_slot(4)
    available_blocks, available_angles = [], []
    current_blocks, current_angles = [], []
    detector = Detection(BLOCK_LIST, 3, 25)
    def scan_blocks():
        nonlocal available_blocks, available_angles
        while not stop_threads.is_set():
            px, py, pz = minescript.player_position()
            detector.origin = (px, py, pz)
            blocks, angles = detector.locate_blocks()
            if blocks: available_blocks[:] = blocks
            if angles: available_angles[:] = angles
            time.sleep(1.5)
    threading.Thread(target=scan_blocks, daemon=True).start()
    while not stop_threads.is_set():
        if not current_blocks and available_blocks:
            current_blocks[:] = available_blocks[:]
            current_angles[:] = available_angles[:]
            render_blocks(current_blocks)
        while current_blocks:
            if stop_threads.is_set():
                stop_rendering()
                return
            block = current_blocks.pop(0)
            angles_for_block = current_angles.pop(0)
            if minescript.get_block(*block) not in BLOCK_LIST:
                stop_rendering(remove_blocks=[block])
                continue
            yaw, pitch = angles_for_block[0]
            look(yaw, pitch)
            while minescript.get_block(*block) in BLOCK_LIST:
                if stop_threads.is_set():
                    stop_rendering(remove_blocks=[block])
                    return
                minescript.player_press_attack(True)
                time.sleep(0.01)
            stop_rendering(remove_blocks=[block])
        time.sleep(0.05)

def threaded_monitor(key, percent_container):
    while not stop_threads.is_set() and percent_container[0] != "DONE":
        updated = get_commissions()
        matching_comm = next((c for c in updated if key.lower() in c.lower()), None)
        if matching_comm:
            try:
                value = float(matching_comm.strip().rstrip("%").split()[-1])
                percent_container[0] = "DONE" if value >= 100 else value
            except ValueError:
                if "DONE" in matching_comm.upper():
                    percent_container[0] = "DONE"
                else:
                    percent_container[0] = 0.0
        time.sleep(1)

if __name__ == "__main__":
    threading.Thread(target=commissions_helper_thread, daemon=True).start()
    while True:
        if not commissions_list:
            minescript.echo("Waiting for commissions...")
            time.sleep(2)
            continue
        for commission in commissions_list:
            commission_lower = commission.lower()
            if any(k in commission_lower for k in ["mithril", "titanium"]):
                commission_type = "Miner"
                key = "mithril" if "mithril" in commission_lower else "titanium"
            elif any(k in commission_lower for k in ["goblin", "glacite"]):
                commission_type = "Slayer"
                key = "goblin" if "goblin" in commission_lower else "glacite"
            else:
                minescript.echo(f"Skipping unknown commission: {commission}")
                continue
            area = next((a for a in AREA_LIST.values() if a.lower() in commission_lower), None)
            travel_to_commission(area)
            percent_container = [extract_percent(commission)]
            stop_threads.clear()
            threading.Thread(target=lambda: threaded_monitor(key, percent_container), daemon=True).start()
            minescript.echo(f"Starting {commission_type} commission: {commission} ({percent_container[0]})")
            time.sleep(2.5)
            if commission_type == "Miner":
                threading.Thread(target=mining, daemon=True).start()
            elif commission_type == "Slayer":
                threading.Thread(target=kill_commission, daemon=True).start()
            while percent_container[0] != "DONE":
                time.sleep(0.1)
            stop_threads.set()
            stop_rendering()
            minescript.player_press_attack(False)
            time.sleep(1)
            minescript.execute("/warp forge")
            time.sleep(5)
            return_to_emissary()
            time.sleep(1)
            px, py, pz = minescript.player_position()
            ex, ey, ez = 42.5, 135.5, 22.5
            dx, dy, dz = ex - px, ey - (py + 1.62), ez - pz
            dist_xz = math.sqrt(dx*dx + dz*dz)
            yaw = math.degrees(math.atan2(-dx, dz))
            pitch = math.degrees(-math.atan2(dy, dist_xz))
            look(yaw, pitch)
            minescript.player_press_use(True)
            minescript.player_press_use(False)
            time.sleep(1)
            claim_commissions()
            minescript.echo(f"Finished {commission_type} commission: {commission} (DONE)")
        time.sleep(1)
