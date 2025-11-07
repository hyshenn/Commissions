from system.lib import minescript
from system.lib.java import JavaClass
from detector import Detection
from rotation import look #I preferred this one since i like it better, your choice tho
from ThetaStar import path_walk_to
from raycast import raycast_block_subregions
import threading, time, math

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
        for wx, wy, wz in EATHER_WARPS.get(area_name, []):
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

    def go(area_name, pre_points=(), go_to_area=True):
        for point in pre_points:
            path_walk_to(point)
            time.sleep(0.1)
        if go_to_area:
            for pos, name in AREA_LIST.items():
                if name == area_name:
                    path_walk_to(pos)
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

def kill_commission():
    commissions = get_commissions()
    target_type = next((t for c in commissions for t in ["Glacite Walker", "Goblin"] if t in c.strip()), None)
    target_type = "Glacite Walker"
    if not target_type:
        return
    available_entities = []
    target_entities = minescript.entities(max_distance=100)

    for t_entity in target_entities:
        if target_type in t_entity.name:
            available_entities.append(t_entity)

    for entity in available_entities:
        while entity.health > 0.1:
            path_walk_to(entity.position)

#mining loop
def mining():
    #make 2 block lists 1 always updating so that when we run out of blocks in the main one we have the other list to fall back on
    available_blocks, available_angles = [], []
    current_blocks, current_angles = [], []
    detector = Detection(BLOCK_LIST, 4, 25)
    #threaded scanner
    def scan_blocks():
        nonlocal available_blocks, available_angles
        while True:
            px, py, pz = minescript.player_position()
            detector.origin = (px, py, pz)
            blocks, angles = detector.locate_blocks()
            if blocks: available_blocks[:] = blocks
            if angles: available_angles[:] = angles
            time.sleep(0.5)
    threading.Thread(target=scan_blocks, daemon=True).start()
    #main loop
    while True:
        if not current_blocks and available_blocks:
            current_blocks[:] = available_blocks[:]
            current_angles[:] = available_angles[:]
        while current_blocks:
            block = current_blocks.pop(0)
            angles_for_block = current_angles.pop(0)
            if minescript.get_block(*block) not in BLOCK_LIST:
                continue
            yaw, pitch = angles_for_block[0]
            look(yaw, pitch)
            while minescript.get_block(*block) in BLOCK_LIST:
                minescript.player_press_attack(True)
        time.sleep(0.05)
kill_commission()
