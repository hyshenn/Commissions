from system.lib.java import eval_pyjinn_script

_render_script = None

def render_blocks(block_positions):
    global _render_script

    _render_script = eval_pyjinn_script("""
from system.pyj.minescript import *
from lib_ren import WorldRendering

WorldRenderEvents = JavaClass("net.fabricmc.fabric.api.client.rendering.v1.WorldRenderEvents")
WorldRenderEventsLast = JavaClass("net.fabricmc.fabric.api.client.rendering.v1.WorldRenderEvents$Last")
render_path = []
callback_ref = None

def set_path(new_path):
    global render_path
    render_path = new_path

def remove_blocks(blocks_to_remove):
    global render_path
    render_path = [b for b in render_path if b not in blocks_to_remove]

def on_world_render_last(context):
    for pos in render_path:
        x, y, z = pos
        WorldRendering.wireframe(
            context,
            (x, y, z, x+1, y+1, z+1),
            (0, 255, 255, 255),
            visible_through_blocks=True
        )

global callback_ref
if callback_ref is None:
    callback_ref = ManagedCallback(on_world_render_last)
    WorldRenderEvents.LAST.register(WorldRenderEventsLast(callback_ref))
""")
    _render_script.get("set_path")(block_positions)
    return _render_script

def stop_rendering(remove_blocks=None):
    global _render_script
    if not _render_script:
        return
    if remove_blocks is None:
        _render_script.get("set_path")([])
    else:
        _render_script.get("remove_blocks")(remove_blocks)
