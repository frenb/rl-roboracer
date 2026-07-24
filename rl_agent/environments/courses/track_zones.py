"""Classifies a car's world (x, z) position into a track "zone" - used to
break crash/traversal counts down by WHERE on the track they happen (added
2026-07-19 per the "how well do we do on easy vs hard corners" request).

Three zone types:
  * "easy_corner" - one of the loop's 4 rectangle corners (present even at
    curriculum stage 0, which has zero chicanes - these are the base loop
    turns).
  * "hard_corner" - a chicane bump on one of the 4 edges (only present once
    a curriculum stage configures chicanes_north/east/south/west > 0).
  * "straight"    - everything else (plain interior edge cells).

This is a direct Python port of TrackGenerator.cs's BuildLoopCells /
ChooseChicaneTriggerIndices / BuildEdgeWithChicanes cell-path algorithm,
rather than an independent geometric approximation - keeping both sides
walking the exact same deterministic cell path is simpler and far less
drift-prone than re-deriving corner/chicane world-space bounding regions
from scratch. See unity/Assets/Scripts/TrackGenerator.cs for the source of
truth this mirrors; if that file's algorithm changes, update the three
_build_*/_choose_* helpers below to match.
"""

# Must match unity/Assets/Scenes/generated-course.unity's serialized
# TrackGenerator field values (the scene overrides the C# script's own
# defaults) - see that file's TrackGenerator component. Only geometry that
# affects the CELL PATH matters here; cornerRadius/chicaneCornerRadius only
# affect where within a cell the goal arc/visual geometry sits, not which
# cells are corners vs chicanes vs straights, so they're intentionally not
# needed by this module.
DEFAULT_TILE_SIZE = 20.0
DEFAULT_LOOP_WIDTH_TILES = 16
DEFAULT_LOOP_HEIGHT_TILES = 7

_ZONE_MAP_CACHE = {}


def _choose_chicane_trigger_indices(count, length):
    """Direct port of TrackGenerator.cs ChooseChicaneTriggerIndices()."""
    if count <= 0:
        return []
    min_spacing = 4
    lo = 1
    hi = length - 4
    if hi < lo:
        return []
    span = hi - lo
    fit_count = count
    while fit_count > 1 and span < min_spacing * (fit_count - 1):
        fit_count -= 1
    if fit_count <= 0:
        return []
    result = []
    for k in range(fit_count):
        if fit_count == 1:
            idx = lo + span // 2
        else:
            idx = lo + round((k / (fit_count - 1)) * span)
        result.append(idx)
    return result


def _build_edge_with_chicanes(start, walk_dir, bump_dir, length, chicane_count):
    """Direct port of TrackGenerator.cs BuildEdgeWithChicanes().

    Returns a list of ((x, y), is_chicane) tuples for this edge's interior
    cells (excluding the rectangle corners at either end, same as Unity).
    """
    edge = []
    if length <= 0:
        return edge
    triggers = set(_choose_chicane_trigger_indices(chicane_count, length))
    i = 0
    while i < length:
        p = (start[0] + walk_dir[0] * i, start[1] + walk_dir[1] * i)
        if i in triggers and i + 2 < length:
            edge.append((p, True))
            edge.append(((p[0] + bump_dir[0], p[1] + bump_dir[1]), True))
            edge.append(((p[0] + bump_dir[0] + walk_dir[0],
                          p[1] + bump_dir[1] + walk_dir[1]), True))
            edge.append(((p[0] + bump_dir[0] + walk_dir[0] * 2,
                          p[1] + bump_dir[1] + walk_dir[1] * 2), True))
            edge.append(((p[0] + walk_dir[0] * 2, p[1] + walk_dir[1] * 2), True))
            i += 3
        else:
            edge.append((p, False))
            i += 1
    return edge


def _build_cell_zone_map(loop_width, loop_height, ch_n, ch_e, ch_s, ch_w):
    """Direct port of TrackGenerator.cs BuildLoopCells(), annotated with a
    zone label per cell instead of just the raw path order.

    Returns dict {(x, y): "easy_corner" | "hard_corner" | "straight"}.
    """
    w = max(3, int(loop_width))
    h = max(3, int(loop_height))
    zones = {}

    def mark(cell, zone):
        # First writer wins - shouldn't matter in practice since
        # ChooseChicaneTriggerIndices guarantees >=1 buffer cell at each
        # edge/corner boundary, but corners are authoritative regardless.
        zones.setdefault(cell, zone)

    # Bottom (SOUTH) edge, W->E.
    mark((0, 0), "easy_corner")
    for cell, is_chicane in _build_edge_with_chicanes(
            (1, 0), (1, 0), (0, -1), w - 2, ch_s):
        mark(cell, "hard_corner" if is_chicane else "straight")
    mark((w - 1, 0), "easy_corner")

    # Right (EAST) edge, S->N.
    for cell, is_chicane in _build_edge_with_chicanes(
            (w - 1, 1), (0, 1), (1, 0), h - 2, ch_e):
        mark(cell, "hard_corner" if is_chicane else "straight")
    mark((w - 1, h - 1), "easy_corner")

    # Top (NORTH) edge, E->W.
    for cell, is_chicane in _build_edge_with_chicanes(
            (w - 2, h - 1), (-1, 0), (0, 1), w - 2, ch_n):
        mark(cell, "hard_corner" if is_chicane else "straight")

    # Left (WEST) edge, N->S.
    mark((0, h - 1), "easy_corner")
    for cell, is_chicane in _build_edge_with_chicanes(
            (0, h - 2), (0, -1), (-1, 0), h - 2, ch_w):
        mark(cell, "hard_corner" if is_chicane else "straight")

    return zones


def classify_zone(x, z, chicanes_north=0, chicanes_east=0, chicanes_south=0,
                   chicanes_west=0, tile_size=DEFAULT_TILE_SIZE,
                   loop_width_tiles=DEFAULT_LOOP_WIDTH_TILES,
                   loop_height_tiles=DEFAULT_LOOP_HEIGHT_TILES):
    """Classifies a world (x, z) position (car location_x/location_z) as
    'easy_corner', 'hard_corner', or 'straight'.

    Snaps to the nearest tileSize grid cell - safe since the car stays
    within roughly +/-roadHalfWidth of the intended cell centre, well
    under tile_size/2 (20m tiles vs ~5.5-7m half-width corridor). The
    zone map is built once per distinct (chicane counts, dimensions)
    combination and cached, since it's re-derived from scratch each call
    otherwise and this gets called on every crash/goal-reached event.

    Returns 'unknown' if x/z can't be read as floats - never raises,
    since callers use this for best-effort stats, not control flow.
    """
    try:
        cx = round(float(x) / tile_size)
        cz = round(float(z) / tile_size)
    except (TypeError, ValueError):
        return "unknown"
    key = (int(chicanes_north), int(chicanes_east), int(chicanes_south),
           int(chicanes_west), int(loop_width_tiles), int(loop_height_tiles),
           float(tile_size))
    zone_map = _ZONE_MAP_CACHE.get(key)
    if zone_map is None:
        zone_map = _build_cell_zone_map(
            loop_width_tiles, loop_height_tiles,
            chicanes_north, chicanes_east, chicanes_south, chicanes_west)
        _ZONE_MAP_CACHE[key] = zone_map
    return zone_map.get((cx, cz), "straight")
