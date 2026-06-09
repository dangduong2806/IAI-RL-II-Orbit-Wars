"""
Orbit Wars - V6: Hybrid Champion Agent
Combines V4's advanced simulation with ver_demo's SPRT-proven heuristics (LB 1039.2)
Includes Comets Integration and Eclipse Avoidance.

Key improvements from ver_demo:
  1. ROI-based scoring: (prod * HORIZON - needed) / arrival_time
  2. AGRO send-full-available: send entire available pool, not just needed
  3. Launch-safety multiplier: penalize front-line planet launches
  4. Score floor 2.4: reject low-value targets
  5. Enemy denial bonus ×2.0: prioritize capturing enemy planets
  6. Sun-doomed fleet skip: ignore enemy fleets that will die in the sun
  7. Planet-doomed skip: don't waste reserves on unsavable planets
  8. Greedy target locking: global sort + one-launch-per-planet, one-attack-per-target
  9. Comets Integration: Parse and capture high-value comet planets
  10. Eclipse Avoidance: Sweep paths to ensure we don't accidentally hit other planets

Retained from V4:
  - predict_enemy_moves() phantom simulation
  - Event-based ships_needed_at_time() accounting for multiple fleets
  - min_ships_to_leave() defense calculation
  - Cooperative defense for high-prod planets
"""
import os
os.environ['KAGGLE_ENVELOPES'] = '0'

import math

SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
LOG_1000 = 6.907755278982137

# === SPRT-tuned constants from ver_demo (LB 1039.2) ===
HORIZON = 30              # Production lookahead for target value
SCORE_FLOOR = 2.4         # Reject candidates below this score
ENEMY_DENIAL_BONUS = 2.0  # Multiplier for enemy-owned targets
LAUNCH_SAFETY_FLOOR = 0.4 # Min multiplier for front-line planets
LAUNCH_SAFETY_SCALE = 30.0 # Distance at which safety reaches 1.0
REINFORCE_MIN_PROD = 2    # Only reinforce planets with prod >= this

PROFILE_NAMES = (
    "balanced",
    "aggressive",
    "economy",
    "defensive",
    "denial",
    "patient",
)

PROFILES = {
    "balanced": {
        "horizon": HORIZON,
        "score_floor": SCORE_FLOOR,
        "enemy_denial_bonus": ENEMY_DENIAL_BONUS,
        "launch_safety_floor": LAUNCH_SAFETY_FLOOR,
        "launch_safety_scale": LAUNCH_SAFETY_SCALE,
        "reinforce_min_prod": REINFORCE_MIN_PROD,
    },
    "aggressive": {
        "horizon": 34,
        "score_floor": 1.8,
        "enemy_denial_bonus": 2.5,
        "launch_safety_floor": 0.60,
        "launch_safety_scale": 24.0,
        "reinforce_min_prod": 3,
    },
    "economy": {
        "horizon": 42,
        "score_floor": 2.0,
        "enemy_denial_bonus": 1.4,
        "launch_safety_floor": 0.35,
        "launch_safety_scale": 34.0,
        "reinforce_min_prod": 2,
    },
    "defensive": {
        "horizon": 24,
        "score_floor": 3.0,
        "enemy_denial_bonus": 1.8,
        "launch_safety_floor": 0.22,
        "launch_safety_scale": 42.0,
        "reinforce_min_prod": 1,
    },
    "denial": {
        "horizon": 30,
        "score_floor": 2.1,
        "enemy_denial_bonus": 3.0,
        "launch_safety_floor": 0.45,
        "launch_safety_scale": 30.0,
        "reinforce_min_prod": 2,
    },
    "patient": {
        "horizon": 26,
        "score_floor": 3.4,
        "enemy_denial_bonus": 2.0,
        "launch_safety_floor": 0.32,
        "launch_safety_scale": 38.0,
        "reinforce_min_prod": 1,
    },
}


def resolve_profile(profile=None):
    if profile is None:
        return PROFILES["balanced"]
    if isinstance(profile, str):
        return PROFILES.get(profile, PROFILES["balanced"])
    merged = dict(PROFILES["balanced"])
    merged.update(profile)
    return merged


def fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    return 1.0 + 5.0 * (math.log(max(ships, 1)) / LOG_1000) ** 1.5


def travel_time(x1: float, y1: float, x2: float, y2: float, ships: int) -> float:
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist / fleet_speed(ships) if ships > 0 else 999.0


def line_seg_min_dist(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(x1 - px, y1 - py)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(x1 + t * dx - px, y1 + t * dy - py)


def path_crosses_sun(x1: float, y1: float, x2: float, y2: float, margin: float = 1.5) -> bool:
    return line_seg_min_dist(x1, y1, x2, y2, SUN_X, SUN_Y) < SUN_RADIUS + margin


def local_turn_angle(src, tgt) -> float:
    """Rotation-invariant tie-break angle from a planet's outward radial axis."""
    radial_x = src['x'] - SUN_X
    radial_y = src['y'] - SUN_Y
    target_x = tgt['x'] - src['x']
    target_y = tgt['y'] - src['y']
    cross = radial_x * target_y - radial_y * target_x
    dot = radial_x * target_x + radial_y * target_y
    return math.atan2(cross, dot)


def player_axis(planets):
    """Return an equivariant orientation vector for geometry-only tie breaks."""
    axis_x = sum(p['x'] - SUN_X for p in planets)
    axis_y = sum(p['y'] - SUN_Y for p in planets)
    if math.hypot(axis_x, axis_y) < 1e-9:
        return 1.0, 0.0
    return axis_x, axis_y


def angle_from_axis(axis, planet) -> float:
    axis_x, axis_y = axis
    point_x = planet['x'] - SUN_X
    point_y = planet['y'] - SUN_Y
    cross = axis_x * point_y - axis_y * point_x
    dot = axis_x * point_x + axis_y * point_y
    return math.atan2(cross, dot)


def planet_geometry_key(planet, axis):
    return (
        round(angle_from_axis(axis, planet), 6),
        round(math.hypot(planet['x'] - SUN_X, planet['y'] - SUN_Y), 6),
        -round(planet['prod'], 6),
        round(planet['ships'], 6),
    )


def predict_orbit(x: float, y: float, omega: float, dt: float):
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)


def solve_intercept(fx: float, fy: float, tx: float, ty: float, orbiting: bool, omega: float, ships: int, iterations: int = 25):
    if not orbiting:
        t = travel_time(fx, fy, tx, ty, ships)
        return tx, ty, t
    theta = math.atan2(ty - SUN_Y, tx - SUN_X)
    r = math.hypot(tx - SUN_X, ty - SUN_Y)
    t = travel_time(fx, fy, tx, ty, ships)
    ix, iy = tx, ty
    for _ in range(iterations):
        ix = SUN_X + r * math.cos(theta + omega * t)
        iy = SUN_Y + r * math.sin(theta + omega * t)
        t2 = travel_time(fx, fy, ix, iy, ships)
        if abs(t2 - t) < 0.05:
            break
        t = t2
    return ix, iy, t


def solve_intercept_safe(fx, fy, target_p, omega, ships):
    if target_p['is_orb'] and 'orb_r' in target_p:
        r = target_p['orb_r']
        theta = target_p['orb_theta']
        t = travel_time(fx, fy, target_p['x'], target_p['y'], ships)
        ix, iy = target_p['x'], target_p['y']
        for _ in range(25):
            ix = SUN_X + r * math.cos(theta + omega * t)
            iy = SUN_Y + r * math.sin(theta + omega * t)
            t2 = travel_time(fx, fy, ix, iy, ships)
            if abs(t2 - t) < 0.05:
                break
            t = t2
    else:
        ix, iy, t = solve_intercept(fx, fy, target_p['x'], target_p['y'], target_p['is_orb'], omega, ships)
        
    if path_crosses_sun(fx, fy, ix, iy, margin=1.5):
        return None
    return ix, iy, t


def solve_intercept_comet(from_x, from_y, comet_info_tuple, ship_speed, speed_tol_frac=0.1):
    path, idx, remaining = comet_info_tuple
    if ship_speed <= 0 or remaining <= 1:
        return None
    
    tol = speed_tol_frac * ship_speed
    for dt in range(1, remaining):
        target_pos = path[idx + dt]
        tx, ty = float(target_pos[0]), float(target_pos[1])
        
        dist = math.hypot(tx - from_x, ty - from_y)
        required_speed = dist / dt
        
        if abs(required_speed - ship_speed) > tol:
            continue
            
        if path_crosses_sun(from_x, from_y, tx, ty, margin=1.5):
            continue
            
        angle = math.atan2(ty - from_y, tx - from_x)
        return tx, ty, float(dt), angle
        
    return None


def get_comet_info(obs):
    comets_data = obs.get('comets', []) if isinstance(obs, dict) else getattr(obs, 'comets', [])
    comet_ids = set(obs.get('comet_planet_ids', []) if isinstance(obs, dict) else getattr(obs, 'comet_planet_ids', []))
    
    comet_info = {}
    for group in comets_data:
        idx = int(group.get('path_index', 0) if isinstance(group, dict) else getattr(group, 'path_index', 0))
        pids = group.get('planet_ids', []) if isinstance(group, dict) else getattr(group, 'planet_ids', [])
        paths = group.get('paths', []) if isinstance(group, dict) else getattr(group, 'paths', [])
        
        for i, pid in enumerate(pids):
            if i >= len(paths):
                continue
            path = paths[i]
            remaining = max(0, len(path) - idx)
            comet_info[int(pid)] = (path, idx, remaining)
    return comet_ids, comet_info


def check_path_clear(fx, fy, angle, speed, arrival, target_id, planets, omega):
    """Ensure fleet doesn't hit another planet before reaching its target."""
    max_steps = int(math.ceil(arrival))
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    
    for t_step in range(1, max_steps + 1):
        fl_x = fx + vx * t_step
        fl_y = fy + vy * t_step
        
        for p in planets.values():
            if p['id'] == target_id:
                continue
            
            if p['is_comet']:
                comet_data = p.get('comet_data')
                if not comet_data:
                    continue
                path, idx, remaining = comet_data
                if t_step < remaining:
                    pos = path[idx + t_step]
                    px, py = float(pos[0]), float(pos[1])
                else:
                    continue
            elif p['is_orb']:
                r = p['orb_r']
                theta = p['orb_theta']
                px = SUN_X + r * math.cos(theta + omega * t_step)
                py = SUN_Y + r * math.sin(theta + omega * t_step)
            else:
                px, py = p['x'], p['y']
                
            dist_sq = (fl_x - px)**2 + (fl_y - py)**2
            if dist_sq <= p['radius']**2:
                return False  # Path is blocked
    return True


def predict_fleet_targets(fleets, planets, omega):
    """Predict which planet each fleet is heading toward."""
    fleet_targets = {}
    for f in fleets.values():
        fx, fy = f['x'], f['y']
        angle = f['angle']
        speed = fleet_speed(int(f['ships']))
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        
        best_target = None
        best_time = float('inf')
        
        for p in planets.values():
            if p['id'] == f['from']: continue
            
            if p['is_comet']:
                comet_data = p.get('comet_data')
                if not comet_data:
                    continue
                path, idx, remaining = comet_data
                for t in range(1, min(150, remaining)):
                    if t >= best_time: break
                    pos = path[idx + t]
                    px, py = float(pos[0]), float(pos[1])
                    fl_x = fx + vx * t
                    fl_y = fy + vy * t
                    if (fl_x - px)**2 + (fl_y - py)**2 <= p['radius']**2:
                        best_time = t
                        best_target = p['id']
                        break
            elif not p['is_orb']:
                dx = p['x'] - fx
                dy = p['y'] - fy
                v_mag_sq = vx*vx + vy*vy
                if v_mag_sq == 0: continue
                t = (dx * vx + dy * vy) / v_mag_sq
                if t < 0: continue
                cx = fx + vx * t
                cy = fy + vy * t
                dist_sq = (cx - p['x'])**2 + (cy - p['y'])**2
                if dist_sq <= p['radius']**2:
                    dt_to_surface = math.sqrt(max(0, p['radius']**2 - dist_sq)) / speed
                    hit_time = t - dt_to_surface
                    if 0 <= hit_time < best_time:
                        best_time = hit_time
                        best_target = p['id']
            else:
                r = p.get('orb_r', math.hypot(p['x'] - SUN_X, p['y'] - SUN_Y))
                theta = p.get('orb_theta', math.atan2(p['y'] - SUN_Y, p['x'] - SUN_X))
                for t in range(1, 150):
                    if t >= best_time: break
                    px = SUN_X + r * math.cos(theta + omega * t)
                    py = SUN_Y + r * math.sin(theta + omega * t)
                    fl_x = fx + vx * t
                    fl_y = fy + vy * t
                    if (fl_x - px)**2 + (fl_y - py)**2 <= p['radius']**2:
                        best_time = t
                        best_target = p['id']
                        break
        
        if best_target is not None:
            fleet_targets[f['id']] = (best_target, best_time)
            
    return fleet_targets


def ships_needed_at_time(p, events, target_time, my_player):
    """Simulate planet state accounting for multiple arriving fleets."""
    events_before = [e for e in events if e['eta'] <= target_time]
    
    grouped = {}
    for e in events_before:
        if e['eta'] not in grouped:
            grouped[e['eta']] = {'friendly': 0, 'enemy': 0}
        if e['owner'] == my_player:
            grouped[e['eta']]['friendly'] += e['ships']
        else:
            grouped[e['eta']]['enemy'] += e['ships']
            
    sorted_etas = sorted(grouped.keys())
    
    current_time = 0
    ships = p['ships']
    owner = p['owner']
    
    for eta in sorted_etas:
        dt = eta - current_time
        if owner != -1:
            ships += p['prod'] * dt
        current_time = eta
        
        f_ships = grouped[eta]['friendly']
        e_ships = grouped[eta]['enemy']
        
        if owner == my_player:
            ships += f_ships
            ships -= e_ships
            if ships < 0:
                owner = -2
                ships = -ships
        else:
            if owner == -1:
                if f_ships > e_ships + ships:
                    owner = my_player
                    ships = f_ships - e_ships - ships
                elif e_ships > f_ships + ships:
                    owner = -2
                    ships = e_ships - f_ships - ships
                else:
                    ships = ships - f_ships - e_ships
                    if ships < 0: ships = 0
            else:
                ships += e_ships
                ships -= f_ships
                if ships < 0:
                    owner = my_player
                    ships = -ships
                    
    dt = target_time - current_time
    if owner != -1:
        ships += p['prod'] * dt
        
    if owner == my_player:
        return 0
    else:
        return int(ships) + 1


def min_ships_to_leave(p, events, my_player, max_time=150):
    """Calculate minimum garrison to survive all incoming threats."""
    events_before = [e for e in events if e['eta'] <= max_time]
    
    grouped = {}
    for e in events_before:
        if e['eta'] not in grouped:
            grouped[e['eta']] = {'friendly': 0, 'enemy': 0}
        if e['owner'] == my_player:
            grouped[e['eta']]['friendly'] += e['ships']
        else:
            grouped[e['eta']]['enemy'] += e['ships']
            
    sorted_etas = sorted(grouped.keys())
    
    current_time = 0
    balance = 0
    min_balance = 0
    
    for eta in sorted_etas:
        dt = eta - current_time
        balance += p['prod'] * dt
        current_time = eta
        
        f_ships = grouped[eta]['friendly']
        e_ships = grouped[eta]['enemy']
        
        balance += f_ships
        balance -= e_ships
        
        if balance < min_balance:
            min_balance = balance
            
    if min_balance >= 0:
        return 0
    return int(-min_balance) + 1


def predict_enemy_moves(enemy_planets, all_planets, planet_events, omega, my_player, axis):
    """Level-2 Simulation: Predict likely enemy attacks and populate phantom events."""
    for src in sorted(enemy_planets, key=lambda p: planet_geometry_key(p, axis)):
        if src['ships'] < 25:
            continue
            
        best_tgt = None
        best_key = None
        best_tt = 0
        best_needed = 0
        
        speed = fleet_speed(src['ships'])
        
        for t in sorted(all_planets.values(), key=lambda p: planet_geometry_key(p, axis)):
            if t['id'] == src['id']:
                continue
            if t['owner'] == src['owner']:
                continue
                
            if t['is_comet']:
                if not t['comet_data']: continue
                sol = solve_intercept_comet(src['x'], src['y'], t['comet_data'], speed)
                if sol is None: continue
                ix, iy, tt, angle = sol
            else:
                intercept = solve_intercept_safe(src['x'], src['y'], t, omega, int(src['ships']))
                if intercept is None:
                    continue
                ix, iy, tt = intercept
                angle = math.atan2(iy - src['y'], ix - src['x'])
            
            # Simple eclipse check for phantom moves
            if not check_path_clear(src['x'], src['y'], angle, speed, tt, t['id'], all_planets, omega):
                continue
            
            score = t['prod'] * 20 - tt * 2.5
            if t['owner'] == my_player:
                score += 35
            elif t['owner'] == -1:
                score += 15
                
            needed = ships_needed_at_time(t, planet_events[t['id']], tt, src['owner'])
            if needed >= src['ships']:
                continue
                
            score -= needed * 0.5
            
            target_key = (
                -round(score, 6),
                round(tt, 6),
                round(local_turn_angle(src, t), 6),
                round(angle_from_axis(axis, t), 6),
            )
            if best_key is None or target_key < best_key:
                best_key = target_key
                best_tgt = t
                best_tt = tt
                best_needed = needed
                
        if best_tgt:
            send = min(int(src['ships']), best_needed + 5)
            planet_events[best_tgt['id']].append({
                'eta': best_tt,
                'owner': src['owner'],
                'ships': send,
                'phantom': True
            })


def compute_launch_safety(
    my_planets,
    enemy_planets,
    fleets,
    player,
    launch_safety_floor=LAUNCH_SAFETY_FLOOR,
    launch_safety_scale=LAUNCH_SAFETY_SCALE,
):
    """
    Launch-safety multiplier (ver_demo ingredient #7):
    Threats = enemy-owned planets + enemy fleet CURRENT POSITIONS.
    Front-line planets get penalized; rear-line planets fire freely.
    """
    threats = []
    for e in enemy_planets:
        threats.append((e['x'], e['y']))
    for f in fleets.values():
        if f['owner'] != player:
            threats.append((f['x'], f['y']))
    
    safety = {}
    for p in my_planets:
        if threats:
            min_d = min(math.hypot(p['x'] - tx, p['y'] - ty) for tx, ty in threats)
            safety[p['id']] = max(launch_safety_floor, min(1.0, min_d / launch_safety_scale))
        else:
            safety[p['id']] = 1.0
    return safety


def fleet_path_hits_sun(fx, fy, angle, speed, steps):
    """Check if a fleet's straight-line path crosses the sun within given steps."""
    ex = fx + speed * math.cos(angle) * steps
    ey = fy + speed * math.sin(angle) * steps
    return path_crosses_sun(fx, fy, ex, ey, margin=0.0)


def agent(obs, config=None, profile=None):
    params = resolve_profile(profile)
    if isinstance(obs, dict):
        player = obs.get('player', 0)
        planets_data = obs.get('planets', [])
        fleets_data = obs.get('fleets', [])
        step = obs.get('step', 0)
        omega = obs.get('angular_velocity', 0.03)
    else:
        player = getattr(obs, 'player', 0)
        planets_data = getattr(obs, 'planets', [])
        fleets_data = getattr(obs, 'fleets', [])
        step = getattr(obs, 'step', 0)
        omega = getattr(obs, 'angular_velocity', 0.03)

    comet_ids, comet_paths = get_comet_info(obs)

    planets = {}
    for p in planets_data:
        pid, owner, x, y, radius, ships, prod = p[:7]
        r = math.hypot(x - SUN_X, y - SUN_Y)
        is_comet = int(pid) in comet_ids
        planets[pid] = {
            'id': pid, 'owner': owner, 'x': x, 'y': y,
            'radius': radius, 'ships': float(ships), 'prod': float(prod),
            'is_orb': (r + radius) < 48.0 and not is_comet,
            'is_comet': is_comet,
            'comet_data': comet_paths.get(int(pid)),
            'orb_r': r,
            'orb_theta': math.atan2(y - SUN_Y, x - SUN_X)
        }

    fleets = {}
    for f in fleets_data:
        fleets[f[0]] = {
            'id': f[0], 'owner': f[1], 'x': f[2], 'y': f[3],
            'angle': f[4], 'from': f[5], 'ships': float(f[6])
        }

    my = [p for p in planets.values() if p['owner'] == player]
    if not my:
        return []

    enemy = [p for p in planets.values() if p['owner'] != player and p['owner'] != -1]
    neutrals = [p for p in planets.values() if p['owner'] == -1]
    axis = player_axis(my)

    # ========================================================================
    # PHASE 1: Fleet target prediction + event building
    # ========================================================================
    fleet_targets = predict_fleet_targets(fleets, planets, omega)
    
    planet_events = {p_id: [] for p_id in planets}
    for f_id, (tgt_id, eta) in fleet_targets.items():
        if tgt_id in planet_events:
            f = fleets[f_id]
            
            # --- Sun-doomed fleet skip (ver_demo ingredient #1) ---
            speed = fleet_speed(int(f['ships']))
            if fleet_path_hits_sun(f['x'], f['y'], f['angle'], speed, eta):
                continue
            
            # --- Eclipse skip for existing fleets (avoid false threats) ---
            if not check_path_clear(f['x'], f['y'], f['angle'], speed, eta, tgt_id, planets, omega):
                continue
            
            planet_events[tgt_id].append({
                'eta': eta,
                'owner': f['owner'],
                'ships': f['ships']
            })

    # Level-2 Simulation: Predict enemy attacks
    predict_enemy_moves(enemy, planets, planet_events, omega, player, axis)

    # ========================================================================
    # PHASE 2: Launch-safety computation (ver_demo ingredient #7)
    # ========================================================================
    launch_safety = compute_launch_safety(
        my,
        enemy,
        fleets,
        player,
        launch_safety_floor=params["launch_safety_floor"],
        launch_safety_scale=params["launch_safety_scale"],
    )

    # ========================================================================
    # PHASE 3: Cooperative Defense — reinforce high-prod planets under threat
    # ========================================================================
    reinforce_moves = []
    reinforced_targets = set()
    
    for target_p in my:
        if target_p['prod'] < params["reinforce_min_prod"]:
            continue
        
        needed_defense = min_ships_to_leave(target_p, planet_events[target_p['id']], player, max_time=150)
        
        # Planet-doomed skip
        if needed_defense <= target_p['ships']:
            continue
        
        gap = int(needed_defense - target_p['ships']) + 1
        if target_p['id'] in reinforced_targets:
            continue
        
        best_src = None
        best_angle = 0.0
        best_tt = float('inf')
        
        enemy_events = [e for e in planet_events[target_p['id']] if e['owner'] != player]
        if not enemy_events:
            continue
        earliest_threat_eta = min(e['eta'] for e in enemy_events)
        
        for src in my:
            if src['id'] == target_p['id']:
                continue
            
            src_defense = min_ships_to_leave(src, planet_events[src['id']], player, max_time=150)
            spare = src['ships'] - src_defense
            if spare < gap:
                continue
            
            speed = fleet_speed(gap)
            intercept = solve_intercept_safe(src['x'], src['y'], target_p, omega, gap)
            if intercept is None:
                continue
            ix, iy, tt = intercept
            angle = math.atan2(iy - src['y'], ix - src['x'])
            
            if tt >= earliest_threat_eta:
                continue
                
            if not check_path_clear(src['x'], src['y'], angle, speed, tt, target_p['id'], planets, omega):
                continue
            
            if tt < best_tt:
                best_tt = tt
                best_src = src
                best_angle = angle
        
        if best_src:
            reinforce_moves.append([best_src['id'], best_angle, gap])
            best_src['ships'] -= gap
            reinforced_targets.add(target_p['id'])
            planet_events[target_p['id']].append({
                'eta': best_tt,
                'owner': player,
                'ships': gap
            })

    # ========================================================================
    # PHASE 4: Offensive — ROI-based scoring with greedy target locking
    # ========================================================================
    all_candidates = []
    used_sources = {m[0] for m in reinforce_moves}
    targets = neutrals + enemy
    
    for src in my:
        if src['id'] in used_sources:
            continue
        if src['ships'] < 5:
            continue

        needed_for_defense = min_ships_to_leave(src, planet_events[src['id']], player, max_time=150)
        available = int(src['ships'] - needed_for_defense)
        
        if available < 5:
            continue

        for t in targets:
            if t['id'] == src['id']:
                continue
            
            speed = fleet_speed(available)
            
            if t['is_comet']:
                if not t['comet_data']:
                    continue
                sol = solve_intercept_comet(src['x'], src['y'], t['comet_data'], speed)
                if sol is None:
                    continue
                ix, iy, tt, angle = sol
            else:
                intercept = solve_intercept_safe(src['x'], src['y'], t, omega, available)
                if intercept is None:
                    continue
                ix, iy, tt = intercept
                angle = math.atan2(iy - src['y'], ix - src['x'])
            
            if tt <= 0 or tt > 500:
                continue
            
            # Eclipse Sweep
            if not check_path_clear(src['x'], src['y'], angle, speed, tt, t['id'], planets, omega):
                continue
            
            needed = ships_needed_at_time(t, planet_events[t['id']], tt, player)
            if needed <= 0 or needed > available:
                continue
            
            if t['is_comet']:
                rem_after = max(0, t['comet_data'][2] - 1 - int(tt))
                effective_horizon = min(params["horizon"], rem_after)
            else:
                effective_horizon = params["horizon"]
                
            score = (t['prod'] * effective_horizon - needed) / max(tt, 1.0)
            
            if t['owner'] != -1 and t['owner'] != player:
                score *= params["enemy_denial_bonus"]
            
            score *= launch_safety[src['id']]
            
            if score < params["score_floor"]:
                continue
            
            send = available
            
            all_candidates.append({
                'score': score,
                'src': src,
                'tgt': t,
                'intercept': (ix, iy, tt),
                'angle': angle,
                'send': send
            })
    
    # Greedy target locking
    all_candidates.sort(key=lambda c: c['score'], reverse=True)
    
    locked_targets = set()
    moves = list(reinforce_moves)
    
    for cand in all_candidates:
        src = cand['src']
        tgt = cand['tgt']
        
        if src['id'] in used_sources:
            continue
        if tgt['id'] in locked_targets:
            continue
        
        ix, iy, tt = cand['intercept']
        angle = cand['angle']
        send = cand['send']
        
        moves.append([src['id'], angle, send])
        used_sources.add(src['id'])
        locked_targets.add(tgt['id'])
        
        planet_events[tgt['id']].append({
            'eta': tt,
            'owner': player,
            'ships': send
        })
        src['ships'] -= send

    # ========================================================================
    # PHASE 5: Fallback — Supply line reinforcement for unused planets
    # ========================================================================
    if enemy:
        frontline_dists = {}
        for p in my:
            min_e_dist = float('inf')
            for e in enemy:
                d = math.hypot(e['x'] - p['x'], e['y'] - p['y'])
                if d < min_e_dist:
                    min_e_dist = d
            frontline_dists[p['id']] = min_e_dist
        
        for src in my:
            if src['id'] in used_sources:
                continue
            
            needed_for_defense = min_ships_to_leave(src, planet_events[src['id']], player, max_time=150)
            available = src['ships'] - needed_for_defense
            
            if available < 40:
                continue
            
            src_dist = frontline_dists.get(src['id'], float('inf'))
            best_reinforce = None
            best_reinforce_key = None
            
            speed = fleet_speed(int(available * 0.8))
            
            for friend in my:
                if friend['id'] == src['id']:
                    continue
                f_dist = frontline_dists.get(friend['id'], float('inf'))
                if f_dist < src_dist - 15:
                    intercept = solve_intercept_safe(src['x'], src['y'], friend, omega, int(available * 0.8))
                    if intercept is not None:
                        ix, iy, tt = intercept
                        angle = math.atan2(iy - src['y'], ix - src['x'])
                        
                        if not check_path_clear(src['x'], src['y'], angle, speed, tt, friend['id'], planets, omega):
                            continue
                            
                        reinforce_key = (
                            round(f_dist, 6),
                            round(tt, 6),
                            local_turn_angle(src, friend),
                        )
                        if best_reinforce_key is None or reinforce_key < best_reinforce_key:
                            best_reinforce_key = reinforce_key
                            best_reinforce = (friend, intercept, angle)
            
            if best_reinforce:
                friend, intercept, angle = best_reinforce
                ix, iy, tt = intercept
                send = int(available * 0.8)
                if send >= 10:
                    moves.append([src['id'], angle, send])
                    used_sources.add(src['id'])
                    planet_events[friend['id']].append({
                        'eta': tt,
                        'owner': player,
                        'ships': send
                    })
                    src['ships'] -= send

    return moves

if __name__ == '__main__':
    print("Orbit Wars V6 (Comets + Eclipse Avoidance) Loaded!")
