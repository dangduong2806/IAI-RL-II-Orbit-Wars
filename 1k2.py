from __future__ import annotations
import dataclasses
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Sequence
import torch
from torch import Tensor

# Hằng số cấu hình toàn cục
DEFAULT_BATCH_SIZE: int = 1024
MAX_PLANET_COUNT: int = 64
MAX_FLEET_COUNT: int = 256
DEFAULT_PLAYERS: int = 2
PLAY_BOARD_SIZE: float = 100.0
PLAY_BOARD_CENTER: float = 50.0
SOLAR_SUN_RADIUS: float = 10.0
LIMIT_SHIP_SPEED: float = 6.0
ORBITAL_LIMIT_RADIUS: float = 50.0
OWNER_SELF: int = 0
OWNER_OPPONENT: int = 1
OWNER_NEUTRAL: int = 2
OWNER_DEAD: int = 3
LIBRARY_CAPACITY: int = 100000
COMET_EVENT_COUNT: int = 5
COMETS_PER_SPAWN: int = 4
COMET_MAX_PATH_LEN: int = 40
COMET_SPAWN_TURNS: tuple[int, ...] = (50, 150, 250, 350, 450)
COMET_COLLISION_RADIUS: float = 1.0
COMET_SHIP_PROD: float = 1.0
EARLY_STOP_MARGIN: float = 2.0
EARLY_STOP_STREAK_2P: int = 5
EARLY_STOP_STREAK_4P: int = 20
EARLY_STOP_PROD_WT_2P: float = 5.0
EARLY_STOP_SHIP_WT_2P: float = 1.0
EARLY_STOP_PROD_WT_4P: float = 1.0
EARLY_STOP_SHIP_WT_4P: float = 0.0
DEFAULT_MAX_EPISODE_STEPS: int = 500

def calculate_orbit_phase(obs_step: Tensor) -> Tensor:
    step_f = obs_step.float()
    return (step_f - (step_f > 0.0).to(step_f.dtype)).clamp(min=0.0)

LOG_SHIPS_BASE: float = float(torch.log(torch.tensor(1000.0)).item())
SPEED_TABLE_MAX_SHIPS: int = 400

def compute_fleet_velocity(ship_count: Tensor) -> Tensor:
    ratio = (torch.log(ship_count) / LOG_SHIPS_BASE).clamp(max=1.0)
    return 1.0 + (LIMIT_SHIP_SPEED - 1.0) * ratio.pow(1.5)

def generate_speed_lookup_table(max_ships: int) -> Tensor:
    idx = torch.arange(max_ships + 1, dtype=torch.float32).clamp(min=1.0)
    return compute_fleet_velocity(idx)

SPEED_LUT_GLOBAL: Tensor = generate_speed_lookup_table(SPEED_TABLE_MAX_SHIPS)
SPEED_LUT_DEVICES_CACHE: dict[tuple, Tensor] = {}

def get_cached_speed_table(device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (device, dtype)
    cached = SPEED_LUT_DEVICES_CACHE.get(key)
    if cached is None:
        cached = SPEED_LUT_GLOBAL.to(device=device, dtype=dtype)
        SPEED_LUT_DEVICES_CACHE[key] = cached
    return cached

def estimate_fleet_speed(ships: Tensor) -> Tensor:
    s = ships.clamp(min=1.0)
    s_lut = s.clamp(max=float(SPEED_TABLE_MAX_SHIPS))
    lo = torch.floor(s_lut).long()
    hi = torch.ceil(s_lut).long()
    frac = s_lut - lo.to(dtype=s.dtype)
    lut = get_cached_speed_table(s.device, s.dtype)
    speed = lut[lo] + (lut[hi] - lut[lo]) * frac
    over = s > float(SPEED_TABLE_MAX_SHIPS)
    speed_formula = compute_fleet_velocity(s)
    return torch.where(over, speed_formula, speed)

@dataclass
class GameState:
    alive: Tensor
    x: Tensor
    y: Tensor
    r: Tensor
    ships: Tensor
    prod: Tensor
    owner_abs: Tensor
    owned: Tensor
    is_enemy: Tensor
    is_neutral: Tensor
    orb_r: Tensor
    orb_a0: Tensor
    is_orbiting: Tensor
    angvel: Tensor
    step: Tensor
    f_alive: Tensor
    f_owner: Tensor
    f_x: Tensor
    f_y: Tensor
    f_angle: Tensor
    f_ships: Tensor
    player_id: int
    num_planets: int
    num_fleets: int
    device: torch.device

def extract_game_state(obs_tensors: dict, player_id: int | None = None) -> GameState:
    planets = obs_tensors['planets']
    initial = obs_tensors['initial_planets']
    fleets = obs_tensors['fleets']
    angvel = obs_tensors['angular_velocity'].float()
    step = obs_tensors['step'].float()
    if player_id is None:
        player_id = int(obs_tensors['player'].flatten()[0].item())
    num_planets, _ = planets.shape
    num_fleets, _ = fleets.shape
    device = planets.device
    pid = planets[..., 0]
    owner_abs = planets[..., 1]
    x = planets[..., 2]
    y = planets[..., 3]
    r = planets[..., 4]
    ships = planets[..., 5]
    prod = planets[..., 6]
    alive = pid >= 0.0
    owned = alive & (owner_abs == float(player_id))
    is_enemy = alive & (owner_abs >= 0.0) & (owner_abs != float(player_id))
    is_neutral = alive & (owner_abs < 0.0)
    ix = initial[..., 2]
    iy = initial[..., 3]
    i_r = initial[..., 4]
    dx0 = ix - PLAY_BOARD_CENTER
    dy0 = iy - PLAY_BOARD_CENTER
    orb_r_raw = torch.sqrt(dx0 * dx0 + dy0 * dy0)
    orb_a0 = torch.atan2(dy0, dx0)
    is_orbiting = alive & (orb_r_raw + i_r < ORBITAL_LIMIT_RADIUS) & (orb_r_raw > 0.5)
    orb_r = torch.where(is_orbiting, orb_r_raw, torch.zeros_like(orb_r_raw))
    f_pid = fleets[..., 0]
    f_alive = f_pid >= 0.0
    f_owner = fleets[..., 1]
    f_x = fleets[..., 2]
    f_y = fleets[..., 3]
    f_angle = fleets[..., 4]
    f_ships = fleets[..., 6]
    return GameState(
        alive=alive, x=x, y=y, r=r, ships=ships, prod=prod, owner_abs=owner_abs,
        owned=owned, is_enemy=is_enemy, is_neutral=is_neutral, orb_r=orb_r,
        orb_a0=orb_a0, is_orbiting=is_orbiting, angvel=angvel, step=step,
        f_alive=f_alive, f_owner=f_owner, f_x=f_x, f_y=f_y, f_angle=f_angle,
        f_ships=f_ships, player_id=player_id, num_planets=num_planets,
        num_fleets=num_fleets, device=device
    )

LAUNCH_SURFACE_OFFSET: float = 0.1
TARGET_HIT_SURFACE_OFFSET: float = 0.0
KAGGLE_SUN_RADIUS: float = SOLAR_SUN_RADIUS

def check_swept_collision(ax: Tensor, ay: Tensor, bx: Tensor, by: Tensor, p0x: Tensor, p0y: Tensor, p1x: Tensor, p1y: Tensor, r: Tensor) -> Tensor:
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = bx - ax - (p1x - p0x)
    dvy = by - ay - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    near_static = a < 1e-12
    c_hit = c <= 0.0
    disc = b * b - 4.0 * a * c
    has_root = disc >= 0.0
    safe_a = torch.where(near_static, torch.ones_like(a), a)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    quad_hit = has_root & (t2 >= 0.0) & (t1 <= 1.0)
    return torch.where(near_static, c_hit, quad_hit)

DEFAULT_MOVEMENT_HORIZON = 20
DEFAULT_DRIFT_EPSILON = 0.0001
DEFAULT_MAX_TRACKED_FLEETS = 64

@dataclass(frozen=True)
class SimulationConfig:
    movement_horizon: int = DEFAULT_MOVEMENT_HORIZON
    drift_epsilon: float = DEFAULT_DRIFT_EPSILON
    track_fleets: bool = False
    player_count: int | None = None
    max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS

@dataclass(frozen=True)
class GarrisonPrediction:
    owner: Tensor
    ships: Tensor
    pre_combat_owner: Tensor | None = None
    pre_combat_ships: Tensor | None = None
    arrivals_by_owner: Tensor | None = None

@dataclass
class StatePredictor:
    x: Tensor
    y: Tensor
    alive_by_step: Tensor
    planet_ids: Tensor
    radii: Tensor
    planet_owner: Tensor
    planet_ships: Tensor
    planet_prod: Tensor
    base_step: Tensor
    comet_planet_ids: Tensor
    comet_path_index: Tensor
    movement_horizon: int = DEFAULT_MOVEMENT_HORIZON
    drift_epsilon: float = DEFAULT_DRIFT_EPSILON
    track_fleets: bool = False
    player_count: int | None = None
    max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS
    fleet_buckets: Tensor | None = None
    fleet_last_step: Tensor | None = None
    tracked_fleet_ids: Tensor | None = None
    tracked_fleet_eta: Tensor | None = None
    tracked_fleet_target_slot: Tensor | None = None
    tracked_fleet_owner: Tensor | None = None
    tracked_fleet_ships: Tensor | None = None
    garrison_owner_cache: Tensor | None = None
    garrison_ships_cache: Tensor | None = None
    garrison_pre_combat_owner_cache: Tensor | None = None
    garrison_pre_combat_ships_cache: Tensor | None = None
    garrison_dirty_from: Tensor | None = None
    pending_source_planets: Tensor | None = None
    pending_ships: Tensor | None = None
    pending_angle: Tensor | None = None
    pending_target_slots: Tensor | None = None
    pending_eta: Tensor | None = None
    pending_owners: Tensor | None = None
    pending_prev_nfid: Tensor | None = None
    pending_stash_step: Tensor | None = None

    @property
    def num_planets(self) -> int:
        return int(self.planet_ids.shape[0])

    @property
    def device(self) -> torch.device:
        return self.x.device

    @property
    def dtype(self) -> torch.dtype:
        return self.x.dtype

    @property
    def config(self) -> SimulationConfig:
        return SimulationConfig(
            movement_horizon=int(self.movement_horizon),
            drift_epsilon=float(self.drift_epsilon),
            track_fleets=bool(self.track_fleets),
            player_count=self.player_count,
            max_tracked_fleets=int(self.max_tracked_fleets)
        )

    @classmethod
    def from_obs_tensors(cls, obs_tensors: dict, *, config: SimulationConfig | None = None, movement_horizon: int = DEFAULT_MOVEMENT_HORIZON, drift_epsilon: float = DEFAULT_DRIFT_EPSILON, track_fleets: bool = False, player_count: int | None = None, max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS) -> StatePredictor:
        cfg = config if config is not None else SimulationConfig(
            movement_horizon=int(movement_horizon),
            drift_epsilon=float(drift_epsilon),
            track_fleets=bool(track_fleets),
            player_count=player_count,
            max_tracked_fleets=int(max_tracked_fleets)
        )
        built = forecast_planet_positions(obs_tensors, int(cfg.movement_horizon))
        resolved_player_count = get_next_fleet_id(obs_tensors, cfg.player_count) if cfg.track_fleets else cfg.player_count
        movement = cls(
            x=built['x'], y=built['y'], alive_by_step=built['alive_by_step'],
            planet_ids=built['planet_ids'], radii=built['radii'], planet_owner=built['owner'],
            planet_ships=built['ships'], planet_prod=built['prod'], base_step=built['step'],
            comet_planet_ids=built['comet_planet_ids'], comet_path_index=built['comet_path_index'],
            movement_horizon=int(cfg.movement_horizon), drift_epsilon=float(cfg.drift_epsilon),
            track_fleets=bool(cfg.track_fleets), player_count=resolved_player_count,
            max_tracked_fleets=int(cfg.max_tracked_fleets)
        )
        if movement.track_fleets:
            movement._init_fleet_tracking(obs_tensors, reset_ledger=True)
            movement._ingest_obs_fleets(obs_tensors)
        return movement

    def update(self, obs_tensors: dict) -> StatePredictor:
        planets = obs_tensors['planets']
        if planets.device != self.device or planets.shape[0] != self.num_planets or int(self.x.shape[0]) != int(self.movement_horizon) + 1:
            fresh = type(self).from_obs_tensors(
                obs_tensors, movement_horizon=self.movement_horizon,
                drift_epsilon=self.drift_epsilon, track_fleets=self.track_fleets,
                player_count=self.player_count, max_tracked_fleets=int(self.max_tracked_fleets)
            )
            self._copy_from(fresh)
            return self
        if self.track_fleets:
            current_player_count = get_next_fleet_id(obs_tensors, self.player_count)
            if self.fleet_buckets is None or self.fleet_last_step is None or self.tracked_fleet_ids is None or (tuple(self.fleet_buckets.shape) != (self.num_planets, int(self.movement_horizon), int(current_player_count))) or (self.fleet_buckets.device != self.device) or (int(self.tracked_fleet_ids.shape[0]) < int(self.max_tracked_fleets)):
                self.player_count = int(current_player_count)
                self._init_fleet_tracking(obs_tensors, reset_ledger=True)
        obs_for_decision = extract_game_state(obs_tensors)
        horizon = int(self.movement_horizon)
        planet_ids_now = planets[..., 0].long()
        radii_now = planets[..., 4].to(dtype=self.dtype)
        owner_now = planets[..., 1].to(device=self.device, dtype=torch.long)
        owner_now = torch.where(obs_for_decision.alive, owner_now, torch.full_like(owner_now, -1))
        ships_now = planets[..., 5].to(device=self.device, dtype=self.dtype)
        prod_now = planets[..., 6].to(device=self.device, dtype=self.dtype)
        step_now = obs_for_decision.step.to(device=self.device, dtype=torch.long)
        comet_ids_now, comet_idx_now = get_comet_info(obs_tensors, self.device)
        current_obs_x = planets[..., 2].to(device=self.device, dtype=self.dtype)
        current_obs_y = planets[..., 3].to(device=self.device, dtype=self.dtype)
        current_alive = obs_for_decision.alive
        ids_same = bool((planet_ids_now == self.planet_ids).all())
        same_step = bool(step_now == self.base_step)
        next_step = bool(step_now == self.base_step + 1)
        comet_same = are_tensors_equal(comet_ids_now, self.comet_planet_ids)
        comet_idx_same = are_tensors_equal(comet_idx_now, self.comet_path_index)
        expected_next_idx = torch.where(self.comet_path_index >= 0, self.comet_path_index + 1, self.comet_path_index)
        comet_idx_next = are_tensors_equal(comet_idx_now, expected_next_idx)
        same_alive_ok = bool((current_alive == self.alive_by_step[0]).all())
        next_alive_ok = bool((current_alive == self.alive_by_step[1]).all())
        same_drift_ok = validate_position_drift(self.x[0], self.y[0], current_obs_x, current_obs_y, current_alive, float(self.drift_epsilon))
        next_drift_ok = validate_position_drift(self.x[1], self.y[1], current_obs_x, current_obs_y, current_alive, float(self.drift_epsilon))
        keep = ids_same and same_step and comet_same and comet_idx_same and same_alive_ok and same_drift_ok
        roll = ids_same and next_step and comet_same and comet_idx_next and next_alive_ok and next_drift_ok
        rebuild = not (keep or roll)
        if rebuild:
            built = forecast_planet_positions(obs_tensors, horizon)
        elif roll:
            last_offset = torch.tensor([horizon], dtype=torch.long, device=self.device)
            built = forecast_planet_positions(obs_tensors, horizon, offsets=last_offset)
        else:
            built = None
        if roll:
            assert built is not None
            self.x[:-1] = self.x[1:].clone()
            self.y[:-1] = self.y[1:].clone()
            self.alive_by_step[:-1] = self.alive_by_step[1:].clone()
            self.x[-1] = built['x'][-1]
            self.y[-1] = built['y'][-1]
            self.alive_by_step[-1] = built['alive_by_step'][-1]
            self._roll_garrison_projection()
        if rebuild:
            assert built is not None
            self.x[:] = built['x']
            self.y[:] = built['y']
            self.alive_by_step[:] = built['alive_by_step']
            self._mark_garrison_dirty_all(0)
        if roll or rebuild:
            self.planet_ids[:] = planet_ids_now
            self.radii[:] = radii_now
            self.base_step = step_now
            self.comet_planet_ids = comet_ids_now
            self.comet_path_index = comet_idx_now
        self._refresh_garrison_base({
            'planet_ids': planet_ids_now, 'radii': radii_now,
            'owner': owner_now, 'ships': ships_now, 'prod': prod_now, 'step': step_now
        })
        if self.track_fleets:
            self._roll_fleet_buckets_phase1(step_now)
            if rebuild and (not ids_same):
                self._reset_fleet_tracking()
            self._reconcile_pending_own_launches(obs_tensors)
            self._ingest_obs_fleets(obs_tensors)
            self._reconcile_obs_fleets(obs_tensors)
        return self

    def all_positions(self, k: int) -> tuple[Tensor, Tensor]:
        idx = self._k_index(k)
        return (self.x[idx], self.y[idx])

    def alive_at(self, k: int) -> Tensor:
        return self.alive_by_step[self._k_index(k)]

    def position_at_slots(self, slots: Tensor, k: int) -> tuple[Tensor, Tensor]:
        slots = slots.to(device=self.device, dtype=torch.long).clamp(0, max(self.num_planets - 1, 0))
        px, py = self.all_positions(k)
        out_x = px[slots].to(dtype=self.dtype)
        out_y = py[slots].to(dtype=self.dtype)
        return (out_x, out_y)

    def pairwise_distance(self, k: int) -> Tensor:
        px, py = self.all_positions(k)
        dx = px.unsqueeze(1) - px.unsqueeze(0)
        dy = py.unsqueeze(1) - py.unsqueeze(0)
        return torch.sqrt((dx * dx + dy * dy).clamp(min=0.0))

    def garrison_status(self, planet_slots: Tensor | None = None, *, max_horizon: int | None = None) -> GarrisonPrediction:
        self._require_fleet_buckets()
        slots, out_prefix = self._normalize_garrison_slots(planet_slots)
        requested_horizon = int(self.movement_horizon if max_horizon is None else max(0, min(int(max_horizon), int(self.movement_horizon))))
        self._refresh_garrison_projection(slots, requested_horizon=requested_horizon)
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_dirty_from is not None
        owner = self.garrison_owner_cache[slots][:, :requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
        ships = self.garrison_ships_cache[slots][:, :requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
        pre_combat_owner: Tensor | None = None
        pre_combat_ships: Tensor | None = None
        if self.garrison_pre_combat_owner_cache is not None and self.garrison_pre_combat_ships_cache is not None:
            pre_combat_owner = self.garrison_pre_combat_owner_cache[slots][:, :requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
            pre_combat_ships = self.garrison_pre_combat_ships_cache[slots][:, :requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
        arrivals_by_owner: Tensor | None = None
        if self.fleet_buckets is not None and requested_horizon > 0:
            num_players = int(self.fleet_buckets.shape[-1])
            arrivals_full = self.fleet_buckets[slots].reshape(*out_prefix, int(self.movement_horizon), num_players)
            arrivals_trimmed = arrivals_full[..., :requested_horizon, :]
            zero_frame = torch.zeros(*out_prefix, 1, num_players, dtype=arrivals_trimmed.dtype, device=self.device)
            arrivals_by_owner = torch.cat([zero_frame, arrivals_trimmed], dim=-2)
        status = GarrisonPrediction(
            owner=owner, ships=ships, pre_combat_owner=pre_combat_owner,
            pre_combat_ships=pre_combat_ships, arrivals_by_owner=arrivals_by_owner
        )
        return status

    def _clear_pending_mask(self, mask: Tensor) -> None:
        if self.pending_owners is None:
            return
        self.pending_owners[mask] = -1
        assert self.pending_source_planets is not None
        self.pending_source_planets[mask] = -1
        assert self.pending_ships is not None
        self.pending_ships[mask] = 0
        assert self.pending_angle is not None
        self.pending_angle[mask] = 0.0
        assert self.pending_target_slots is not None
        self.pending_target_slots[mask] = -1
        assert self.pending_eta is not None
        self.pending_eta[mask] = 0.0
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid[mask] = 0
        assert self.pending_stash_step is not None
        self.pending_stash_step[mask] = -1

    def _ensure_pending_capacity(self, needed: int) -> None:
        device = self.device
        if self.pending_owners is None:
            initial = max(4, int(needed))
            shape = (initial,)
            self.pending_owners = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_source_planets = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_ships = torch.zeros(shape, dtype=torch.long, device=device)
            self.pending_angle = torch.zeros(shape, dtype=self.dtype, device=device)
            self.pending_target_slots = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_eta = torch.zeros(shape, dtype=self.dtype, device=device)
            self.pending_prev_nfid = torch.zeros(shape, dtype=torch.long, device=device)
            self.pending_stash_step = torch.full(shape, -1, dtype=torch.long, device=device)
            return
        assert self.pending_owners is not None
        empty_count = int((self.pending_owners == -1).sum().item())
        shortage = int(needed) - empty_count
        if shortage <= 0:
            return
        cur_length = int(self.pending_owners.shape[0])
        extra = max(shortage, cur_length)
        new_length = cur_length + extra

        def _grow(t: Tensor, fill: float | int) -> Tensor:
            extension = torch.full((new_length - cur_length,), fill, dtype=t.dtype, device=device)
            return torch.cat([t, extension], dim=0)

        self.pending_owners = _grow(self.pending_owners, -1)
        assert self.pending_source_planets is not None
        self.pending_source_planets = _grow(self.pending_source_planets, -1)
        assert self.pending_ships is not None
        self.pending_ships = _grow(self.pending_ships, 0)
        assert self.pending_angle is not None
        self.pending_angle = _grow(self.pending_angle, 0.0)
        assert self.pending_target_slots is not None
        self.pending_target_slots = _grow(self.pending_target_slots, -1)
        assert self.pending_eta is not None
        self.pending_eta = _grow(self.pending_eta, 0.0)
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid = _grow(self.pending_prev_nfid, 0)
        assert self.pending_stash_step is not None
        self.pending_stash_step = _grow(self.pending_stash_step, -1)

    def stash_pending_own_launches(self, *, owner_id: int | Tensor, source_slots: Tensor, ships: Tensor, angle: Tensor, target_slots: Tensor, eta: Tensor, valid: Tensor, prev_next_fleet_id: int | Tensor) -> None:
        if not self.track_fleets:
            return
        device = self.device
        valid_mask = valid.to(device=device, dtype=torch.bool).reshape(-1)
        if not bool(valid_mask.any()):
            return
        src = source_slots.to(device=device, dtype=torch.long).reshape(-1)
        ships_t = ships.to(device=device, dtype=torch.long).reshape(-1)
        angle_t = angle.to(device=device, dtype=self.dtype).reshape(-1)
        tgt_t = target_slots.to(device=device, dtype=torch.long).reshape(-1)
        eta_t = eta.to(device=device, dtype=self.dtype).reshape(-1)
        src_safe = src.clamp(min=0, max=max(int(self.num_planets) - 1, 0))
        source_planet_ids = self.planet_ids[src_safe]
        num_launches = int(valid_mask.shape[0])
        if isinstance(prev_next_fleet_id, Tensor):
            prev_nfid_scalar = int(prev_next_fleet_id.flatten()[0].item())
        else:
            prev_nfid_scalar = int(prev_next_fleet_id)
        prev_nfid_list = torch.full((num_launches,), prev_nfid_scalar, dtype=torch.long, device=device)
        owner_scalar = int(owner_id.flatten()[0].item()) if isinstance(owner_id, Tensor) else int(owner_id)
        owner_list = torch.full((num_launches,), owner_scalar, dtype=torch.long, device=device)
        stash_step_scalar = int(self.base_step.item()) if isinstance(self.base_step, Tensor) else -1
        stash_step_list = torch.full((num_launches,), stash_step_scalar, dtype=torch.long, device=device)
        if self.pending_owners is not None:
            same_owner = self.pending_owners == owner_scalar
            if bool(same_owner.any()):
                self._clear_pending_mask(same_owner)
        per_needed = int(valid_mask.sum().item())
        self._ensure_pending_capacity(per_needed)
        assert self.pending_owners is not None
        empty_slots = torch.nonzero(self.pending_owners == -1, as_tuple=True)[0]
        k_in = torch.nonzero(valid_mask, as_tuple=True)[0]
        slot_in_pending = empty_slots[:k_in.numel()]
        self.pending_owners[slot_in_pending] = owner_list[k_in]
        assert self.pending_source_planets is not None
        self.pending_source_planets[slot_in_pending] = source_planet_ids[k_in]
        assert self.pending_ships is not None
        self.pending_ships[slot_in_pending] = ships_t[k_in]
        assert self.pending_angle is not None
        self.pending_angle[slot_in_pending] = angle_t[k_in]
        assert self.pending_target_slots is not None
        self.pending_target_slots[slot_in_pending] = tgt_t[k_in]
        assert self.pending_eta is not None
        self.pending_eta[slot_in_pending] = eta_t[k_in]
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid[slot_in_pending] = prev_nfid_list[k_in]
        assert self.pending_stash_step is not None
        self.pending_stash_step[slot_in_pending] = stash_step_list[k_in]

    def _reconcile_pending_own_launches(self, obs_tensors: dict) -> None:
        if not self.track_fleets:
            return
        if self.pending_owners is None or self.tracked_fleet_ids is None:
            return
        active_mask = self.pending_owners != -1
        if not bool(active_mask.any()):
            return
        device = self.device
        step_tensor = obs_tensors.get('step')
        if step_tensor is not None:
            assert self.pending_stash_step is not None
            step_scalar = int(step_tensor.flatten()[0].item()) if isinstance(step_tensor, Tensor) else int(step_tensor)
            advanced = step_scalar > self.pending_stash_step
            active_mask = active_mask & advanced
        if not bool(active_mask.any()):
            return
        fleets = obs_tensors['fleets'].to(device=device)
        fleet_ids = fleets[..., 0].to(dtype=torch.long)
        obs_owner = fleets[..., 1].to(dtype=torch.long)
        obs_angle = fleets[..., 4].to(dtype=self.dtype)
        obs_from = fleets[..., 5].to(dtype=torch.long)
        obs_ships = fleets[..., 6].to(dtype=torch.long)
        assert self.pending_owners is not None
        assert self.pending_source_planets is not None
        assert self.pending_ships is not None
        assert self.pending_angle is not None
        assert self.pending_target_slots is not None
        assert self.pending_eta is not None
        assert self.pending_prev_nfid is not None
        match_FL = active_mask.unsqueeze(0) & (fleet_ids.unsqueeze(1) >= 0) & (obs_owner.unsqueeze(1) == self.pending_owners.unsqueeze(0)) & (obs_from.unsqueeze(1) == self.pending_source_planets.unsqueeze(0)) & (obs_ships.unsqueeze(1) == self.pending_ships.unsqueeze(0)) & (obs_angle.unsqueeze(1) == self.pending_angle.unsqueeze(0)) & (fleet_ids.unsqueeze(1) >= self.pending_prev_nfid.unsqueeze(0))
        INF = torch.iinfo(torch.long).max
        id_for_match = torch.where(match_FL, fleet_ids.unsqueeze(1).expand_as(match_FL), torch.full_like(match_FL, INF, dtype=torch.long))
        chosen_id, _ = id_for_match.min(dim=0)
        eta_now = torch.ceil(self.pending_eta).to(dtype=torch.long) - 1
        expect_obs_match = active_mask & (eta_now > 0)
        no_match = expect_obs_match & (chosen_id == INF)
        matched = expect_obs_match & (chosen_id != INF)
        if int(active_mask.shape[0]) > 1:
            chosen_for_matched = torch.where(matched, chosen_id, torch.full_like(chosen_id, INF))
            sorted_ids, _ = chosen_for_matched.sort()
            dup = bool(((sorted_ids[1:] == sorted_ids[:-1]) & (sorted_ids[1:] != INF)).any())
            if dup:
                raise AssertionError('Pending-launch reconciliation: multiple pending entries resolved to the same engine fleet id. This usually means multi-launch from the same source with identical (ships, angle) tuples processed in an unexpected order.')
        if bool(matched.any()):
            l_idx = torch.where(matched)[0]
            real_ids = chosen_id[l_idx]
            self._ledger_bulk_insert(real_ids, eta_now[l_idx], self.pending_target_slots[l_idx], self.pending_owners[l_idx], self.pending_ships[l_idx].to(dtype=self.dtype))
        if bool(no_match.any()):
            self._decrement_unmatched_arrivals(no_match)
        self._clear_pending_mask(active_mask)

    def _decrement_unmatched_arrivals(self, no_match: Tensor) -> None:
        assert self.pending_eta is not None
        assert self.pending_owners is not None
        assert self.pending_ships is not None
        assert self.pending_target_slots is not None
        buckets = self._require_fleet_buckets()
        eta_now = torch.ceil(self.pending_eta).to(dtype=torch.long) - 1
        h_idx_now = eta_now - 1
        horizon = int(self.movement_horizon)
        num_players = int(buckets.shape[2])
        valid = no_match & (h_idx_now >= 0) & (h_idx_now < horizon) & (self.pending_target_slots >= 0) & (self.pending_target_slots < int(self.num_planets)) & (self.pending_owners >= 0) & (self.pending_owners < num_players) & (self.pending_ships > 0)
        if not bool(valid.any()):
            return
        target = self.pending_target_slots[valid]
        h_idx_sel = h_idx_now[valid]
        owner_sel = self.pending_owners[valid]
        ships_sel = self.pending_ships[valid].to(dtype=self.dtype)
        buckets.index_put_((target, h_idx_sel, owner_sel), -ships_sel, accumulate=True)
        self._mark_garrison_dirty(target, h_idx_sel + 1)

    def record_fleet_arrivals(self, *, target_slots: Tensor, owner_ids: Tensor | int, ships: Tensor, eta: Tensor, valid: Tensor | None = None) -> None:
        buckets = self._require_fleet_buckets()
        target_slots, ships, eta = torch.broadcast_tensors(target_slots.to(device=self.device, dtype=torch.long), ships.to(device=self.device, dtype=self.dtype), eta.to(device=self.device, dtype=self.dtype))
        if isinstance(owner_ids, int):
            owner = torch.full_like(target_slots, int(owner_ids), dtype=torch.long, device=self.device)
        else:
            owner = torch.broadcast_to(owner_ids.to(device=self.device, dtype=torch.long), target_slots.shape)
        if valid is None:
            valid_mask = torch.ones_like(target_slots, dtype=torch.bool)
        else:
            valid_mask = torch.broadcast_to(valid.to(device=self.device, dtype=torch.bool), target_slots.shape)
        h_idx = torch.ceil(eta).to(dtype=torch.long) - 1
        valid_mask = valid_mask & (target_slots >= 0) & (target_slots < self.num_planets) & (owner >= 0) & (owner < int(buckets.shape[2])) & (h_idx >= 0) & (h_idx < int(self.movement_horizon)) & (ships > 0.0)
        if not bool(valid_mask.any()):
            return
        buckets.index_put_((target_slots[valid_mask], h_idx[valid_mask], owner[valid_mask]), ships[valid_mask], accumulate=True)
        self._mark_garrison_dirty(target_slots[valid_mask], h_idx[valid_mask] + 1)

    def _normalize_garrison_slots(self, planet_slots: Tensor | None) -> tuple[Tensor, torch.Size]:
        if planet_slots is None:
            slots = torch.arange(self.num_planets, dtype=torch.long, device=self.device)
            return (slots, slots.shape)
        raw = planet_slots.to(device=self.device, dtype=torch.long)
        out_prefix = raw.shape
        slots = raw.reshape(-1).clamp(0, max(self.num_planets - 1, 0))
        return (slots, out_prefix)

    def _ensure_garrison_cache(self) -> None:
        self._ensure_garrison_cache_impl()

    def _ensure_garrison_cache_impl(self) -> None:
        expected_owner = (self.num_planets, int(self.movement_horizon) + 1)
        expected_dirty = (self.num_planets,)
        if self.garrison_owner_cache is not None and self.garrison_ships_cache is not None and (self.garrison_pre_combat_owner_cache is not None) and (self.garrison_pre_combat_ships_cache is not None) and (self.garrison_dirty_from is not None) and (tuple(self.garrison_owner_cache.shape) == expected_owner) and (tuple(self.garrison_ships_cache.shape) == expected_owner) and (tuple(self.garrison_pre_combat_owner_cache.shape) == expected_owner) and (tuple(self.garrison_pre_combat_ships_cache.shape) == expected_owner) and (tuple(self.garrison_dirty_from.shape) == expected_dirty) and (self.garrison_owner_cache.device == self.device) and (self.garrison_ships_cache.device == self.device):
            return
        horizon = int(self.movement_horizon)
        self.garrison_owner_cache = torch.full((self.num_planets, horizon + 1), -1, dtype=torch.long, device=self.device)
        self.garrison_ships_cache = torch.zeros(self.num_planets, horizon + 1, dtype=self.dtype, device=self.device)
        self.garrison_pre_combat_owner_cache = self.garrison_owner_cache.clone()
        self.garrison_pre_combat_ships_cache = self.garrison_ships_cache.clone()
        self.garrison_owner_cache[:, 0] = self.planet_owner
        self.garrison_ships_cache[:, 0] = self.planet_ships
        self.garrison_pre_combat_owner_cache[:, 0] = self.planet_owner
        self.garrison_pre_combat_ships_cache[:, 0] = self.planet_ships
        self.garrison_dirty_from = torch.zeros(self.num_planets, dtype=torch.long, device=self.device)

    def _refresh_garrison_projection(self, slots: Tensor, *, requested_horizon: int | None = None) -> None:
        self._ensure_garrison_cache()
        assert self.fleet_buckets is not None
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_dirty_from is not None
        p_idx = torch.unique(slots.reshape(-1).clamp(min=0, max=max(self.num_planets - 1, 0)))
        if p_idx.numel() == 0:
            return
        dirty = self.garrison_dirty_from[p_idx]
        horizon = int(self.movement_horizon if requested_horizon is None else max(0, min(int(requested_horizon), int(self.movement_horizon))))
        needs_refresh = dirty <= horizon
        if not bool(needs_refresh.any()):
            return
        p_idx = p_idx[needs_refresh]
        owner = self.planet_owner[p_idx].clone()
        ships = self.planet_ships[p_idx].clone()
        self.garrison_owner_cache[p_idx, 0] = owner
        self.garrison_ships_cache[p_idx, 0] = ships
        assert self.garrison_pre_combat_owner_cache is not None
        assert self.garrison_pre_combat_ships_cache is not None
        self.garrison_pre_combat_owner_cache[p_idx, 0] = owner
        self.garrison_pre_combat_ships_cache[p_idx, 0] = ships
        prod = self.planet_prod[p_idx]
        if horizon == 0:
            self.garrison_dirty_from[p_idx] = horizon + 1
            return
        self._fill_garrison_trajectory(p_idx=p_idx, init_owner=owner, init_ships=ships, prod=prod, horizon=horizon)
        self.garrison_dirty_from[p_idx] = horizon + 1

    def _fill_garrison_trajectory(self, *, p_idx: Tensor, init_owner: Tensor, init_ships: Tensor, prod: Tensor, horizon: int) -> None:
        assert self.fleet_buckets is not None
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_pre_combat_owner_cache is not None
        assert self.garrison_pre_combat_ships_cache is not None
        horizon_val = int(horizon)
        num_targets = int(p_idx.numel())
        if num_targets == 0 or horizon_val == 0:
            return
        alive_step = self.alive_by_step[:, p_idx].transpose(0, 1)
        alive_before = alive_step[:, :horizon_val]
        alive_now = alive_step[:, 1:]
        arrivals = self.fleet_buckets[p_idx, :horizon_val, :]
        has_any_arrival = (arrivals > 0.0).any(dim=-1).any(dim=-1)
        alive_all_true = alive_step.all(dim=1)
        simple_mask = ~has_any_arrival & alive_all_true
        alive_step_full = alive_step
        n_simple = int(simple_mask.sum().item())
        n_complex = num_targets - n_simple
        if n_simple > 0:
            simple_p = p_idx[simple_mask]
            simple_owner = init_owner[simple_mask]
            simple_ships = init_ships[simple_mask]
            simple_prod = prod[simple_mask]
            owner_alive_factor = (simple_owner >= 0).to(dtype=simple_ships.dtype)
            k_range = torch.arange(1, horizon_val + 1, device=self.device, dtype=simple_ships.dtype)
            ships_traj = simple_ships.unsqueeze(1) + simple_prod.unsqueeze(1) * owner_alive_factor.unsqueeze(1) * k_range.unsqueeze(0)
            owner_traj = simple_owner.unsqueeze(1).expand(-1, horizon_val)
            self.garrison_owner_cache[simple_p, 1:horizon_val + 1] = owner_traj
            self.garrison_ships_cache[simple_p, 1:horizon_val + 1] = ships_traj
            self.garrison_pre_combat_owner_cache[simple_p, 1:horizon_val + 1] = owner_traj
            self.garrison_pre_combat_ships_cache[simple_p, 1:horizon_val + 1] = ships_traj
        if n_complex == 0:
            return
        complex_mask = ~simple_mask
        cp = p_idx[complex_mask]
        arrivals_c = arrivals[complex_mask]
        alive_before_c = alive_before[complex_mask]
        alive_now_c = alive_now[complex_mask]
        alive_step_c = alive_step_full[complex_mask]
        state_owner = init_owner[complex_mask].clone()
        state_ships = init_ships[complex_mask].clone()
        prod_c = prod[complex_mask]
        num_players = int(arrivals_c.shape[-1])
        if num_players >= 2:
            top2 = arrivals_c.topk(k=2, dim=-1)
            top_ships_traj = top2.values[..., 0]
            second_ships_traj = top2.values[..., 1]
            top_owner_traj = top2.indices[..., 0].to(dtype=torch.long)
        else:
            top_ships_traj, top_owner_traj = arrivals_c.max(dim=-1)
            second_ships_traj = torch.zeros_like(top_ships_traj)
            top_owner_traj = top_owner_traj.to(dtype=torch.long)
        tied = top_ships_traj == second_ships_traj
        survivor_ships_traj = torch.where(tied, torch.zeros_like(top_ships_traj), (top_ships_traj - second_ships_traj).clamp(min=0.0))
        survivor_owner_traj = top_owner_traj
        zero_ships_scalar = torch.zeros((), dtype=state_ships.dtype, device=self.device)
        neg_one_owner_scalar = torch.full((), -1, dtype=state_owner.dtype, device=self.device)
        zero_prod_scalar = torch.zeros((), dtype=prod_c.dtype, device=self.device)
        combat_event_per_step = (survivor_ships_traj > 0.0) & alive_now_c
        alive_change_per_step = alive_before_c != alive_now_c
        any_event_per_step = (combat_event_per_step | alive_change_per_step).any(dim=0)
        arange_h = torch.arange(1, horizon_val + 1, device=self.device, dtype=torch.long)
        k_last_tensor = torch.where(any_event_per_step, arange_h, torch.zeros_like(arange_h)).max()
        k_last = int(k_last_tensor.item())
        loop_iters = max(0, k_last)
        tail_steps = horizon_val - loop_iters
        if loop_iters > 0:
            for k in range(1, loop_iters + 1):
                a_before = alive_before_c[:, k - 1]
                a_now = alive_now_c[:, k - 1]
                s_owner = survivor_owner_traj[:, k - 1]
                s_ships = survivor_ships_traj[:, k - 1]
                produces = a_before & (state_owner >= 0)
                state_ships = state_ships + torch.where(produces, prod_c, zero_prod_scalar)
                pre_owner = torch.where(a_now, state_owner, neg_one_owner_scalar)
                pre_ships = torch.where(a_now, state_ships, zero_ships_scalar)
                self.garrison_pre_combat_owner_cache[cp, k] = pre_owner
                self.garrison_pre_combat_ships_cache[cp, k] = pre_ships
                has_combat = (s_ships > 0.0) & a_now
                same = state_owner == s_owner
                diff = state_ships - s_ships
                attacker_wins = ~same & (diff < 0.0)
                combat_ships = torch.where(same, state_ships + s_ships, diff.abs())
                combat_owner = torch.where(attacker_wins, s_owner, state_owner)
                state_ships = torch.where(has_combat, combat_ships, state_ships)
                state_owner = torch.where(has_combat, combat_owner, state_owner)
                state_owner = torch.where(a_now, state_owner, neg_one_owner_scalar)
                state_ships = torch.where(a_now, state_ships, zero_ships_scalar)
                self.garrison_owner_cache[cp, k] = state_owner
                self.garrison_ships_cache[cp, k] = state_ships
        if tail_steps > 0:
            alive_at_k_last = alive_step_c[:, k_last]
            state_owner = torch.where(alive_at_k_last, state_owner, neg_one_owner_scalar)
            state_ships = torch.where(alive_at_k_last, state_ships, zero_ships_scalar)
            owner_alive_factor = (state_owner >= 0).to(dtype=state_ships.dtype) * alive_at_k_last.to(dtype=state_ships.dtype)
            dk_range = torch.arange(1, tail_steps + 1, device=self.device, dtype=state_ships.dtype)
            ships_traj_tail = state_ships.unsqueeze(1) + prod_c.unsqueeze(1) * owner_alive_factor.unsqueeze(1) * dk_range.unsqueeze(0)
            owner_traj_tail = state_owner.unsqueeze(1).expand(-1, tail_steps)
            self.garrison_owner_cache[cp, k_last + 1:horizon_val + 1] = owner_traj_tail
            self.garrison_ships_cache[cp, k_last + 1:horizon_val + 1] = ships_traj_tail
            self.garrison_pre_combat_owner_cache[cp, k_last + 1:horizon_val + 1] = owner_traj_tail
            self.garrison_pre_combat_ships_cache[cp, k_last + 1:horizon_val + 1] = ships_traj_tail

    def _roll_garrison_projection(self) -> None:
        if self.garrison_owner_cache is None or self.garrison_ships_cache is None or self.garrison_pre_combat_owner_cache is None or (self.garrison_pre_combat_ships_cache is None) or (self.garrison_dirty_from is None):
            return
        horizon = int(self.movement_horizon)
        if horizon > 0:
            self.garrison_owner_cache[:, :-1] = self.garrison_owner_cache[:, 1:].clone()
            self.garrison_ships_cache[:, :-1] = self.garrison_ships_cache[:, 1:].clone()
            self.garrison_pre_combat_owner_cache[:, :-1] = self.garrison_pre_combat_owner_cache[:, 1:].clone()
            self.garrison_pre_combat_ships_cache[:, :-1] = self.garrison_pre_combat_ships_cache[:, 1:].clone()
            self.garrison_dirty_from = (self.garrison_dirty_from - 1).clamp(min=0)
            self.garrison_dirty_from = torch.minimum(self.garrison_dirty_from, torch.full_like(self.garrison_dirty_from, horizon))
        else:
            self.garrison_dirty_from[:] = 0

    def _refresh_garrison_base(self, built: dict[str, Tensor]) -> None:
        owner = built['owner'].to(device=self.device, dtype=torch.long)
        ships = built['ships'].to(device=self.device, dtype=self.dtype)
        prod = built['prod'].to(device=self.device, dtype=self.dtype)
        prod_changed = tuple(self.planet_prod.shape) != tuple(prod.shape) or self.planet_prod != prod
        self.planet_owner = owner
        self.planet_ships = ships
        self.planet_prod = prod
        if self.garrison_owner_cache is None or self.garrison_ships_cache is None or self.garrison_dirty_from is None:
            return
        base_changed = (self.garrison_owner_cache[:, 0] != owner) | (self.garrison_ships_cache[:, 0] != ships)
        self.garrison_owner_cache[:, 0] = owner
        self.garrison_ships_cache[:, 0] = ships
        if self.garrison_pre_combat_owner_cache is not None:
            self.garrison_pre_combat_owner_cache[:, 0] = owner
        if self.garrison_pre_combat_ships_cache is not None:
            self.garrison_pre_combat_ships_cache[:, 0] = ships
        if bool(base_changed.any()):
            self.garrison_dirty_from[base_changed] = 0
        if isinstance(prod_changed, Tensor) and bool(prod_changed.any()):
            self.garrison_dirty_from[prod_changed] = torch.minimum(self.garrison_dirty_from[prod_changed], torch.ones_like(self.garrison_dirty_from[prod_changed]))
        elif not isinstance(prod_changed, Tensor) and prod_changed:
            self.garrison_dirty_from[:] = torch.minimum(self.garrison_dirty_from, torch.ones_like(self.garrison_dirty_from))

    def _mark_garrison_dirty(self, planet_idx: Tensor, start_step: Tensor | int) -> None:
        if self.garrison_dirty_from is None:
            return
        p = planet_idx.to(device=self.device, dtype=torch.long)
        if isinstance(start_step, int):
            start = torch.full((), int(start_step), dtype=torch.long, device=self.device)
        else:
            start = start_step.to(device=self.device, dtype=torch.long)
        p, start = torch.broadcast_tensors(p, start)
        p = p.reshape(-1)
        start = start.reshape(-1)
        if p.numel() == 0:
            return
        start = start.clamp(min=0, max=int(self.movement_horizon))
        valid = (p >= 0) & (p < self.num_planets)
        if not bool(valid.any()):
            return
        p = p[valid]
        start = start[valid]
        flat = self.garrison_dirty_from
        unique_idx, inverse = torch.unique(p, return_inverse=True)
        if unique_idx.numel() == p.numel():
            flat[unique_idx] = torch.minimum(flat[unique_idx], start)
            return
        sentinel = int(self.movement_horizon) + 1
        candidate = torch.full((unique_idx.shape[0],), sentinel, dtype=torch.long, device=self.device)
        candidate.scatter_reduce_(0, inverse, start, reduce='amin', include_self=True)
        flat[unique_idx] = torch.minimum(flat[unique_idx], candidate)

    def _mark_garrison_dirty_all(self, start_step: int) -> None:
        if self.garrison_dirty_from is None:
            return
        self.garrison_dirty_from = torch.minimum(self.garrison_dirty_from, torch.full_like(self.garrison_dirty_from, int(start_step)))

    def _init_fleet_tracking(self, obs_tensors: dict, *, reset_ledger: bool) -> None:
        _ = reset_ledger
        player_count = get_next_fleet_id(obs_tensors, self.player_count)
        self.player_count = int(player_count)
        self.fleet_buckets = torch.zeros(self.num_planets, int(self.movement_horizon), int(player_count), dtype=self.dtype, device=self.device)
        step = obs_tensors['step'].to(device=self.device, dtype=torch.long)
        self.fleet_last_step = step.detach().clone()
        M = max(1, int(self.max_tracked_fleets))
        self.max_tracked_fleets = M
        self.tracked_fleet_ids = torch.full((M,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_eta = torch.zeros((M,), dtype=torch.long, device=self.device)
        self.tracked_fleet_target_slot = torch.full((M,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_owner = torch.zeros((M,), dtype=torch.long, device=self.device)
        self.tracked_fleet_ships = torch.zeros((M,), dtype=self.dtype, device=self.device)
        if self.garrison_dirty_from is not None:
            self.garrison_dirty_from[:] = torch.minimum(self.garrison_dirty_from, torch.full_like(self.garrison_dirty_from, 1))

    def _clear_tracked_rows(self) -> None:
        if self.tracked_fleet_ids is None or self.tracked_fleet_eta is None or self.tracked_fleet_target_slot is None or (self.tracked_fleet_owner is None) or (self.tracked_fleet_ships is None):
            return
        self.tracked_fleet_ids[:] = -1
        self.tracked_fleet_eta[:] = 0
        self.tracked_fleet_target_slot[:] = -1
        self.tracked_fleet_owner[:] = 0
        self.tracked_fleet_ships[:] = 0.0

    def _ledger_bulk_insert(self, fleet_ids: Tensor, eta_remaining: Tensor, target_slots: Tensor, owners: Tensor, ships: Tensor) -> None:
        if fleet_ids.numel() == 0:
            return
        assert self.tracked_fleet_ids is not None
        assert self.tracked_fleet_eta is not None
        assert self.tracked_fleet_target_slot is not None
        assert self.tracked_fleet_owner is not None
        assert self.tracked_fleet_ships is not None
        max_tracked = int(self.tracked_fleet_ids.shape[0])
        fleet_ids = fleet_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        eta_remaining = eta_remaining.to(device=self.device, dtype=torch.long).reshape(-1)
        target_slots = target_slots.to(device=self.device, dtype=torch.long).reshape(-1)
        owners = owners.to(device=self.device, dtype=torch.long).reshape(-1)
        ships = ships.to(device=self.device, dtype=self.dtype).reshape(-1)
        valid_rows = fleet_ids >= 0
        if not bool(valid_rows.any()):
            return
        fleet_ids = fleet_ids[valid_rows]
        eta_remaining = eta_remaining[valid_rows]
        target_slots = target_slots[valid_rows]
        owners = owners[valid_rows]
        ships = ships[valid_rows]
        n = int(fleet_ids.numel())
        empty_mask = self.tracked_fleet_ids == -1
        empty_count = int(empty_mask.sum().item())
        if n > empty_count:
            occupied_count = max_tracked - empty_count
            self._grow_ledger_capacity(occupied_count + n)
            assert self.tracked_fleet_ids is not None
            empty_mask = self.tracked_fleet_ids == -1
        empty_slots = torch.nonzero(empty_mask, as_tuple=True)[0]
        slot_idx = empty_slots[:n]
        self.tracked_fleet_ids[slot_idx] = fleet_ids
        self.tracked_fleet_eta[slot_idx] = eta_remaining
        self.tracked_fleet_target_slot[slot_idx] = target_slots
        self.tracked_fleet_owner[slot_idx] = owners
        self.tracked_fleet_ships[slot_idx] = ships

    def _grow_ledger_capacity(self, required_capacity: int) -> None:
        if self.tracked_fleet_ids is None or self.tracked_fleet_eta is None or self.tracked_fleet_target_slot is None or (self.tracked_fleet_owner is None) or (self.tracked_fleet_ships is None):
            return
        old_capacity = int(self.tracked_fleet_ids.shape[0])
        target_capacity = max(int(required_capacity), old_capacity)
        if target_capacity <= old_capacity:
            return
        new_capacity = max(target_capacity, old_capacity * 2)
        old_ids = self.tracked_fleet_ids
        old_eta = self.tracked_fleet_eta
        old_tgt = self.tracked_fleet_target_slot
        old_owner = self.tracked_fleet_owner
        old_ships = self.tracked_fleet_ships
        self.tracked_fleet_ids = torch.full((new_capacity,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_eta = torch.zeros((new_capacity,), dtype=torch.long, device=self.device)
        self.tracked_fleet_target_slot = torch.full((new_capacity,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_owner = torch.zeros((new_capacity,), dtype=torch.long, device=self.device)
        self.tracked_fleet_ships = torch.zeros((new_capacity,), dtype=self.dtype, device=self.device)
        self.tracked_fleet_ids[:old_capacity] = old_ids
        self.tracked_fleet_eta[:old_capacity] = old_eta
        self.tracked_fleet_target_slot[:old_capacity] = old_tgt
        self.tracked_fleet_owner[:old_capacity] = old_owner
        self.tracked_fleet_ships[:old_capacity] = old_ships

    def _ledger_decrement_and_expire(self) -> None:
        if self.tracked_fleet_ids is None or self.tracked_fleet_eta is None or self.tracked_fleet_target_slot is None or (self.tracked_fleet_owner is None) or (self.tracked_fleet_ships is None):
            return
        valid = self.tracked_fleet_ids >= 0
        eta = torch.where(valid, self.tracked_fleet_eta - 1, self.tracked_fleet_eta)
        expire = valid & (eta <= 0)
        self.tracked_fleet_eta = eta
        self.tracked_fleet_ids = torch.where(expire, torch.full_like(self.tracked_fleet_ids, -1), self.tracked_fleet_ids)
        self.tracked_fleet_eta = torch.where(expire, torch.zeros_like(self.tracked_fleet_eta), self.tracked_fleet_eta)
        self.tracked_fleet_target_slot = torch.where(expire, torch.full_like(self.tracked_fleet_target_slot, -1), self.tracked_fleet_target_slot)
        self.tracked_fleet_owner = torch.where(expire, torch.zeros_like(self.tracked_fleet_owner), self.tracked_fleet_owner)
        self.tracked_fleet_ships = torch.where(expire, torch.zeros_like(self.tracked_fleet_ships), self.tracked_fleet_ships)

    def _roll_fleet_buckets_phase1(self, current_step: Tensor) -> None:
        if self.fleet_buckets is None or self.fleet_last_step is None:
            return
        step = current_step.to(device=self.device, dtype=torch.long)
        delta = step - self.fleet_last_step.to(device=self.device, dtype=torch.long)
        horizon = int(self.movement_horizon)
        reset = bool((delta < 0) | (step <= 0))
        if reset:
            self.fleet_buckets[:] = 0.0
            self._clear_tracked_rows()
            self._mark_garrison_dirty_all(1)
        rolled_once = not reset and bool(delta == 1)
        if rolled_once and horizon > 0:
            self.fleet_buckets[:, :-1, :] = self.fleet_buckets[:, 1:, :].clone()
            self.fleet_buckets[:, -1, :] = 0.0
            self._ledger_decrement_and_expire()
            self._mark_garrison_dirty_all(1)
        delta_bad = not reset and bool(delta > 1)
        if delta_bad:
            self._reset_fleet_tracking()
        self.fleet_last_step = step.detach().clone()

    def _reset_fleet_tracking(self) -> None:
        if self.fleet_buckets is None:
            return
        self.fleet_buckets[:] = 0.0
        self._clear_tracked_rows()
        self._mark_garrison_dirty_all(1)

    def _ingest_obs_fleets(self, obs_tensors: dict) -> None:
        if self.fleet_buckets is None or self.tracked_fleet_ids is None or int(self.movement_horizon) <= 0:
            return
        fleets = obs_tensors['fleets'].to(device=self.device, dtype=self.dtype)
        fleet_ids = fleets[..., 0].to(dtype=torch.long)
        alive = fleet_ids >= 0
        tracked = (fleet_ids.unsqueeze(1) == self.tracked_fleet_ids.unsqueeze(0)).any(dim=1)
        process_mask = alive & ~tracked
        n_alive = int(alive.sum().item())
        n_tracked = int((alive & tracked).sum().item())
        n_to_process = n_alive - n_tracked
        if n_to_process == 0:
            return
        fleet_slot = torch.where(process_mask)[0]
        proc_ids = fleet_ids[fleet_slot]
        estimate = _estimate_new_fleet_arrivals(movement=self, obs_fleets=fleets, fleet_slot=fleet_slot)
        valid_owner = (estimate['owner'] >= 0) & (estimate['owner'] < int(self.fleet_buckets.shape[2]))
        valid_hit = estimate['has_hit'] & valid_owner
        if not bool(valid_hit.any()):
            return
        buckets = self._require_fleet_buckets()
        buckets.index_put_((estimate['target_slot'][valid_hit], estimate['eta_index'][valid_hit], estimate['owner'][valid_hit]), estimate['ships'][valid_hit], accumulate=True)
        self._mark_garrison_dirty(estimate['target_slot'][valid_hit], estimate['eta_index'][valid_hit] + 1)
        eta_remaining = estimate['eta_index'][valid_hit].to(dtype=torch.long) + 1
        self._ledger_bulk_insert(proc_ids[valid_hit], eta_remaining, estimate['target_slot'][valid_hit], estimate['owner'][valid_hit], estimate['ships'][valid_hit])

    def _reconcile_obs_fleets(self, obs_tensors: dict) -> None:
        if self.fleet_buckets is None or self.tracked_fleet_ids is None or self.tracked_fleet_eta is None or (self.tracked_fleet_target_slot is None) or (self.tracked_fleet_owner is None) or (self.tracked_fleet_ships is None) or (int(self.movement_horizon) <= 0):
            return
        obs_ids = obs_tensors['fleets'][..., 0].to(device=self.device, dtype=torch.long)
        in_flight = (self.tracked_fleet_ids >= 0) & (self.tracked_fleet_eta > 0)
        if not bool(in_flight.any()):
            return
        match = (self.tracked_fleet_ids.unsqueeze(1) == obs_ids.unsqueeze(0)).any(dim=1)
        phantom = in_flight & ~match
        if not bool(phantom.any()):
            return
        m_idx = torch.where(phantom)[0]
        h_idx = (self.tracked_fleet_eta[m_idx] - 1).clamp(min=0)
        num_planets = int(self.fleet_buckets.shape[0])
        horizon = int(self.fleet_buckets.shape[1])
        num_players = int(self.fleet_buckets.shape[2])
        in_horizon = h_idx < horizon
        if not bool(in_horizon.any()):
            self.tracked_fleet_ids[m_idx] = -1
            self.tracked_fleet_eta[m_idx] = 0
            self.tracked_fleet_target_slot[m_idx] = -1
            self.tracked_fleet_owner[m_idx] = 0
            self.tracked_fleet_ships[m_idx] = 0.0
            return
        m_sel = m_idx[in_horizon]
        h_sel = h_idx[in_horizon]
        slots = self.tracked_fleet_target_slot[m_sel].clamp(min=0, max=max(num_planets - 1, 0))
        owners = self.tracked_fleet_owner[m_sel].clamp(min=0, max=max(num_players - 1, 0))
        ships = self.tracked_fleet_ships[m_sel]
        self.fleet_buckets.index_put_((slots, h_sel, owners), -ships, accumulate=True)
        self._mark_garrison_dirty(slots, h_sel + 1)
        self.tracked_fleet_ids[m_idx] = -1
        self.tracked_fleet_eta[m_idx] = 0
        self.tracked_fleet_target_slot[m_idx] = -1
        self.tracked_fleet_owner[m_idx] = 0
        self.tracked_fleet_ships[m_idx] = 0.0

    def _require_fleet_buckets(self) -> Tensor:
        if self.fleet_buckets is None:
            raise RuntimeError('StatePredictor fleet tracking is not enabled')
        return self.fleet_buckets

    def _k_index(self, k: int) -> int:
        if k < 0 or k > int(self.movement_horizon):
            raise IndexError(f'k must be in [0, {self.movement_horizon}], got {k}')
        return int(k)

    def _copy_from(self, other: StatePredictor) -> None:
        self.x = other.x
        self.y = other.y
        self.alive_by_step = other.alive_by_step
        self.planet_ids = other.planet_ids
        self.radii = other.radii
        self.planet_owner = other.planet_owner
        self.planet_ships = other.planet_ships
        self.planet_prod = other.planet_prod
        self.base_step = other.base_step
        self.comet_planet_ids = other.comet_planet_ids
        self.comet_path_index = other.comet_path_index
        self.movement_horizon = other.movement_horizon
        self.drift_epsilon = other.drift_epsilon
        self.track_fleets = other.track_fleets
        self.player_count = other.player_count
        self.max_tracked_fleets = other.max_tracked_fleets
        self.fleet_buckets = other.fleet_buckets
        self.fleet_last_step = other.fleet_last_step
        self.tracked_fleet_ids = other.tracked_fleet_ids
        self.tracked_fleet_eta = other.tracked_fleet_eta
        self.tracked_fleet_target_slot = other.tracked_fleet_target_slot
        self.tracked_fleet_owner = other.tracked_fleet_owner
        self.tracked_fleet_ships = other.tracked_fleet_ships
        self.garrison_owner_cache = other.garrison_owner_cache
        self.garrison_ships_cache = other.garrison_ships_cache
        self.garrison_dirty_from = other.garrison_dirty_from

def get_next_fleet_id(obs_tensors: dict, player_count: int | None) -> int:
    if player_count is not None:
        if int(player_count) not in (2, 4):
            raise ValueError('player_count must be 2 or 4')
        return int(player_count)
    metadata_count = obs_tensors.get('player_count')
    if metadata_count is not None:
        count = int(metadata_count.flatten()[0].item()) if isinstance(metadata_count, Tensor) else int(metadata_count)
        if count not in (2, 4):
            raise ValueError('player_count metadata must be 2 or 4')
        return count
    planets = obs_tensors['planets']
    fleets = obs_tensors['fleets']
    planet_alive = planets[..., 0] >= 0
    fleet_alive = fleets[..., 0] >= 0
    owner_values = []
    if bool(planet_alive.any()):
        owner_values.append(planets[..., 1][planet_alive].to(dtype=torch.long))
    if bool(fleet_alive.any()):
        owner_values.append(fleets[..., 1][fleet_alive].to(dtype=torch.long))
    if not owner_values:
        return 2
    owners = torch.cat(owner_values)
    owners = owners[owners >= 0]
    if owners.numel() == 0:
        return 2
    return 4 if int(owners.max().item()) >= 2 else 2

def _estimate_new_fleet_arrivals(*, movement: StatePredictor, obs_fleets: Tensor, fleet_slot: Tensor) -> dict[str, Tensor]:
    num_launches = int(fleet_slot.numel())
    device = movement.device
    dtype = movement.dtype
    horizon = int(movement.movement_horizon)
    num_planets = int(movement.num_planets)
    if num_launches == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        empty_float = torch.empty(0, dtype=dtype, device=device)
        return {'owner': empty_long, 'target_slot': empty_long, 'eta_index': empty_long, 'has_hit': empty_bool, 'ships': empty_float}
    rows = obs_fleets[fleet_slot]
    owner = rows[:, 1].to(dtype=torch.long)
    x = rows[:, 2].to(dtype=dtype)
    y = rows[:, 3].to(dtype=dtype)
    angle = rows[:, 4].to(dtype=dtype)
    ships = rows[:, 6].to(dtype=dtype)
    times = torch.arange(1, horizon + 1, dtype=dtype, device=device).view(1, horizon)
    speed = estimate_fleet_speed(ships).clamp(min=1e-06)
    ux = torch.cos(angle)
    uy = torch.sin(angle)
    old_x = x.view(num_launches, 1) + ux.view(num_launches, 1) * speed.view(num_launches, 1) * (times - 1.0)
    old_y = y.view(num_launches, 1) + uy.view(num_launches, 1) * speed.view(num_launches, 1) * (times - 1.0)
    new_x = x.view(num_launches, 1) + ux.view(num_launches, 1) * speed.view(num_launches, 1) * times
    new_y = y.view(num_launches, 1) + uy.view(num_launches, 1) * speed.view(num_launches, 1) * times
    in_bounds = (new_x >= 0.0) & (new_x <= PLAY_BOARD_SIZE) & (new_y >= 0.0) & (new_y <= PLAY_BOARD_SIZE)
    sun_dist_sq = sq_dist_point_to_line_segment(torch.full_like(new_x, PLAY_BOARD_CENTER), torch.full_like(new_y, PLAY_BOARD_CENTER), old_x, old_y, new_x, new_y)
    env_kill = ~in_bounds | (sun_dist_sq < SOLAR_SUN_RADIUS * SOLAR_SUN_RADIUS)
    planet_x = movement.x.unsqueeze(0).expand(num_launches, horizon + 1, num_planets)
    planet_y = movement.y.unsqueeze(0).expand(num_launches, horizon + 1, num_planets)
    planet_alive = movement.alive_by_step.unsqueeze(0).expand(num_launches, horizon + 1, num_planets)
    radii = movement.radii.unsqueeze(0).expand(num_launches, num_planets).to(dtype=dtype)
    old_px = planet_x[:, :-1, :]
    old_py = planet_y[:, :-1, :]
    new_px = planet_x[:, 1:, :]
    new_py = planet_y[:, 1:, :]
    alive_old = planet_alive[:, :-1, :]
    check_collision = alive_old & (old_px >= 0.0) & (old_py >= 0.0)
    swept_collides = check_swept_collision_batched(old_x.unsqueeze(2), old_y.unsqueeze(2), new_x.unsqueeze(2), new_y.unsqueeze(2), old_px, old_py, new_px, new_py, radii.view(num_launches, 1, num_planets)) & check_collision
    step_raw_has_hit = swept_collides.any(dim=2)
    hit_rank = swept_collides.to(torch.int32).cumsum(dim=2)
    first_hit = swept_collides & (hit_rank == 1)
    step_hit_slot = first_hit.to(torch.int64).argmax(dim=2)
    step_hit_slot = step_hit_slot.where(step_raw_has_hit, torch.full_like(step_hit_slot, -1))
    kill_event = step_raw_has_hit | env_kill
    cum_kill_inclusive = kill_event.cummax(dim=1).values
    alive_before_t = torch.cat([torch.ones((num_launches, 1), dtype=torch.bool, device=device), ~cum_kill_inclusive[:, :-1]], dim=1)
    step_has_hit = step_raw_has_hit & alive_before_t
    has_hit = step_has_hit.any(dim=1)
    eta_index = step_has_hit.to(torch.int64).argmax(dim=1)
    target_slot = step_hit_slot.gather(1, eta_index.view(num_launches, 1)).squeeze(1).clamp(min=0, max=max(num_planets - 1, 0))
    return {'owner': owner, 'target_slot': target_slot, 'eta_index': eta_index, 'has_hit': has_hit, 'ships': ships}

def sq_dist_point_to_line_segment(px: Tensor, py: Tensor, x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor) -> Tensor:
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    t = ((px - x1) * dx + (py - y1) * dy) / safe_denom
    t = t.clamp(0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return (px - proj_x) ** 2 + (py - proj_y) ** 2

def check_swept_collision_batched(ax: Tensor, ay: Tensor, bx: Tensor, by: Tensor, p0x: Tensor, p0y: Tensor, p1x: Tensor, p1y: Tensor, r: Tensor) -> Tensor:
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = bx - ax - (p1x - p0x)
    dvy = by - ay - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    near_static = a < 1e-12
    c_hit = c <= 0.0
    disc = b * b - 4.0 * a * c
    has_root = disc >= 0.0
    safe_a = torch.where(near_static, torch.ones_like(a), a)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    quad_hit = has_root & (t2 >= 0.0) & (t1 <= 1.0)
    return torch.where(near_static, c_hit, quad_hit)

def forecast_planet_positions(obs_tensors: dict, movement_horizon: int, *, offsets: Tensor | None = None) -> dict[str, Tensor]:
    obs = extract_game_state(obs_tensors)
    horizon = int(movement_horizon)
    planets = obs_tensors['planets']
    dtype = planets.dtype
    device = planets.device
    num_planets, _ = planets.shape
    planet_ids = planets[..., 0].long()
    radii = planets[..., 4].to(dtype=dtype)
    owner = planets[..., 1].to(device=device, dtype=torch.long)
    owner = torch.where(obs.alive, owner, torch.full_like(owner, -1))
    ships = planets[..., 5].to(device=device, dtype=dtype)
    prod = planets[..., 6].to(device=device, dtype=dtype)
    step = obs.step.to(device=device, dtype=torch.long)
    if offsets is None:
        offsets_long = torch.arange(horizon + 1, dtype=torch.long, device=device)
    else:
        offsets_long = offsets.to(device=device, dtype=torch.long).reshape(-1)
    num_offsets = int(offsets_long.shape[0])
    offsets_d = offsets_long.to(dtype=dtype)
    future_phase = calculate_orbit_phase(obs.step.to(dtype=dtype) + offsets_d).to(device=device, dtype=dtype)
    angle = obs.orb_a0.to(dtype=dtype).view(1, num_planets) + obs.angvel.to(dtype=dtype) * future_phase.view(num_offsets, 1)
    orb_x = PLAY_BOARD_CENTER + obs.orb_r.to(dtype=dtype).view(1, num_planets) * torch.cos(angle)
    orb_y = PLAY_BOARD_CENTER + obs.orb_r.to(dtype=dtype).view(1, num_planets) * torch.sin(angle)
    is_orbiting = obs.is_orbiting.view(1, num_planets)
    x = torch.where(is_orbiting, orb_x, obs.x.to(dtype=dtype).view(1, num_planets).expand(num_offsets, num_planets)).contiguous()
    y = torch.where(is_orbiting, orb_y, obs.y.to(dtype=dtype).view(1, num_planets).expand(num_offsets, num_planets)).contiguous()
    alive_by_step = obs.alive.view(1, num_planets).expand(num_offsets, num_planets).clone()
    comet_planet_ids, comet_path_index = get_comet_info(obs_tensors, device)
    x, y, alive_by_step = project_comet_trajectories(
        x=x, y=y, alive_by_step=alive_by_step, planet_ids=planet_ids,
        comet_planet_ids=comet_planet_ids, comet_path_index=comet_path_index,
        obs_tensors=obs_tensors, offsets=offsets_long
    )
    zero_idx = (offsets_long == 0).nonzero(as_tuple=True)[0]
    if int(zero_idx.numel()) > 0:
        x[zero_idx, :] = obs.x.to(dtype=dtype).view(1, num_planets)
        y[zero_idx, :] = obs.y.to(dtype=dtype).view(1, num_planets)
        alive_by_step[zero_idx, :] = obs.alive.view(1, num_planets)
    return {
        'x': x, 'y': y, 'alive_by_step': alive_by_step, 'planet_ids': planet_ids,
        'radii': radii, 'owner': owner, 'ships': ships, 'prod': prod, 'step': step,
        'comet_planet_ids': comet_planet_ids, 'comet_path_index': comet_path_index, '_offsets': offsets_long
    }

def get_comet_info(obs_tensors: dict, device: torch.device) -> tuple[Tensor, Tensor]:
    comets = obs_tensors.get('comets') or {}
    comet_ids = comets.get('planet_ids')
    if comet_ids is None:
        flat_ids = obs_tensors.get('comet_planet_ids')
        if flat_ids is None:
            flat_ids = torch.full((0,), -1, dtype=torch.long, device=device)
        else:
            flat_ids = flat_ids.to(device=device, dtype=torch.long)
        path_index = torch.full((0,), -1, dtype=torch.long, device=device)
        return (flat_ids, path_index)
    comet_ids = comet_ids.to(device=device, dtype=torch.long)
    flat_ids = comet_ids.reshape(-1)
    path_index = comets.get('path_index')
    if path_index is None:
        path_index = torch.full((comet_ids.shape[0],), -1, dtype=torch.long, device=device)
    else:
        path_index = path_index.to(device=device, dtype=torch.long)
    return (flat_ids, path_index)

def project_comet_trajectories(*, x: Tensor, y: Tensor, alive_by_step: Tensor, planet_ids: Tensor, comet_planet_ids: Tensor, comet_path_index: Tensor, obs_tensors: dict, offsets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    comets = obs_tensors.get('comets') or {}
    paths = comets.get('paths')
    ids_grid = comets.get('planet_ids')
    if paths is None or ids_grid is None or comet_planet_ids.numel() == 0:
        return (x, y, alive_by_step)
    num_offsets, num_planets = x.shape
    paths = paths.to(device=x.device, dtype=x.dtype)
    ids_grid = ids_grid.to(device=x.device, dtype=torch.long)
    num_events = int(ids_grid.shape[0])
    comets_per_group = int(ids_grid.shape[1])
    max_path_len = int(paths.shape[2])
    if num_events == 0 or comets_per_group == 0 or max_path_len == 0:
        return (x, y, alive_by_step)
    flat_ids = ids_grid.reshape(num_events * comets_per_group)
    matches = (planet_ids.unsqueeze(1) == flat_ids.unsqueeze(0)) & (flat_ids.unsqueeze(0) >= 0)
    is_comet = matches.any(dim=1)
    flat_slot = matches.to(torch.float32).argmax(dim=1).long()
    flat_paths_x = paths[..., 0].reshape(num_events * comets_per_group, max_path_len)
    flat_paths_y = paths[..., 1].reshape(num_events * comets_per_group, max_path_len)
    path_x_by_slot = flat_paths_x[flat_slot]
    path_y_by_slot = flat_paths_y[flat_slot]
    finite = torch.isfinite(flat_paths_x)
    path_len = finite.sum(dim=1).to(dtype=torch.long)
    len_by_slot = path_len[flat_slot]
    group_idx = (flat_slot // comets_per_group).clamp(min=0, max=max(num_events - 1, 0))
    path_idx_by_slot = comet_path_index[group_idx]
    offsets_v = offsets.to(device=x.device, dtype=torch.long).view(num_offsets, 1)
    future_idx = path_idx_by_slot.view(1, num_planets) + offsets_v
    valid_future = is_comet.view(1, num_planets) & (future_idx >= 0) & (future_idx < len_by_slot.view(1, num_planets))
    idx_clamped = future_idx.clamp(min=0, max=max(max_path_len - 1, 0))
    p_index = torch.arange(num_planets, device=x.device).view(1, num_planets).expand(num_offsets, num_planets)
    comet_x = path_x_by_slot[p_index, idx_clamped]
    comet_y = path_y_by_slot[p_index, idx_clamped]
    x = torch.where(valid_future, comet_x, x)
    y = torch.where(valid_future, comet_y, y)
    alive_by_step = torch.where(is_comet.view(1, num_planets), valid_future, alive_by_step)
    return (x, y, alive_by_step)

def are_tensors_equal(a: Tensor, b: Tensor) -> bool:
    if a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    return bool((a == b.to(device=a.device, dtype=a.dtype)).all())

def validate_position_drift(pred_x: Tensor, pred_y: Tensor, cur_x: Tensor, cur_y: Tensor, alive: Tensor, epsilon: float) -> bool:
    diff = torch.maximum((pred_x - cur_x).abs(), (pred_y - cur_y).abs())
    diff = torch.where(alive, diff, torch.zeros_like(diff))
    return bool((diff <= float(epsilon)).all())

# Tinh toan chênh lệch dòng quân để đánh giá lựa chọn phóng tàu
# Phân tích xem nếu phóng quân, dòng quân ròng thay đổi như thế nào.

@dataclass(frozen=True)
class FleetLaunchGroup:
    source_slots: Tensor
    target_slots: Tensor
    ships: Tensor
    eta: Tensor
    owner: Tensor
    valid: Tensor

    @property
    def has_candidate_axis(self) -> bool:
        return self.source_slots.dim() >= 2

def resolve_combat_survivors(arrivals: Tensor) -> tuple[Tensor, Tensor]:
    num_players = int(arrivals.shape[-1])
    if num_players >= 2:
        top2 = arrivals.topk(k=2, dim=-1)
        top_ships = top2.values[..., 0]
        second_ships = top2.values[..., 1]
        top_owner = top2.indices[..., 0].to(dtype=torch.long)
    else:
        top_ships, top_owner = arrivals.max(dim=-1)
        second_ships = torch.zeros_like(top_ships)
        top_owner = top_owner.to(dtype=torch.long)
    tied = top_ships == second_ships
    survivor_ships = torch.where(tied, torch.zeros_like(top_ships), (top_ships - second_ships).clamp(min=0.0))
    return (top_owner, survivor_ships)

def simulate_combat_recurrence(*, init_owner: Tensor, init_ships: Tensor, prod: Tensor, alive: Tensor, arrivals: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    num_candidates, num_planets = init_owner.shape
    horizon = int(arrivals.shape[2])
    device = init_ships.device
    owner_out = torch.empty(num_candidates, num_planets, horizon + 1, dtype=init_owner.dtype, device=device)
    ships_out = torch.empty(num_candidates, num_planets, horizon + 1, dtype=init_ships.dtype, device=device)
    pre_owner_out = torch.empty_like(owner_out)
    pre_ships_out = torch.empty_like(ships_out)
    owner_out[..., 0] = init_owner
    ships_out[..., 0] = init_ships
    pre_owner_out[..., 0] = init_owner
    pre_ships_out[..., 0] = init_ships
    survivor_owner, survivor_ships = resolve_combat_survivors(arrivals)
    state_owner = init_owner.clone()
    state_ships = init_ships.clone()
    zero_ships = torch.zeros((), dtype=state_ships.dtype, device=device)
    neg_one = torch.full((), -1, dtype=state_owner.dtype, device=device)
    zero_prod = torch.zeros((), dtype=prod.dtype, device=device)
    for k in range(1, horizon + 1):
        a_before = alive[..., k - 1]
        a_now = alive[..., k]
        s_owner = survivor_owner[..., k - 1]
        s_ships = survivor_ships[..., k - 1]
        produces = a_before & (state_owner >= 0)
        state_ships = state_ships + torch.where(produces, prod, zero_prod)
        pre_owner_out[..., k] = torch.where(a_now, state_owner, neg_one)
        pre_ships_out[..., k] = torch.where(a_now, state_ships, zero_ships)
        has_combat = (s_ships > 0.0) & a_now
        same = state_owner == s_owner
        diff = state_ships - s_ships
        attacker_wins = ~same & (diff < 0.0)
        combat_ships = torch.where(same, state_ships + s_ships, diff.abs())
        combat_owner = torch.where(attacker_wins, s_owner, state_owner)
        state_ships = torch.where(has_combat, combat_ships, state_ships)
        state_owner = torch.where(has_combat, combat_owner, state_owner)
        state_owner = torch.where(a_now, state_owner, neg_one)
        state_ships = torch.where(a_now, state_ships, zero_ships)
        owner_out[..., k] = state_owner
        ships_out[..., k] = state_ships
    return (owner_out, ships_out, pre_owner_out, pre_ships_out)

def check_simulation_dimensions(status: GarrisonPrediction, prod: Tensor, alive_by_step: Tensor, player_count: int) -> tuple[int, int, int]:
    if status.arrivals_by_owner is None:
        raise ValueError('garrison status must carry arrivals_by_owner (build it from a StatePredictor with track_fleets=True)')
    if status.pre_combat_owner is None or status.pre_combat_ships is None:
        raise ValueError('garrison status must carry pre_combat_owner/ships')
    if status.owner.dim() != 2:
        raise ValueError(f'expected a full-board status with owner shaped [num_planets, horizon+1]; got {tuple(status.owner.shape)}')
    num_planets, H1 = status.owner.shape
    horizon = H1 - 1
    num_players = int(status.arrivals_by_owner.shape[-1])
    if int(player_count) != num_players:
        raise ValueError(f'player_count={player_count} disagrees with arrivals owner axis num_players={num_players}')
    if tuple(prod.shape) != (num_planets,):
        raise ValueError(f'prod must be [num_planets]=({num_planets},); got {tuple(prod.shape)}')
    if tuple(alive_by_step.shape) != (H1, num_planets):
        raise ValueError(f'alive_by_step must be [horizon+1, num_planets]=({H1}, {num_planets}); got {tuple(alive_by_step.shape)}')
    return (num_planets, horizon, num_players)

@dataclass(frozen=True)
class FlowDelta:
    player_id: int
    ships_produced_current: Tensor
    ships_produced_hypothetical: Tensor
    ships_produced_delta: Tensor
    ships_lost_combat_current: Tensor
    ships_lost_combat_hypothetical: Tensor
    ships_lost_combat_delta: Tensor
    net_ship_delta: Tensor

    @property
    def player_count(self) -> int:
        return int(self.ships_produced_delta.shape[-1])

def calculate_planet_flow_stats(*, owner: Tensor, pre_owner: Tensor, pre_ships: Tensor, arr_full: Tensor, prod: Tensor, alive_pmajor: Tensor) -> tuple[Tensor, Tensor]:
    num_players = int(arr_full.shape[-1])
    horizon = int(owner.shape[-1]) - 1
    fdtype = pre_ships.dtype
    a_idx = torch.arange(num_players, device=owner.device)
    producing_owner = owner[..., :horizon]
    amount = prod.unsqueeze(-1) * alive_pmajor[..., :horizon].to(fdtype)
    prod_owner_oh = producing_owner.unsqueeze(-1) == a_idx
    produced = (amount.unsqueeze(-1) * prod_owner_oh.to(fdtype)).sum(dim=-2)
    arr_k = arr_full[..., 1:, :]
    survivor_owner, survivor_ships = resolve_combat_survivors(arr_k)
    survived = torch.where(a_idx == survivor_owner.unsqueeze(-1), survivor_ships.unsqueeze(-1), torch.zeros_like(survivor_ships).unsqueeze(-1))
    attacker_lost = (arr_k - survived).clamp(min=0.0)
    prior_owner = pre_owner[..., 1:]
    prior_ships = pre_ships[..., 1:]
    fights_garrison = (survivor_ships > 0.0) & (survivor_owner != prior_owner) & (survivor_owner >= 0)
    garrison_loss = torch.where(fights_garrison, torch.minimum(prior_ships, survivor_ships), torch.zeros_like(prior_ships))
    is_survivor = (a_idx == survivor_owner.unsqueeze(-1)) & fights_garrison.unsqueeze(-1)
    is_prior = (a_idx == prior_owner.unsqueeze(-1)) & fights_garrison.unsqueeze(-1) & (prior_owner >= 0).unsqueeze(-1)
    garrison_lost = garrison_loss.unsqueeze(-1) * (is_survivor.to(fdtype) + is_prior.to(fdtype))
    combat_lost = (attacker_lost + garrison_lost).sum(dim=-2)
    return (produced, combat_lost)

def standardize_launch_axes(launches: FleetLaunchGroup) -> tuple[Tensor, ...]:
    fields = (launches.source_slots, launches.target_slots, launches.ships, launches.eta, launches.owner, launches.valid)
    if launches.has_candidate_axis:
        return fields
    return tuple((f.unsqueeze(0) for f in fields))

def compute_hypothetical_flow_delta(status: GarrisonPrediction, *, prod: Tensor, alive_by_step: Tensor, player_count: int, launches: FleetLaunchGroup, player_id: int = 0) -> FlowDelta:
    num_planets, horizon, num_players = check_simulation_dimensions(status, prod, alive_by_step, player_count)
    device = status.owner.device
    fdtype = status.ships.dtype
    assert status.pre_combat_owner is not None and status.pre_combat_ships is not None
    assert status.arrivals_by_owner is not None
    src, tgt, ships, eta, owner, valid = standardize_launch_axes(launches)
    num_candidates = int(src.shape[0])
    num_launches = int(src.shape[-1])
    src = src.to(device=device, dtype=torch.long)
    tgt = tgt.to(device=device, dtype=torch.long)
    ships = ships.to(device=device, dtype=fdtype)
    owner = owner.to(device=device, dtype=torch.long)
    valid = valid.to(device=device, dtype=torch.bool)
    h_idx = torch.ceil(eta.to(device=device, dtype=fdtype)).to(torch.long) - 1
    valid_t = valid & (ships > 0) & (tgt >= 0) & (tgt < num_planets) & (owner >= 0) & (owner < num_players) & (h_idx >= 0) & (h_idx < horizon)
    valid_s = valid & (ships > 0) & (src >= 0) & (src < num_planets)
    src_safe = src.clamp(0, max(num_planets - 1, 0))
    tgt_safe = tgt.clamp(0, max(num_planets - 1, 0))
    affected = torch.zeros(num_candidates, num_planets, dtype=fdtype, device=device)
    affected.scatter_add_(1, src_safe, valid_s.to(fdtype))
    affected.scatter_add_(1, tgt_safe, valid_t.to(fdtype))
    affected_mask = affected > 0
    base_prod_pp, base_combat_pp = calculate_planet_flow_stats(
        owner=status.owner, pre_owner=status.pre_combat_owner,
        pre_ships=status.pre_combat_ships, arr_full=status.arrivals_by_owner,
        prod=prod, alive_pmajor=alive_by_step.permute(1, 0)
    )
    base_prod = base_prod_pp.sum(dim=0)
    base_combat = base_combat_pp.sum(dim=0)
    produced_delta = torch.zeros(num_candidates, num_players, dtype=fdtype, device=device)
    combat_delta = torch.zeros(num_candidates, num_players, dtype=fdtype, device=device)
    if bool(affected_mask.any()):
        c_aff, p_aff = affected_mask.nonzero(as_tuple=True)
        num_cells = int(c_aff.numel())
        cell_id = torch.full((num_candidates, num_planets), -1, dtype=torch.long, device=device)
        cell_id[c_aff, p_aff] = torch.arange(num_cells, device=device)
        debit_cp = torch.zeros(num_candidates, num_planets, dtype=fdtype, device=device)
        debit_cp.scatter_add_(1, src_safe, torch.where(valid_s, ships, torch.zeros_like(ships)))
        debit_aff = debit_cp[c_aff, p_aff]
        arr_aff = torch.zeros(num_cells, horizon, num_players, dtype=fdtype, device=device)
        launch_cell = cell_id.gather(1, tgt_safe)
        m = valid_t
        cells, hh, oo, ss = (launch_cell[m], h_idx[m], owner[m], ships[m])
        ok = cells >= 0
        arr_aff.index_put_((cells[ok], hh[ok], oo[ok]), ss[ok], accumulate=True)
        base_arr_k = status.arrivals_by_owner[..., 1:, :]
        arrivals_cell = base_arr_k[p_aff] + arr_aff
        init_owner = status.owner[p_aff, 0]
        init_ships = (status.ships[p_aff, 0] - debit_aff).clamp(min=0.0)
        prod_aff = prod[p_aff]
        alive_aff = alive_by_step[:, p_aff].transpose(0, 1)
        o_t, _s_t, po_t, ps_t = simulate_combat_recurrence(
            init_owner=init_owner.unsqueeze(1), init_ships=init_ships.unsqueeze(1),
            prod=prod_aff.unsqueeze(1), alive=alive_aff.unsqueeze(1), arrivals=arrivals_cell.unsqueeze(1)
        )
        zero_frame = torch.zeros(num_cells, 1, 1, num_players, dtype=fdtype, device=device)
        arr_full_cell = torch.cat([zero_frame, arrivals_cell.unsqueeze(1)], dim=-2)
        hyp_prod_pp, hyp_combat_pp = calculate_planet_flow_stats(
            owner=o_t, pre_owner=po_t, pre_ships=ps_t, arr_full=arr_full_cell,
            prod=prod_aff.unsqueeze(1), alive_pmajor=alive_aff.unsqueeze(1)
        )
        dprod = hyp_prod_pp.squeeze(1) - base_prod_pp[p_aff]
        dcombat = hyp_combat_pp.squeeze(1) - base_combat_pp[p_aff]
        produced_delta.index_put_((c_aff,), dprod, accumulate=True)
        combat_delta.index_put_((c_aff,), dcombat, accumulate=True)
    produced_current = base_prod.unsqueeze(0)
    combat_current = base_combat.unsqueeze(0)
    diff = FlowDelta(
        player_id=int(player_id), ships_produced_current=produced_current,
        ships_produced_hypothetical=produced_current + produced_delta, ships_produced_delta=produced_delta,
        ships_lost_combat_current=combat_current, ships_lost_combat_hypothetical=combat_current + combat_delta,
        ships_lost_combat_delta=combat_delta, net_ship_delta=produced_delta - combat_delta
    )
    if not launches.has_candidate_axis:
        def _sq(t: Tensor) -> Tensor:
            return t.squeeze(0)
        diff = FlowDelta(
            player_id=diff.player_id, ships_produced_current=base_prod,
            ships_produced_hypothetical=_sq(diff.ships_produced_hypothetical), ships_produced_delta=_sq(diff.ships_produced_delta),
            ships_lost_combat_current=base_combat, ships_lost_combat_hypothetical=_sq(diff.ships_lost_combat_hypothetical),
            ships_lost_combat_delta=_sq(diff.ships_lost_combat_delta), net_ship_delta=_sq(diff.net_ship_delta)
        )
    return diff

# Bộ nhớ đệm lưu khoảng cách liên thời gian giữa các hành tinh.

@dataclass
class SpatialDistanceCache:
    cross_dist: Tensor
    alive_by_step: Tensor
    K: int

    @property
    def num_planets(self) -> int:
        return int(self.cross_dist.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.cross_dist.device

    @property
    def dtype(self) -> torch.dtype:
        return self.cross_dist.dtype

def initialize_distance_matrix(movement: StatePredictor, *, max_k: int) -> SpatialDistanceCache:
    K = max(0, min(int(max_k), int(movement.movement_horizon)))
    num_planets = int(movement.num_planets)
    src_x0 = movement.x[0]
    src_y0 = movement.y[0]
    tgt_x = movement.x[:K + 1]
    tgt_y = movement.y[:K + 1]
    dx = src_x0.view(1, num_planets, 1) - tgt_x.unsqueeze(1)
    dy = src_y0.view(1, num_planets, 1) - tgt_y.unsqueeze(1)
    cross_dist = torch.sqrt((dx * dx + dy * dy).clamp(min=0.0))
    alive_by_step = movement.alive_by_step[:K + 1]
    return SpatialDistanceCache(cross_dist=cross_dist, alive_by_step=alive_by_step, K=K)

def find_closest_target_distance(cache: SpatialDistanceCache, source_mask: Tensor, target_mask: Tensor, *, max_k: int) -> Tensor:
    if source_mask.shape[-1] != cache.num_planets or target_mask.shape[-1] != cache.num_planets:
        raise ValueError('source_mask and target_mask must have shape [num_planets]')
    K = max(0, min(int(max_k), int(cache.K)))
    if K <= 0:
        return torch.zeros(cache.num_planets, dtype=cache.dtype, device=cache.device)
    cross = cache.cross_dist[1:K + 1].clone()
    alive_steps = cache.alive_by_step[1:K + 1]
    src_mask = source_mask.to(device=cache.device, dtype=torch.bool)
    tgt_mask = target_mask.to(device=cache.device, dtype=torch.bool)
    inf_v = float('inf')
    cross.masked_fill_(~alive_steps.unsqueeze(1), inf_v)
    cross.masked_fill_(~src_mask.view(1, cache.num_planets, 1), inf_v)
    cross.masked_fill_(~tgt_mask.view(1, 1, cache.num_planets), inf_v)
    best_per_target = cross.amin(dim=(0, 1))
    return torch.where(torch.isfinite(best_per_target), best_per_target, torch.zeros_like(best_per_target))

# Tính toán góc chặn tối ưu cho hành tinh đang di chuyển trên quỹ đạo.

_FP_ITERS = 6
_BIG = 1000000.0

def calculate_intercept_angle(movement: StatePredictor, source_slots: Tensor, target_slots: Tensor, fleet_sizes: Tensor, *, fp_iters: int = _FP_ITERS, active: Tensor | None = None) -> dict[str, Tensor]:
    dev = movement.device
    dt = movement.dtype
    horizon = int(movement.movement_horizon)
    src, tgt, ships = torch.broadcast_tensors(source_slots.to(device=dev), target_slots.to(device=dev), fleet_sizes.to(device=dev, dtype=dt))
    shape = src.shape
    src = src.long().clamp(0, max(movement.num_planets - 1, 0)).reshape(-1)
    tgt = tgt.long().clamp(0, max(movement.num_planets - 1, 0)).reshape(-1)
    ships = ships.to(dt).clamp(min=1.0).reshape(-1)
    num_launches = src.shape[0]
    sx, sy = movement.position_at_slots(src, 0)
    src_r = movement.radii[src]
    tgt_r = movement.radii[tgt]
    speed = estimate_fleet_speed(ships).clamp(min=1e-06)
    t0x, t0y = movement.position_at_slots(tgt, 0)
    t1x, t1y = movement.position_at_slots(tgt, 1)
    R = torch.sqrt(((t0x - PLAY_BOARD_CENTER) ** 2 + (t0y - PLAY_BOARD_CENTER) ** 2).clamp(min=0.0))
    a0 = torch.atan2(t0y - PLAY_BOARD_CENTER, t0x - PLAY_BOARD_CENTER)
    a1 = torch.atan2(t1y - PLAY_BOARD_CENTER, t1x - PLAY_BOARD_CENTER)
    omega = torch.atan2(torch.sin(a1 - a0), torch.cos(a1 - a0))
    gap = src_r + LAUNCH_SURFACE_OFFSET + tgt_r + TARGET_HIT_SURFACE_OFFSET

    def target_pos(t: Tensor):
        ang = a0 + omega * t
        return (PLAY_BOARD_CENTER + R * torch.cos(ang), PLAY_BOARD_CENTER + R * torch.sin(ang))

    d0 = torch.sqrt(((t0x - sx) ** 2 + (t0y - sy) ** 2).clamp(min=0.0))
    t_star = ((d0 - gap) / speed).clamp(min=0.0, max=float(horizon))
    for _ in range(int(fp_iters)):
        tx, ty = target_pos(t_star)
        d = torch.sqrt(((tx - sx) ** 2 + (ty - sy) ** 2).clamp(min=0.0))
        t_star = ((d - gap) / speed).clamp(min=0.0, max=float(horizon))
    tx, ty = target_pos(t_star)
    angle = torch.atan2(ty - sy, tx - sx)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    launch_x = sx + cos_a * (src_r + LAUNCH_SURFACE_OFFSET)
    launch_y = sy + sin_a * (src_r + LAUNCH_SURFACE_OFFSET)
    eta_cap = (t_star + 2.0).clamp(max=float(horizon))
    seg_len = speed * eta_cap + tgt_r + 2.0
    px = movement.x[:horizon + 1, :]
    py = movement.y[:horizon + 1, :]
    radii_p = movement.radii
    alive0 = movement.alive_at(0)
    if active is None:
        contact, eta_c = detect_first_collision(launch_x=launch_x, launch_y=launch_y, cos_a=cos_a, sin_a=sin_a, speed=speed, px=px, py=py, p_alive0=alive0, radii=radii_p, H=horizon, seg_len=seg_len)
    else:
        act = active.broadcast_to(shape).reshape(num_launches).to(torch.bool)
        n_max = max(1, int(act.sum().item()))
        order = (~act).to(torch.int8).argsort(stable=True)
        midx = order[:n_max]
        keep = act[midx]
        contact_m, eta_cm = detect_first_collision(launch_x=launch_x[midx], launch_y=launch_y[midx], cos_a=cos_a[midx], sin_a=sin_a[midx], speed=speed[midx], px=px, py=py, p_alive0=alive0, radii=radii_p, H=horizon, seg_len=seg_len[midx])
        contact = torch.full((num_launches,), -1, dtype=contact_m.dtype, device=dev)
        eta_c = torch.full((num_launches,), float(horizon), dtype=eta_cm.dtype, device=dev)
        contact[midx] = torch.where(keep, contact_m, torch.full_like(contact_m, -1))
        eta_c[midx] = torch.where(keep, eta_cm, torch.full_like(eta_cm, float(horizon)))
    viable = contact == tgt
    eta_out = torch.where(viable, eta_c.to(dt), torch.full_like(eta_c.to(dt), float('inf')))
    return {'angle': angle.reshape(shape), 'eta': eta_out.reshape(shape), 'viable': viable.reshape(shape)}

def detect_first_collision(*, launch_x: Tensor, launch_y: Tensor, cos_a: Tensor, sin_a: Tensor, speed: Tensor, px: Tensor, py: Tensor, p_alive0: Tensor, radii: Tensor, H: int, seg_len: Tensor | None = None, max_bytes: int = 256 * 1024 * 1024):
    num_launches = cos_a.shape[0]
    num_planets = px.shape[-1]
    dev = cos_a.device
    dt = launch_x.dtype
    N = num_launches
    big = _BIG
    lx = launch_x.reshape(N)
    ly = launch_y.reshape(N)
    ca = cos_a.reshape(N)
    sa = sin_a.reshape(N)
    sp = speed.reshape(N)
    slen = sp * float(H) if seg_len is None else seg_len.reshape(N)
    end_x = lx + ca * slen
    end_y = ly + sa * slen
    seg_xmin = torch.minimum(lx, end_x)
    seg_xmax = torch.maximum(lx, end_x)
    seg_ymin = torch.minimum(ly, end_y)
    seg_ymax = torch.maximum(ly, end_y)
    bb_xmin = px.amin(0) - radii
    bb_xmax = px.amax(0) + radii
    bb_ymin = py.amin(0) - radii
    bb_ymax = py.amax(0) + radii
    keep = ~((seg_xmax.unsqueeze(1) < bb_xmin) | (seg_xmin.unsqueeze(1) > bb_xmax) | (seg_ymax.unsqueeze(1) < bb_ymin) | (seg_ymin.unsqueeze(1) > bb_ymax))
    K = max(1, int(keep.sum(1).amax().item()))
    order = (~keep).to(torch.int8).argsort(dim=1, stable=True)
    shortlist = order[:, :K]
    valid = keep.gather(1, shortlist)
    k = torch.arange(H + 1, device=dev, dtype=dt)
    t_ax = torch.arange(H + 1, device=dev).view(1, H + 1, 1)
    step_h = torch.arange(1, H + 1, device=dev, dtype=dt).view(1, H, 1)
    bytes_per = max(1, 16 * H * K * 4)
    chunk = max(4096, max_bytes // bytes_per)
    chunk = min(chunk, max(N, 1))
    contacts: list[Tensor] = []
    etas: list[Tensor] = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sl = shortlist[s:e]
        fx = lx[s:e].view(-1, 1) + ca[s:e].view(-1, 1) * sp[s:e].view(-1, 1) * k
        fy = ly[s:e].view(-1, 1) + sa[s:e].view(-1, 1) * sp[s:e].view(-1, 1) * k
        sl_e = sl.view(-1, 1, K)
        pxc = px[t_ax, sl_e]
        pyc = py[t_ax, sl_e]
        radc = radii[sl]
        alivec = p_alive0[sl] & valid[s:e]
        real_slot = sl.to(dt)
        fx0 = fx[:, :-1].unsqueeze(-1)
        fy0 = fy[:, :-1].unsqueeze(-1)
        fx1 = fx[:, 1:].unsqueeze(-1)
        fy1 = fy[:, 1:].unsqueeze(-1)
        hit = check_swept_collision(fx0, fy0, fx1, fy1, pxc[:, :-1, :], pyc[:, :-1, :], pxc[:, 1:, :], pyc[:, 1:, :], radc.unsqueeze(1))
        hit = hit & alivec.unsqueeze(1)
        planet_hit_step = torch.where(hit, step_h, torch.full_like(step_h, big)).amin(1)
        first_planet_step = planet_hit_step.amin(1)
        is_first = planet_hit_step == first_planet_step.unsqueeze(-1)
        contact_planet = torch.where(is_first, real_slot, torch.full_like(real_slot, big)).amin(1)
        nfx = fx[:, 1:]
        nfy = fy[:, 1:]
        ofx = fx[:, :-1]
        ofy = fy[:, :-1]
        oob = (nfx < 0) | (nfx > PLAY_BOARD_SIZE) | (nfy < 0) | (nfy > PLAY_BOARD_SIZE)
        vx = nfx - ofx
        vy = nfy - ofy
        wx = PLAY_BOARD_CENTER - ofx
        wy = PLAY_BOARD_CENTER - ofy
        vv = (vx * vx + vy * vy).clamp(min=1e-12)
        t = ((wx * vx + wy * vy) / vv).clamp(0.0, 1.0)
        cxp = ofx + t * vx
        cyp = ofy + t * vy
        sun = (cxp - PLAY_BOARD_CENTER) ** 2 + (cyp - PLAY_BOARD_CENTER) ** 2 < SOLAR_SUN_RADIUS * SOLAR_SUN_RADIUS
        env = oob | sun
        death_step = torch.where(env, step_h.squeeze(-1), torch.full_like(env, big, dtype=dt)).amin(1)
        ht = (first_planet_step <= death_step) & (first_planet_step < big)
        contacts.append(torch.where(ht, contact_planet, torch.full_like(contact_planet, -1.0)).long())
        etas.append(torch.where(ht, first_planet_step, torch.full_like(first_planet_step, float(H))))
    contact = (contacts[0] if len(contacts) == 1 else torch.cat(contacts)).view(num_launches)
    eta = (etas[0] if len(etas) == 1 else torch.cat(etas)).view(num_launches)
    return (contact, eta)

@dataclass(frozen=True)
class ScheduledMissions:
    source_slots: Tensor
    angle: Tensor
    ships: Tensor
    target_slots: Tensor
    eta_turns: Tensor
    valid: Tensor
    fleet_ids: Tensor

@dataclass(frozen=True)
class MissionEntries:
    source_slots: Tensor
    target_slots: Tensor
    ships: Tensor
    angle: Tensor
    eta: Tensor
    valid: Tensor

    @property
    def width(self) -> int:
        return int(self.source_slots.shape[0])

def merge_mission_entries(entries: Sequence[MissionEntries]) -> MissionEntries:
    if not entries:
        raise ValueError('merge_mission_entries requires at least one entry table')
    if len(entries) == 1:
        return entries[0]
    return MissionEntries(
        source_slots=torch.cat([e.source_slots for e in entries], dim=0),
        target_slots=torch.cat([e.target_slots for e in entries], dim=0),
        ships=torch.cat([e.ships for e in entries], dim=0),
        angle=torch.cat([e.angle for e in entries], dim=0),
        eta=torch.cat([e.eta for e in entries], dim=0),
        valid=torch.cat([e.valid for e in entries], dim=0)
    )

def resolve_duplicate_missions(entries: MissionEntries, *, epsilon: float = 1e-05) -> MissionEntries:
    src = entries.source_slots
    ang = entries.angle
    ships = entries.ships
    valid = entries.valid
    num_launches = src.shape[0]
    if num_launches < 2 or not bool(valid.any()):
        return entries
    device = src.device
    src_i = src.unsqueeze(1)
    src_j = src.unsqueeze(0)
    ang_i = ang.unsqueeze(1)
    ang_j = ang.unsqueeze(0)
    ships_i = ships.unsqueeze(1)
    ships_j = ships.unsqueeze(0)
    valid_i = valid.unsqueeze(1)
    valid_j = valid.unsqueeze(0)
    j_indices = torch.arange(num_launches, device=device).view(1, num_launches)
    i_indices = torch.arange(num_launches, device=device).view(num_launches, 1)
    earlier = j_indices < i_indices
    match = valid_i & valid_j & (src_i == src_j) & (ang_i == ang_j) & (ships_i == ships_j) & earlier
    if not bool(match.any()):
        return entries
    dup_count = match.sum(dim=1).to(ang.dtype)
    new_angle = ang + dup_count * float(epsilon)
    return MissionEntries(
        source_slots=entries.source_slots, target_slots=entries.target_slots,
        ships=entries.ships, angle=new_angle, eta=entries.eta, valid=entries.valid
    )

def get_or_create_predictor(*, obs_tensors: dict, expected_cfg: SimulationConfig, cached_movement: StatePredictor | None) -> StatePredictor:
    if cached_movement is not None and cached_movement.config == expected_cfg:
        cached_movement.update(obs_tensors)
        return cached_movement
    return StatePredictor.from_obs_tensors(obs_tensors, config=expected_cfg)

def _resolve_player_next_fleet_id(obs_tensors: dict, *, device: torch.device) -> Tensor:
    next_fleet_id = obs_tensors.get('player_next_fleet_id', obs_tensors.get('next_fleet_id'))
    if next_fleet_id is None:
        return torch.zeros((), dtype=torch.long, device=device)
    return next_fleet_id.to(device=device, dtype=torch.long)

def generate_scheduled_missions(*, obs_tensors: dict, movement: StatePredictor, entries: MissionEntries, player_id: int) -> ScheduledMissions:
    source_slots = entries.source_slots
    angle = entries.angle
    ships = entries.ships
    launch_valid = entries.valid
    num_launches = source_slots.shape[0]
    device = source_slots.device
    num_planets = max(int(movement.num_planets), 1)
    next_fleet_id = _resolve_player_next_fleet_id(obs_tensors, device=device)
    launch_long = launch_valid.to(torch.long)
    launch_rank = launch_long.cumsum(0) - launch_long
    fleet_ids = next_fleet_id + launch_rank
    src_safe = source_slots.clamp(min=0, max=num_planets - 1)
    launch_x, launch_y = movement.position_at_slots(src_safe, 0)
    source_r = movement.radii[src_safe]
    start_x = launch_x + torch.cos(angle) * (source_r + 0.1)
    start_y = launch_y + torch.sin(angle) * (source_r + 0.1)
    source_planet_ids = movement.planet_ids[src_safe]
    rows = torch.full((num_launches, 7), -1.0, dtype=movement.dtype, device=device)
    rows[..., 0] = fleet_ids.to(dtype=movement.dtype)
    rows[..., 1] = float(player_id)
    rows[..., 2] = start_x.to(dtype=movement.dtype)
    rows[..., 3] = start_y.to(dtype=movement.dtype)
    rows[..., 4] = angle.to(dtype=movement.dtype)
    rows[..., 5] = source_planet_ids.to(dtype=movement.dtype)
    rows[..., 6] = ships.to(dtype=movement.dtype)
    rows[..., 0] = torch.where(launch_valid, rows[..., 0], torch.full_like(rows[..., 0], -1.0))
    target_slots = torch.zeros(num_launches, dtype=torch.long, device=device)
    eta_turns = torch.zeros(num_launches, dtype=torch.float32, device=device)
    intent_valid = torch.zeros(num_launches, dtype=torch.bool, device=device)
    fleet_slot = torch.where(launch_valid)[0]
    if int(fleet_slot.numel()) > 0:
        estimate = _estimate_new_fleet_arrivals(movement=movement, obs_fleets=rows, fleet_slot=fleet_slot)
        valid_hit = estimate['has_hit']
        if bool(valid_hit.any()):
            src = fleet_slot[valid_hit]
            target_slots[src] = estimate['target_slot'][valid_hit]
            eta_turns[src] = estimate['eta_index'][valid_hit].to(dtype=torch.float32) + 1.0
            intent_valid[src] = True
    return ScheduledMissions(source_slots=source_slots, angle=angle, ships=ships, target_slots=target_slots, eta_turns=eta_turns, valid=intent_valid, fleet_ids=fleet_ids)

def commit_planned_missions(*, movement: StatePredictor, launches: ScheduledMissions, owner_id: int, obs_tensors: dict) -> None:
    if not movement.track_fleets:
        return
    movement.record_fleet_arrivals(target_slots=launches.target_slots, owner_ids=int(owner_id), ships=launches.ships, eta=launches.eta_turns, valid=launches.valid)
    nfid = obs_tensors.get('next_fleet_id')
    if nfid is None:
        raise ValueError("obs_tensors is missing 'next_fleet_id'")
    movement.stash_pending_own_launches(owner_id=int(owner_id), source_slots=launches.source_slots, ships=launches.ships, angle=launches.angle, target_slots=launches.target_slots, eta=launches.eta_turns, valid=launches.valid, prev_next_fleet_id=nfid)

# Bộ lập kế hoạch chiến thuật cốt lõi.

def detect_player_count(obs_tensors: dict) -> int:
    metadata_count = obs_tensors.get('player_count')
    if metadata_count is not None:
        count = int(metadata_count.flatten()[0].item()) if isinstance(metadata_count, Tensor) else int(metadata_count)
        if count in (2, 4):
            return count
    initial = obs_tensors['initial_planets']
    pid = initial[:, 0]
    owner = initial[:, 1]
    mask = (pid >= 0) & (owner >= 0)
    owners = owner[mask]
    n_max = 2
    if owners.numel() > 0:
        n_max = max(n_max, int(torch.unique(owners.long()).numel()))
    return n_max

def create_launch_group(*, source_slots: Tensor, target_slots: Tensor, ships: Tensor, eta: Tensor, valid: Tensor, player_id: int) -> FleetLaunchGroup:
    owner = torch.full_like(source_slots, int(player_id), dtype=torch.long)
    return FleetLaunchGroup(source_slots=source_slots.to(torch.long), target_slots=target_slots.to(torch.long), ships=ships, eta=eta, owner=owner, valid=valid.to(torch.bool))

def evaluate_competitive_score(diff: FlowDelta, *, player_id: int) -> Tensor:
    net = diff.net_ship_delta
    me = net[..., int(player_id)]
    opp = net.sum(dim=-1) - me
    return me - opp

def evaluate_launch_options(status: GarrisonPrediction, *, prod: Tensor, alive_by_step: Tensor, player_count: int, launches: FleetLaunchGroup, player_id: int) -> Tensor:
    diff = compute_hypothetical_flow_delta(status, prod=prod, alive_by_step=alive_by_step, player_count=int(player_count), launches=launches, player_id=int(player_id))
    return evaluate_competitive_score(diff, player_id=int(player_id))

def get_top_k_indices_stable(ranked: Tensor, k: int) -> Tensor:
    order = torch.argsort(ranked, dim=-1, descending=True, stable=True)
    return order[..., :max(1, int(k))]

def get_argmax_stable(scores: Tensor) -> Tensor:
    num_candidates = int(scores.shape[-1])
    is_max = scores == scores.max(dim=-1, keepdim=True).values
    idx = torch.arange(num_candidates, device=scores.device).expand_as(scores)
    return torch.where(is_max, idx, torch.full_like(idx, num_candidates)).argmin(dim=-1)

def select_top_candidates(values: Tensor, mask: Tensor, cap: int) -> tuple[Tensor, Tensor]:
    p_count = values.shape[0]
    k = p_count if cap <= 0 else min(int(cap), p_count)
    neg_inf = torch.full_like(values, float('-inf'))
    ranked = torch.where(mask, values, neg_inf)
    top_idx = get_top_k_indices_stable(ranked, max(1, k))
    top_vals = ranked[top_idx]
    return (top_idx, top_vals > float('-inf'))

def check_is_comet_planet(obs_tensors: dict, num_planets: int, device: torch.device) -> Tensor | None:
    comet_ids = obs_tensors.get('comet_planet_ids')
    planets = obs_tensors.get('planets')
    if comet_ids is None or planets is None:
        return None
    planet_ids = planets[..., 0].long()
    comet_ids = comet_ids.to(device=device)
    mask = torch.zeros(num_planets, dtype=torch.bool, device=device)
    for c in range(int(comet_ids.shape[-1])):
        cid = comet_ids[c]
        mask = mask | (planet_ids == cid) & (cid >= 0)
    return mask

def compute_timing_factor(eta: Tensor, *, eta_free: float, eta_scale: float) -> Tensor:
    scale = max(float(eta_scale), 1e-06)
    return ((eta - float(eta_free)) / scale).clamp(0.0, 1.0)

def calculate_required_capture_ships(garrison_status: GarrisonPrediction, *, target_idx: Tensor, k_max: int, capture_overhead: float, player_id: int, reinforcement: Tensor | None = None) -> Tensor:
    ships = garrison_status.ships
    owner = garrison_status.owner
    dtype = ships.dtype if ships.is_floating_point() else torch.float32
    num_targets = target_idx.shape[0]
    H_axis = int(ships.shape[-1])
    num_planets = int(ships.shape[0])
    K = max(0, min(int(k_max), H_axis - 1))
    if K == 0:
        return torch.empty(num_targets, 0, dtype=dtype, device=ships.device)
    tgt = target_idx.clamp(min=0, max=max(num_planets - 1, 0))
    gathered = ships[tgt].to(dtype=dtype)
    owner_g = owner[tgt]
    k_idx = torch.arange(1, K + 1, device=ships.device).view(1, K).expand(num_targets, K)
    defenders = gathered.gather(-1, k_idx)
    mine_at_k = owner_g.gather(-1, k_idx) == int(player_id)
    if reinforcement is not None:
        assert reinforcement.shape[-1] >= K, f'reinforcement last dim {reinforcement.shape[-1]} < capture_floor K={K}'
        extra = reinforcement[..., :K].to(dtype=dtype, device=ships.device)
    else:
        extra = 0.0
    cap = (defenders + float(capture_overhead) + extra).clamp(min=1.0).ceil()
    return torch.where(mine_at_k, torch.ones_like(cap), cap)

def get_attackable_planets_mask(obs, obs_tensors: dict) -> Tensor:
    mask = (obs.is_enemy | obs.is_neutral) & obs.alive
    comet = check_is_comet_planet(obs_tensors, obs.num_planets, obs.device)
    if comet is not None:
        mask = mask & ~comet
    return mask

def find_endangered_friendly_planets(obs, garrison_status: GarrisonPrediction, *, horizon: int, prod: Tensor) -> tuple[Tensor, Tensor]:
    num_planets = obs.num_planets
    device = obs.device
    pid = int(obs.player_id)
    if horizon <= 0:
        z = torch.zeros(num_planets, device=device)
        return (torch.zeros(num_planets, dtype=torch.bool, device=device), z)
    owner_h = garrison_status.owner[..., 1:]
    flips = obs.owned.unsqueeze(-1) & (owner_h != pid)
    any_flip = flips.any(dim=-1)
    flip_turn = get_argmax_stable(flips.to(torch.int64)) + 1
    remaining = (float(horizon) - flip_turn.to(prod.dtype)).clamp(min=0.0)
    urgency = prod * remaining + obs.ships
    urgency = torch.where(any_flip, urgency, torch.full_like(urgency, float('-inf')))
    return (any_flip, urgency)

def identify_promising_targets(obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask):
    num_planets = obs.num_planets
    device = obs.device
    n_attack = max(1, min(int(config.max_offensive_targets), num_planets))
    R = max(0, min(int(config.max_defensive_targets), num_planets))
    attack_mask = get_attackable_planets_mask(obs, obs_tensors)
    proximity = find_closest_target_distance(cache, source_mask, attack_mask, max_k=K_eta)
    attack_pref = torch.where(attack_mask, -proximity, torch.full_like(proximity, float('-inf')))
    atk_idx, atk_exists = select_top_candidates(attack_pref, attack_mask, n_attack)
    if R > 0:
        flip_mask, urgency = find_endangered_friendly_planets(obs, garrison_status, horizon=H, prod=prod)
        def_idx, def_exists = select_top_candidates(urgency, flip_mask, R)
        target_idx = torch.cat([atk_idx, def_idx], dim=0)
        target_exists = torch.cat([atk_exists, def_exists], dim=0)
    else:
        target_idx, target_exists = (atk_idx, atk_exists)
    return (target_idx, target_exists)

def compute_reachability_mask(movement: StatePredictor, *, source_idx: Tensor, target_idx: Tensor, fleet_sizes: Tensor, eta_cap: Tensor, eps: float = 0.0001) -> Tensor:
    num_sources, num_targets, num_launches = fleet_sizes.shape
    num_planets = int(movement.num_planets)
    dt = movement.dtype
    K = max(1, min(int(movement.movement_horizon), int(torch.ceil(eta_cap.max()).item())))
    src = source_idx.clamp(0, num_planets - 1)
    tgt = target_idx.clamp(0, num_planets - 1)
    sx = movement.x[0][src].view(num_sources, 1, 1)
    sy = movement.y[0][src].view(num_sources, 1, 1)
    tx = movement.x[:K + 1].gather(1, tgt.view(1, num_targets).expand(K + 1, num_targets))
    ty = movement.y[:K + 1].gather(1, tgt.view(1, num_targets).expand(K + 1, num_targets))
    ax = tx[:K, :].view(1, K, num_targets)
    ay = ty[:K, :].view(1, K, num_targets)
    bx = tx[1:, :].view(1, K, num_targets)
    by = ty[1:, :].view(1, K, num_targets)
    abx = bx - ax
    aby = by - ay
    apx = sx - ax
    apy = sy - ay
    denom = (abx * abx + aby * aby).clamp(min=1e-12)
    u = ((apx * abx + apy * aby) / denom).clamp(0.0, 1.0)
    cx = ax + u * abx
    cy = ay + u * aby
    seg_dist = torch.sqrt(((sx - cx) ** 2 + (sy - cy) ** 2).clamp(min=0.0))
    src_r = movement.radii[src].view(num_sources, 1, 1)
    tgt_r = movement.radii[tgt].view(1, 1, num_targets)
    gap = src_r + tgt_r + (LAUNCH_SURFACE_OFFSET + TARGET_HIT_SURFACE_OFFSET)
    surf = (seg_dist - gap).clamp(min=0.0)
    kv = torch.arange(1, K + 1, device=movement.device, dtype=dt).view(1, K, 1)
    ratio = surf / kv
    within = kv <= eta_cap.view(1, 1, num_targets)
    ratio = torch.where(within, ratio, torch.full_like(ratio, float('inf')))
    min_ratio = ratio.amin(dim=1)
    speed = estimate_fleet_speed(fleet_sizes.clamp(min=1.0))
    reachable = min_ratio.unsqueeze(-1) <= speed * (1.0 + float(eps))
    distinct = (src.view(num_sources, 1) != tgt.view(1, num_targets)).unsqueeze(-1)
    return reachable & distinct

def greedy_mission_selection(*, num_planets, max_waves, device, dtype, score, cand_src, cand_send, cand_angle, cand_eta, cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def, source_budget, target_exists, roi_threshold) -> MissionEntries:
    num_candidates, num_launches = (int(cand_src.shape[0]), int(cand_src.shape[1]))
    target_taken = ~target_exists.clone()
    defended = torch.zeros(num_planets, dtype=torch.bool, device=device)
    used_src = torch.zeros(num_planets, dtype=torch.bool, device=device)
    w_src = torch.zeros(max_waves, num_launches, dtype=torch.long, device=device)
    w_send = torch.zeros(max_waves, num_launches, dtype=dtype, device=device)
    w_angle = torch.zeros(max_waves, num_launches, dtype=dtype, device=device)
    w_eta = torch.ones(max_waves, num_launches, dtype=dtype, device=device)
    w_tgt = torch.zeros(max_waves, num_launches, dtype=torch.long, device=device)
    w_active = torch.zeros(max_waves, num_launches, dtype=torch.bool, device=device)
    for w in range(max_waves):
        taken_cand = target_taken[cand_tgt_short]
        budget_at = source_budget[cand_src]
        can_fund = ((cand_send <= budget_at) | ~cand_active).all(dim=-1)
        tgt_used_as_src = used_src[cand_tgt_slot]
        contrib_defended = (defended[cand_src] & cand_active).any(dim=-1)
        mask = torch.isfinite(score) & ~taken_cand & can_fund & ~tgt_used_as_src & ~contrib_defended
        masked = torch.where(mask, score, torch.full_like(score, float('-inf')))
        best_c = get_argmax_stable(masked)
        best_score = masked[best_c]
        fired = bool(torch.isfinite(best_score) & (best_score > roi_threshold))
        if not fired:
            break
        sel_src = cand_src[best_c]
        sel_send = cand_send[best_c]
        sel_active = cand_active[best_c]
        w_src[w] = sel_src
        w_send[w] = torch.where(sel_active, sel_send, torch.zeros_like(sel_send))
        w_angle[w] = cand_angle[best_c]
        w_eta[w] = cand_eta[best_c]
        w_tgt[w] = cand_tgt_slot[best_c]
        w_active[w] = sel_active
        debit = torch.zeros_like(source_budget)
        debit.scatter_add_(0, sel_src, torch.where(sel_active, sel_send, torch.zeros_like(sel_send)))
        source_budget = (source_budget - debit).clamp(min=0.0)
        target_taken[cand_tgt_short[best_c]] = True
        src_mark = torch.zeros(num_planets, dtype=torch.long, device=device)
        src_mark.scatter_add_(0, sel_src, sel_active.to(torch.long))
        used_src = used_src | (src_mark > 0)
        sel_tgt = cand_tgt_slot[best_c]
        sel_is_def = bool(cand_is_def[best_c])
        defended[sel_tgt] = defended[sel_tgt] | sel_is_def
    WL = max_waves * num_launches
    entries = MissionEntries(
        source_slots=w_src.reshape(WL), target_slots=w_tgt.reshape(WL),
        ships=torch.where(w_active, w_send, torch.zeros_like(w_send)).reshape(WL),
        angle=torch.where(w_active, w_angle, torch.zeros_like(w_angle)).reshape(WL),
        eta=torch.where(w_active, w_eta, torch.ones_like(w_eta)).reshape(WL),
        valid=w_active.reshape(WL)
    )
    return (entries, source_budget)

def plan_regroup_missions(*, movement, obs, obs_tensors, garrison_status, leftover, original_ships, pressure, config, H) -> MissionEntries:
    num_planets = obs.num_planets
    device = obs.device
    dtype = original_ships.dtype
    pid = int(obs.player_id)
    min_send = float(config.min_ships_to_launch)
    src_mask = obs.owned & obs.alive & (leftover >= min_send)
    if not bool(src_mask.any()):
        return create_empty_missions(device, dtype)
    S_cap = max(1, min(int(config.max_regroup_sources_per_lane), num_planets))
    src_idx, src_exists = select_top_candidates(leftover, src_mask, S_cap)
    num_sources = int(src_idx.shape[0])
    leftover_s = leftover[src_idx.clamp(0, num_planets - 1)]
    orig_s = original_ships[src_idx.clamp(0, num_planets - 1)]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain_s = calculate_safe_garrison_drain(garrison_status, source_idx=src_idx, source_ships=orig_s, H_eff=H_eff, player_id=pid)
    committed_s = (orig_s - leftover_s).clamp(min=0.0)
    regroup_cap = torch.minimum(leftover_s, (drain_s - committed_s).clamp(min=0.0)).floor()
    can_send = src_exists & (regroup_cap >= min_send)
    if not bool(can_send.any()):
        return create_empty_missions(device, dtype)
    dst_mask = obs.owned & obs.alive
    comet = check_is_comet_planet(obs_tensors, num_planets, device)
    if comet is not None:
        dst_mask = dst_mask & ~comet
    T_cap = max(1, min(int(config.max_regroup_targets_per_source), num_planets))
    dst_idx, dst_exists = select_top_candidates(pressure, dst_mask, T_cap)
    num_targets = int(dst_idx.shape[0])
    regroup_active = compute_reachability_mask(movement, source_idx=src_idx, target_idx=dst_idx, fleet_sizes=regroup_cap.view(num_sources, 1, 1).expand(num_sources, num_targets, 1), eta_cap=torch.full((num_targets,), float(config.max_regroup_time), device=device)).squeeze(-1)
    aim = calculate_intercept_angle(movement, src_idx.unsqueeze(1), dst_idx.unsqueeze(0), regroup_cap.unsqueeze(1), active=regroup_active)
    angle = aim['angle']
    eta = aim['eta']
    viable = aim['viable']
    src_pres = pressure[src_idx.clamp(0, num_planets - 1)].view(num_sources, 1)
    dst_pres = pressure[dst_idx.clamp(0, num_planets - 1)].view(1, num_targets)
    gap = dst_pres - src_pres
    owner = garrison_status.owner
    H_axis = int(owner.shape[-1])
    dst_owner = owner[dst_idx.clamp(0, num_planets - 1)]
    k = torch.ceil(eta).clamp(min=0, max=H_axis - 1).to(torch.long)
    owner_at_k = dst_owner.unsqueeze(0).expand(num_sources, num_targets, H_axis).gather(-1, k.unsqueeze(-1)).squeeze(-1)
    still_mine = owner_at_k == pid
    src_neq_dst = src_idx.view(num_sources, 1) != dst_idx.view(1, num_targets)
    valid = viable & still_mine & src_neq_dst & (gap > float(config.regroup_pressure_delta_min)) & (eta <= float(config.max_regroup_time)) & can_send.view(num_sources, 1) & dst_exists.view(1, num_targets)
    sc = torch.where(valid, gap - float(config.regroup_time_penalty_weight) * eta, torch.full_like(gap, float('-inf')))
    best_t = get_argmax_stable(sc)
    best_score = sc.gather(-1, best_t.unsqueeze(-1)).squeeze(-1)
    best_valid = torch.isfinite(best_score)
    s_ar = torch.arange(num_sources, device=device)
    best_dst = dst_idx[best_t]
    best_angle = angle[s_ar, best_t]
    best_eta = eta[s_ar, best_t]
    return MissionEntries(
        source_slots=src_idx, target_slots=best_dst,
        ships=torch.where(best_valid, regroup_cap, torch.zeros_like(regroup_cap)),
        angle=torch.where(best_valid, best_angle, torch.zeros_like(best_angle)),
        eta=torch.where(best_valid, best_eta, torch.ones_like(best_eta)),
        valid=best_valid
    )

def create_empty_missions(device: torch.device, dtype: torch.dtype) -> MissionEntries:
    z = torch.zeros(0, dtype=dtype, device=device)
    zl = torch.zeros(0, dtype=torch.long, device=device)
    return MissionEntries(source_slots=zl, target_slots=zl, ships=z, angle=z, eta=z, valid=torch.zeros(0, dtype=torch.bool, device=device))

def serialize_missions_to_payload(entries: MissionEntries, *, planet_ids: Tensor) -> dict[str, Tensor]:
    num_launches = entries.source_slots.shape[0]
    device = entries.source_slots.device
    num_planets = int(planet_ids.shape[0])
    valid_long = entries.valid.to(torch.int64)
    counts = valid_long.sum().to(torch.int32)
    max_count = int(counts.item())
    out_from = torch.full((max_count,), -1, dtype=torch.int32, device=device)
    out_angle = torch.zeros((max_count,), dtype=torch.float32, device=device)
    out_ships = torch.zeros((max_count,), dtype=torch.float32, device=device)
    if max_count == 0:
        return {'from_planet_id': out_from, 'angle': out_angle, 'num_ships': out_ships, 'counts': counts}
    safe_src = entries.source_slots.clamp(min=0, max=max(num_planets - 1, 0))
    from_pid_full = planet_ids[safe_src].to(torch.int32)
    launch_rank = valid_long.cumsum(0) - valid_long
    l_idx = torch.where(entries.valid)[0]
    pos = launch_rank[l_idx]
    out_from[pos] = from_pid_full[l_idx]
    out_angle[pos] = entries.angle[l_idx].to(torch.float32)
    out_ships[pos] = entries.ships[l_idx].to(torch.float32)
    return {'from_planet_id': out_from, 'angle': out_angle, 'num_ships': out_ships, 'counts': counts}

def get_empty_action_payload(device: torch.device) -> dict[str, Tensor]:
    return {
        'from_planet_id': torch.full((0,), -1, dtype=torch.int32, device=device),
        'angle': torch.zeros((0,), dtype=torch.float32, device=device),
        'num_ships': torch.zeros((0,), dtype=torch.float32, device=device),
        'counts': torch.zeros((), dtype=torch.int32, device=device)
    }

def calculate_safe_garrison_drain(garrison_status: GarrisonPrediction, *, source_idx: Tensor, source_ships: Tensor, H_eff: Tensor, player_id: int = 0) -> Tensor:
    num_sources = source_idx.shape[0]
    ships_cache = garrison_status.ships
    dtype = ships_cache.dtype if ships_cache.is_floating_point() else torch.float32
    device = ships_cache.device
    H_axis = int(ships_cache.shape[-1])
    horizon = max(H_axis - 1, 0)
    num_planets = int(ships_cache.shape[0])
    if horizon == 0:
        return torch.zeros(num_sources, dtype=dtype, device=device)
    src_idx_safe = source_idx.clamp(min=0, max=max(num_planets - 1, 0))
    src_ships_traj = ships_cache[src_idx_safe][..., 1:].to(dtype=dtype)
    src_owner_traj = garrison_status.owner[src_idx_safe][..., 1:]
    me_owned = src_owner_traj == int(player_id)
    turn_grid = torch.arange(1, horizon + 1, device=device, dtype=dtype).view(1, horizon)
    within_horizon = turn_grid <= H_eff
    held = me_owned & within_horizon & (src_ships_traj > 0.0)
    inf_fill = torch.full_like(src_ships_traj, float('inf'))
    cap_traj = torch.where(held, src_ships_traj, inf_fill)
    min_slack = cap_traj.min(dim=-1).values
    return torch.minimum(min_slack, source_ships.to(dtype)).clamp(min=0.0)

# Bộ chuyển đổi dữ liệu giữa định dạng danh sách di chuyển của môi trường và các tensor.

def estimate_active_players(planets: list[Any], fleets: list[Any], player_id: int) -> int:
    owners: list[int] = [int(player_id)]
    for row in planets:
        if len(row) >= 2 and int(row[0]) >= 0 and (int(row[1]) >= 0):
            owners.append(int(row[1]))
    for row in fleets:
        if len(row) >= 2 and int(row[0]) >= 0 and (int(row[1]) >= 0):
            owners.append(int(row[1]))
    return 4 if max(owners, default=0) >= 2 else 2

def convert_obs_dict_to_tensors(obs: dict[str, Any], player_id: int, num_planets: int = MAX_PLANET_COUNT, num_fleets: int = MAX_FLEET_COUNT, device: Any = 'cpu') -> dict[str, Any]:
    dev = torch.device(device)
    planets_raw = obs.get('planets', [])
    initial_planets_raw = obs.get('initial_planets', planets_raw)
    fleets_raw = obs.get('fleets', [])
    comets_raw = obs.get('comets', [])
    comet_planet_ids_raw = obs.get('comet_planet_ids', [])
    step = int(obs.get('step', 0))
    angvel = float(obs.get('angular_velocity', 0.03))
    max_steps = int(obs.get('episode_steps', DEFAULT_MAX_EPISODE_STEPS))
    remaining_overtime = float(obs.get('remainingOverageTime', 2.0))
    next_fleet_id = int(obs.get('next_fleet_id', 0))
    planet_t = torch.zeros(num_planets, 7, dtype=torch.float32, device=dev)
    planet_t[..., 0] = -1.0
    for i, p in enumerate(planets_raw[:num_planets]):
        pid, owner, x, y, r, ships, prod = p[:7]
        planet_t[i, 0] = float(pid)
        planet_t[i, 1] = float(owner)
        planet_t[i, 2] = float(x)
        planet_t[i, 3] = float(y)
        planet_t[i, 4] = float(r)
        planet_t[i, 5] = float(ships)
        planet_t[i, 6] = float(prod)
    initial_planet_t = torch.zeros(num_planets, 7, dtype=torch.float32, device=dev)
    initial_planet_t[..., 0] = -1.0
    for i, p in enumerate(initial_planets_raw[:num_planets]):
        pid, owner, x, y, r, ships, prod = p[:7]
        initial_planet_t[i, 0] = float(pid)
        initial_planet_t[i, 1] = float(owner)
        initial_planet_t[i, 2] = float(x)
        initial_planet_t[i, 3] = float(y)
        initial_planet_t[i, 4] = float(r)
        initial_planet_t[i, 5] = float(ships)
        initial_planet_t[i, 6] = float(prod)
    fleet_t = torch.zeros(num_fleets, 7, dtype=torch.float32, device=dev)
    fleet_t[..., 0] = -1.0
    fleet_t[..., 5] = -1.0
    for i, f in enumerate(fleets_raw[:num_fleets]):
        fid, owner, x, y, angle, from_id, ships = f[:7]
        fleet_t[i, 0] = float(fid)
        fleet_t[i, 1] = float(owner)
        fleet_t[i, 2] = float(x)
        fleet_t[i, 3] = float(y)
        fleet_t[i, 4] = float(angle)
        fleet_t[i, 5] = float(from_id)
        fleet_t[i, 6] = float(ships)
    comet_ids = torch.full((COMET_EVENT_COUNT, COMETS_PER_SPAWN), -1, dtype=torch.int32, device=dev)
    comet_paths = torch.full((COMET_EVENT_COUNT, COMETS_PER_SPAWN, COMET_MAX_PATH_LEN, 2), float('nan'), dtype=torch.float32, device=dev)
    comet_path_index = torch.full((COMET_EVENT_COUNT,), -1, dtype=torch.int32, device=dev)
    for group_idx, group in enumerate(comets_raw[:COMET_EVENT_COUNT]):
        comet_path_index[group_idx] = int(group.get('path_index', -1))
        group_ids = group.get('planet_ids', [])
        group_paths = group.get('paths', [])
        for comet_idx, pid in enumerate(group_ids[:COMETS_PER_SPAWN]):
            comet_ids[group_idx, comet_idx] = int(pid)
        for comet_idx, path in enumerate(group_paths[:COMETS_PER_SPAWN]):
            for point_idx, point in enumerate(path[:COMET_MAX_PATH_LEN]):
                comet_paths[group_idx, comet_idx, point_idx, 0] = float(point[0])
                comet_paths[group_idx, comet_idx, point_idx, 1] = float(point[1])
    comet_planet_ids = torch.full((COMET_EVENT_COUNT * COMETS_PER_SPAWN,), -1, dtype=torch.int32, device=dev)
    for idx, pid in enumerate(comet_planet_ids_raw[:COMET_EVENT_COUNT * COMETS_PER_SPAWN]):
        comet_planet_ids[idx] = int(pid)
    return {
        'planets': planet_t, 'fleets': fleet_t,
        'player': torch.tensor(player_id, dtype=torch.int32, device=dev),
        'player_count': torch.tensor(estimate_active_players(planets_raw, fleets_raw, player_id), dtype=torch.int32, device=dev),
        'angular_velocity': torch.tensor(angvel, dtype=torch.float32, device=dev),
        'initial_planets': initial_planet_t,
        'next_fleet_id': torch.tensor(next_fleet_id, dtype=torch.int32, device=dev),
        'comets': {'planet_ids': comet_ids, 'paths': comet_paths, 'path_index': comet_path_index},
        'comet_planet_ids': comet_planet_ids, 'step': torch.tensor(step, dtype=torch.int32, device=dev),
        'episode_steps': torch.tensor(max_steps, dtype=torch.int32, device=dev),
        'remainingOverageTime': torch.tensor(remaining_overtime, dtype=torch.float32, device=dev)
    }

def convert_payload_to_moves(action_payload: dict[str, Any], obs: dict[str, Any], player_id: int) -> list[list[Any]]:
    from_pid_t = action_payload['from_planet_id']
    angle_t = action_payload['angle']
    num_ships_t = action_payload['num_ships']
    counts = int(action_payload['counts'].item())
    planets_by_id = {int(p[0]): p for p in obs.get('planets', []) if len(p) >= 7}
    moves: list[list[Any]] = []
    for launch_idx in range(counts):
        from_pid = int(from_pid_t[launch_idx].item())
        ships = float(num_ships_t[launch_idx].item())
        angle = float(angle_t[launch_idx].item())
        if ships < 1.0:
            continue
        source = planets_by_id.get(from_pid)
        if source is None:
            continue
        owner = int(source[1])
        available = float(source[5])
        if owner != int(player_id):
            continue
        if ships != float(round(ships)) or ships > available:
            raise ValueError(f'Invalid launch ship count in sparse action payload at from_planet_id={from_pid}: requested={ships}, available={available}. Counts must be finite, integer-valued, >= 0, and <= available planet ships.')
        moves.append([from_pid, angle, int(ships)])
    return moves

def preprocess_observation(obs: dict[str, Any], *, player_id: int, num_planets: int = MAX_PLANET_COUNT, num_fleets: int = MAX_FLEET_COUNT, device: Any = 'cpu') -> dict[str, Any]:
    return convert_obs_dict_to_tensors(obs, player_id=player_id, num_planets=num_planets, num_fleets=num_fleets, device=device)

def postprocess_action_payload(action_payload: dict[str, Any], obs: dict[str, Any], *, player_id: int) -> list[list[Any]]:
    return convert_payload_to_moves(action_payload, obs, player_id=int(player_id))

@dataclass(frozen=True)
class AgentStrategySettings:
    horizon: int = 18
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
    max_defensive_targets: int = 4
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5
    min_ships_to_launch: float = 4.0
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = 'none'
    regroup_time_penalty_weight: float = 0.001
    enable_potential_risk: bool = False
    risk_blend_weight: float = 1.0
    risk_enemy_prod_weight: float = 2.0
    risk_self_prod_weight: float = 2.0
    risk_support_weight: float = 0.5
    enable_focus_fire: bool = True
    max_strike_sources: int = 4

def create_predictor_config(config: AgentStrategySettings, *, player_count: int) -> SimulationConfig:
    return SimulationConfig(movement_horizon=int(config.horizon), drift_epsilon=0.001, track_fleets=True, player_count=int(player_count), max_tracked_fleets=128)

def compute_proximity_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    num_planets = int(obs.num_planets)
    device = obs.device
    dtype = obs.ships.dtype
    if num_planets == 0:
        return torch.zeros(num_planets, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = estimate_fleet_speed(ships.clamp(min=1e-06))
    reach_dist = (speeds.view(num_planets, 1) * float(horizon)).clamp(min=1e-06)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(num_planets, device=device, dtype=torch.bool)
    valid = enemy.view(num_planets, 1) & obs.alive.view(1, num_planets) & ~eye
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib = torch.where(valid, ships.view(num_planets, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)

def compute_potential_attack_risk(obs, cache, *, horizon: float, player_id: int, config) -> Tensor:
    num_planets = int(obs.num_planets)
    device = obs.device
    dtype = obs.ships.dtype
    if num_planets == 0:
        return torch.zeros(num_planets, dtype=dtype, device=device)
    H = max(float(horizon), 1e-06)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    prod = obs.prod.to(dtype)
    speeds = estimate_fleet_speed(ships.clamp(min=1e-06))
    reach = (speeds.view(num_planets, 1) * H).clamp(min=1e-06)
    decay = (1.0 - d0 / reach).clamp(min=0.0)
    eye = torch.eye(num_planets, device=device, dtype=torch.bool)
    x = obs.x.to(dtype)
    y = obs.y.to(dtype)
    ax = x.view(num_planets, 1)
    ay = y.view(num_planets, 1)
    bx = x.view(1, num_planets)
    by = y.view(1, num_planets)
    abx = bx - ax
    aby = by - ay
    denom = (abx * abx + aby * aby).clamp(min=1e-12)
    u = (((PLAY_BOARD_CENTER - ax) * abx + (PLAY_BOARD_CENTER - ay) * aby) / denom).clamp(0.0, 1.0)
    cx = ax + u * abx
    cy = ay + u * aby
    sun_dist = torch.sqrt(((cx - PLAY_BOARD_CENTER) ** 2 + (cy - PLAY_BOARD_CENTER) ** 2).clamp(min=0.0))
    los_clear = (sun_dist >= float(SOLAR_SUN_RADIUS)).to(dtype)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    strength = ships + float(config.risk_enemy_prod_weight) * prod
    valid_e = enemy.view(num_planets, 1) & obs.alive.view(1, num_planets) & ~eye
    threat = torch.where(valid_e, strength.view(num_planets, 1) * decay * los_clear, torch.zeros_like(decay))
    enemy_threat = threat.sum(dim=0)
    own = obs.owned & obs.alive
    valid_o = own.view(num_planets, 1) & obs.alive.view(1, num_planets) & ~eye
    support = torch.where(valid_o, (1.0 + ships).view(num_planets, 1) * decay, torch.zeros_like(decay)).sum(dim=0)
    value = 1.0 + float(config.risk_self_prod_weight) * prod
    return value * enemy_threat / (1.0 + float(config.risk_support_weight) * support)

def optimize_waves(*, movement: StatePredictor, obs, obs_tensors: dict, cache, garrison_status, prod: Tensor, alive_by_step: Tensor, config: AgentStrategySettings, player_count: int):
    num_planets = obs.num_planets
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    H_axis = int(garrison_status.ships.shape[-1])
    horizon = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), horizon))
    max_waves = max(1, int(config.max_waves_per_turn))
    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return create_empty_missions(device, dtype)
    S_cap = max(1, min(int(config.max_sources_per_lane), num_planets))
    source_idx, source_exists = select_top_candidates(obs.ships, source_mask, S_cap)
    target_idx, target_exists = identify_promising_targets(obs, obs_tensors, garrison_status, cache, config=config, K_eta=K_eta, H=horizon, prod=prod, source_mask=source_mask)
    if not bool(target_exists.any()):
        return create_empty_missions(device, dtype)
    num_sources = int(source_idx.shape[0])
    num_targets = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, num_planets - 1)]
    source_ships = obs.ships[source_idx.clamp(0, num_planets - 1)].to(dtype)
    H_eff = torch.full((), float(horizon), dtype=dtype, device=device)
    drain = calculate_safe_garrison_drain(garrison_status, source_idx=source_idx, source_ships=source_ships, H_eff=H_eff, player_id=pid)
    eta_cap = torch.full((num_targets,), float(K_eta), dtype=dtype, device=device)
    floor = calculate_required_capture_ships(garrison_status, target_idx=target_idx, k_max=K_eta, capture_overhead=1.0, player_id=pid)
    K = int(floor.shape[-1])
    sizes = drain.view(num_sources, 1).expand(num_sources, num_targets).floor()
    active = compute_reachability_mask(movement, source_idx=source_idx, target_idx=target_idx, fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap).squeeze(-1)
    aim = calculate_intercept_angle(movement, source_idx.unsqueeze(1), target_idx.unsqueeze(0), sizes, active=active)
    angle = aim['angle']
    eta = aim['eta']
    viable = aim['viable'] & (eta <= eta_cap.view(1, num_targets))
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(num_sources, num_targets, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(num_sources, num_targets, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr
    src_neq_dst = source_idx.view(num_sources, 1) != target_idx.view(1, num_targets)
    valid = viable & clears_floor & (sizes >= 1.0) & src_neq_dst & source_exists.view(num_sources, 1) & target_exists.view(1, num_targets)
    if not bool(config.enable_focus_fire):
        num_launches = 1
        num_candidates = num_sources * num_targets
        cand_src = source_idx.view(num_sources, 1).expand(num_sources, num_targets).reshape(num_candidates, num_launches)
        cand_tgt_slot = target_idx.view(1, num_targets).expand(num_sources, num_targets).reshape(num_candidates)
        cand_tgt_short = torch.arange(num_targets, device=device).view(1, num_targets).expand(num_sources, num_targets).reshape(num_candidates)
        cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(num_candidates, num_launches)
        cand_angle = angle.reshape(num_candidates, num_launches)
        cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(num_candidates, num_launches)
        cand_active = valid.reshape(num_candidates, num_launches)
        cand_valid = valid.reshape(num_candidates)
    else:
        num_launches = max(1, int(config.max_strike_sources))
        ST = num_sources * num_targets
        ss_src = torch.zeros(ST, num_launches, dtype=torch.long, device=device)
        ss_src[:, 0] = source_idx.view(num_sources, 1).expand(num_sources, num_targets).reshape(-1)
        ss_send = torch.zeros(ST, num_launches, dtype=dtype, device=device)
        ss_send[:, 0] = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(-1)
        ss_angle = torch.zeros(ST, num_launches, dtype=dtype, device=device)
        ss_angle[:, 0] = angle.reshape(-1)
        ss_eta = torch.ones(ST, num_launches, dtype=dtype, device=device)
        ss_eta[:, 0] = torch.where(valid, eta, torch.ones_like(eta)).reshape(-1)
        ss_active = torch.zeros(ST, num_launches, dtype=torch.bool, device=device)
        ss_active[:, 0] = valid.reshape(-1)
        ss_tgt_slot = target_idx.view(1, num_targets).expand(num_sources, num_targets).reshape(-1)
        ss_tgt_short = torch.arange(num_targets, device=device).view(1, num_targets).expand(num_sources, num_targets).reshape(-1)
        ss_valid = valid.reshape(-1)
        eligible = viable & (sizes >= 1.0) & src_neq_dst & source_exists.view(num_sources, 1) & target_exists.view(1, num_targets)
        step_arr = eta.clamp(min=1.0, max=float(K_eta)).ceil().long()
        pooled = []
        if num_launches >= 2 and K > 0:
            for t in range(num_targets):
                if bool(target_is_mine[t]):
                    continue
                rows = torch.nonzero(eligible[:, t], as_tuple=False).flatten()
                if int(rows.numel()) < 2:
                    continue
                steps_t = step_arr[rows, t]
                for k in torch.unique(steps_t).tolist():
                    k = int(k)
                    if k < 1 or k - 1 >= K:
                        continue
                    grp = rows[steps_t == k]
                    if int(grp.numel()) < 2:
                        continue
                    gd = sizes[grp, t]
                    order = torch.argsort(gd, descending=True, stable=True)
                    grp = grp[order]
                    csum = torch.cumsum(gd[order], dim=0)
                    need = floor[t, k - 1]
                    hit = torch.nonzero(csum >= need, as_tuple=False)
                    if int(hit.numel()) == 0:
                        continue
                    j = int(hit[0].item()) + 1
                    if j < 2 or j > num_launches:
                        continue
                    pooled.append((t, grp[:j]))
        if pooled:
            num_candidates2 = len(pooled)
            p_src = torch.zeros(num_candidates2, num_launches, dtype=torch.long, device=device)
            p_send = torch.zeros(num_candidates2, num_launches, dtype=dtype, device=device)
            p_angle = torch.zeros(num_candidates2, num_launches, dtype=dtype, device=device)
            p_eta = torch.ones(num_candidates2, num_launches, dtype=dtype, device=device)
            p_active = torch.zeros(num_candidates2, num_launches, dtype=torch.bool, device=device)
            p_tgt_slot = torch.zeros(num_candidates2, dtype=torch.long, device=device)
            p_tgt_short = torch.zeros(num_candidates2, dtype=torch.long, device=device)
            for i, (t, grp) in enumerate(pooled):
                j = int(grp.numel())
                p_src[i, :j] = source_idx[grp]
                p_send[i, :j] = sizes[grp, t]
                p_angle[i, :j] = angle[grp, t]
                p_eta[i, :j] = eta[grp, t]
                p_active[i, :j] = True
                p_tgt_slot[i] = target_idx[t]
                p_tgt_short[i] = t
            cand_src = torch.cat([ss_src, p_src], dim=0)
            cand_send = torch.cat([ss_send, p_send], dim=0)
            cand_angle = torch.cat([ss_angle, p_angle], dim=0)
            cand_eta = torch.cat([ss_eta, p_eta], dim=0)
            cand_active = torch.cat([ss_active, p_active], dim=0)
            cand_tgt_slot = torch.cat([ss_tgt_slot, p_tgt_slot], dim=0)
            cand_tgt_short = torch.cat([ss_tgt_short, p_tgt_short], dim=0)
            cand_valid = torch.cat([ss_valid, torch.ones(num_candidates2, dtype=torch.bool, device=device)], dim=0)
        else:
            cand_src, cand_send, cand_angle = (ss_src, ss_send, ss_angle)
            cand_eta, cand_active = (ss_eta, ss_active)
            cand_tgt_slot, cand_tgt_short, cand_valid = (ss_tgt_slot, ss_tgt_short, ss_valid)
        num_candidates = int(cand_src.shape[0])
    cand_is_def = target_is_mine[cand_tgt_short]
    launches = create_launch_group(source_slots=cand_src, target_slots=cand_tgt_slot.unsqueeze(-1).expand(num_candidates, num_launches), ships=cand_send, eta=cand_eta, valid=cand_active & cand_valid.unsqueeze(-1), player_id=pid)
    score = evaluate_launch_options(garrison_status, prod=prod, alive_by_step=alive_by_step, player_count=int(player_count), launches=launches, player_id=pid)
    score = torch.where(cand_valid, score, torch.full_like(score, float('-inf')))
    wave_entries, leftover = greedy_mission_selection(
        num_planets=num_planets, max_waves=max_waves, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold)
    )
    if not bool(config.enable_regroup):
        return wave_entries
    enemy_mass = compute_proximity_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    if bool(config.enable_potential_risk):
        enemy_mass = enemy_mass + float(config.risk_blend_weight) * compute_potential_attack_risk(obs, cache, horizon=float(K_eta), player_id=pid, config=config)
    regroup_entries = plan_regroup_missions(movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status, leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass, config=config, H=horizon)
    return merge_mission_entries([wave_entries, regroup_entries])

def execute_agent_turn(obs_tensors: dict, *, config: AgentStrategySettings, player_count: int, memory) -> dict:
    device = obs_tensors['planets'].device
    obs = extract_game_state(obs_tensors)
    num_planets = obs.num_planets
    if num_planets == 0:
        return get_empty_action_payload(device)
    movement = get_or_create_predictor(obs_tensors=obs_tensors, expected_cfg=create_predictor_config(config, player_count=int(player_count)), cached_movement=getattr(memory, 'movement', None))
    memory.movement = movement
    cache = initialize_distance_matrix(movement, max_k=int(config.horizon))
    horizon = int(config.horizon)
    status = movement.garrison_status(max_horizon=horizon)
    alive_by_step = movement.alive_by_step[:horizon + 1]
    entries = optimize_waves(movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache, garrison_status=status, prod=movement.planet_prod, alive_by_step=alive_by_step, config=config, player_count=int(player_count))
    entries = resolve_duplicate_missions(entries)
    launches = generate_scheduled_missions(obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id))
    commit_planned_missions(movement=movement, launches=launches, owner_id=int(obs.player_id), obs_tensors=obs_tensors)
    planet_ids = obs_tensors['planets'][..., 0].long()
    return serialize_missions_to_payload(entries, planet_ids=planet_ids)

CONFIG_4P = dataclasses.replace(
    AgentStrategySettings(), horizon=13, max_sources_per_lane=6,
    max_defensive_targets=2, max_regroup_time=6.0, max_regroup_targets_per_source=8,
    risk_blend_weight=0.5, max_strike_sources=3
)

def select_strategy_config(player_count: int) -> AgentStrategySettings:
    return CONFIG_4P if int(player_count) >= 4 else AgentStrategySettings()

class AgentStateMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None

class AgentDecisionEngine:
    def __init__(self, memory: AgentStateMemory | None = None) -> None:
        self.memory = memory if memory is not None else AgentStateMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors['step'] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = detect_player_count(obs_tensors)
        config = select_strategy_config(mem.cached_player_count)
        row = execute_agent_turn(obs_tensors, config=config, player_count=int(mem.cached_player_count), memory=mem)
        mem.last_sparse_action_row = row
        return row

_RUNTIME = AgentDecisionEngine()

def agent(obs):
    player = obs.get('player', 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = preprocess_observation(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return postprocess_action_payload(sparse_row, obs, player_id=player_id)
