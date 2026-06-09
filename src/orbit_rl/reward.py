from __future__ import annotations

from typing import Any

PLANET_ADV_COEF = 1.0
SHIP_ADV_COEF = 0.01
PROD_ADV_COEF = 0.05
TERMINAL_REWARD_COEF = 30.0


def _read(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_observation(state: Any) -> Any:
    return _read(state, "observation", state)


def extract_status(state: Any) -> str:
    return _read(state, "status", "ACTIVE")


def extract_reward(state: Any) -> float:
    value = _read(state, "reward", 0.0)
    return 0.0 if value is None else float(value)


def _planet_stats(obs: Any, player: int):
    planets = _read(obs, "planets", []) or []
    fleets = _read(obs, "fleets", []) or []
    my_p = enemy_p = neutral_p = 0
    my_ships = enemy_ships = neutral_ships = 0.0
    my_prod = enemy_prod = neutral_prod = 0.0

    for p in planets:
        owner = int(p[1])
        ships = float(p[5])
        prod = float(p[6])
        if owner == player:
            my_p += 1; my_ships += ships; my_prod += prod
        elif owner == -1:
            neutral_p += 1; neutral_ships += ships; neutral_prod += prod
        else:
            enemy_p += 1; enemy_ships += ships; enemy_prod += prod

    for f in fleets:
        owner = int(f[1])
        ships = float(f[6])
        if owner == player:
            my_ships += ships
        elif owner != -1:
            enemy_ships += ships

    return {
        "my_p": my_p, "enemy_p": enemy_p, "neutral_p": neutral_p,
        "my_ships": my_ships, "enemy_ships": enemy_ships, "neutral_ships": neutral_ships,
        "my_prod": my_prod, "enemy_prod": enemy_prod, "neutral_prod": neutral_prod,
    }


def shaped_reward(prev_obs: Any, next_obs: Any, done: bool = False, terminal_reward: float = 0.0) -> float:
    if prev_obs is None or next_obs is None:
        return TERMINAL_REWARD_COEF * float(terminal_reward) if done else 0.0
    player = int(_read(prev_obs, "player", 0) or 0)
    a = _planet_stats(prev_obs, player)
    b = _planet_stats(next_obs, player)

    planet_adv_prev = a["my_p"] - a["enemy_p"]
    planet_adv_now = b["my_p"] - b["enemy_p"]
    ship_adv_prev = a["my_ships"] - a["enemy_ships"]
    ship_adv_now = b["my_ships"] - b["enemy_ships"]
    prod_adv_prev = a["my_prod"] - a["enemy_prod"]
    prod_adv_now = b["my_prod"] - b["enemy_prod"]

    r = 0.0
    r += PLANET_ADV_COEF * (planet_adv_now - planet_adv_prev)
    r += SHIP_ADV_COEF * (ship_adv_now - ship_adv_prev)
    r += PROD_ADV_COEF * (prod_adv_now - prod_adv_prev)
    if done:
        r += TERMINAL_REWARD_COEF * float(terminal_reward)
    return float(r)
