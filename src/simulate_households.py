# src/simulate_households.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class HouseholdArchetype:
    name: str
    base_kw_profile: np.ndarray  # 24-length array
    ev_prob: float               # Probability household owns EV
    ev_kw: float                 # Charging power if EV present [kW]
    ev_min_hours: int            # Min charging duration
    ev_max_hours: int            # Max charging duration


@dataclass
class Household:
    id: int
    archetype: HouseholdArchetype
    has_ev: bool
    has_dr: bool                 # enrolled in DR program


def demo_archetypes() -> List[HouseholdArchetype]:
    """
    Returns a small set of synthetic archetypes for testing.
    These are placeholders; later we can load from archetypes.csv.
    """
    hours = np.arange(24)

    # Simple shapes (kW): low night, bump AM, evening peak.
    # You can tune these later.
    def shape(base: float, pm_peak: float) -> np.ndarray:
        prof = np.full(24, base, dtype=float)
        prof[6:9] += 0.3        # morning bump
        prof[17:21] += pm_peak  # evening bump
        return prof

    return [
        HouseholdArchetype(
            name="Small_Home",
            base_kw_profile=shape(0.3, 0.7),
            ev_prob=0.20,
            ev_kw=7.0,
            ev_min_hours=2,
            ev_max_hours=4,
        ),
        HouseholdArchetype(
            name="Med_Home",
            base_kw_profile=shape(0.4, 1.0),
            ev_prob=0.35,
            ev_kw=7.0,
            ev_min_hours=2,
            ev_max_hours=4,
        ),
        HouseholdArchetype(
            name="Large_Home",
            base_kw_profile=shape(0.5, 1.3),
            ev_prob=0.45,
            ev_kw=11.0,
            ev_min_hours=2,
            ev_max_hours=5,
        ),
    ]


def sample_households(
    n_households: int,
    dr_participation: float = 0.3,
    seed: Optional[int] = None,
) -> List[Household]:
    """
    Sample households from demo archetypes with EV + DR flags.
    """
    rng = np.random.default_rng(seed)
    archetypes = demo_archetypes()
    probs = np.array([0.4, 0.4, 0.2])  # distribution over archetypes

    households: List[Household] = []
    for i in range(n_households):
        arch = rng.choice(archetypes, p=probs)
        has_ev = rng.random() < arch.ev_prob
        has_dr = rng.random() < dr_participation
        households.append(Household(id=i, archetype=arch, has_ev=has_ev, has_dr=has_dr))

    return households


def simulate_household_profile(
    hh: Household,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a 24h load profile [kW] for one household:
    - base archetype profile
    - optional EV session with stochastic start/duration
    - simple DR behavior: if enrolled, shift EV to later off-peak window
    """
    load = hh.archetype.base_kw_profile.copy()

    if hh.has_ev:
        # Base EV start distribution: 17:00-22:00 arrival
        start_hour = int(rng.integers(17, 23))
        duration = int(rng.integers(hh.archetype.ev_min_hours,
                                    hh.archetype.ev_max_hours + 1))
        ev_kw = hh.archetype.ev_kw

        # DR behavior: if participating, shift charging to 21:00-6:00 window
        if hh.has_dr:
            # choose a start in [21..23] or [0..5]
            if rng.random() < 0.6:
                start_hour = int(rng.integers(21, 24))
            else:
                start_hour = int(rng.integers(0, 6))

        for h in range(duration):
            idx = (start_hour + h) % 24
            load[idx] += ev_kw

    return load


def simulate_households_day(
    n_households: int,
    dr_participation: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[Household]]:
    """
    Simulate 24h load profiles for n_households.

    Returns:
        loads_kw : [n_households, 24] array
        households : list of Household objects
    """
    households = sample_households(
        n_households=n_households,
        dr_participation=dr_participation,
        seed=seed,
    )
    rng = np.random.default_rng(seed)

    loads = np.zeros((n_households, 24), dtype=float)
    for i, hh in enumerate(households):
        loads[i, :] = simulate_household_profile(hh, rng)

    return loads, households
