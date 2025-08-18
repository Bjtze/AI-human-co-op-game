import numpy as np
import random
import pygame

# ==============================
# === Game / World Constants ===
# ==============================
WORLD_SIZE = 5
MAX_ENERGY = 12
MAX_HYDRATION = 12

FOOD_POS = 2
WATER_POS = 4

# Finite, regenerating timers (simple respawn model)
FOOD_RESPAWN_STEPS = 3
WATER_RESPAWN_STEPS = 3

# Day–night cycle
DAY_LEN_STEPS = 10  # day for 10 steps, night for 10 steps, repeat

# Rewards & costs
STEP_REWARD = -0.25
FRIENDSHIP_COST = 0.5
INITIAL_REWARD = 16
BASE_TRADE_CHUNK = 3     # target chunk to exchange if possible
MOVE_COST = 0.5
IDLE_INTERACT_COST = 0.3
THIRST_DRAIN = 0.3
NIGHT_MOVE_PENALTY = 0.1
APPROACH_SHAPING = 0.06
max_steps_per_episode = 60

# Grace period to avoid early deaths during exploration
GRACE_STEPS = 10           # first N steps per episode have reduced drain
GRACE_SCALE = 0.5          # 50% metabolism in grace period

# Trade caps/limits ("only give a certain amount")
GIVE_CAP_PER_TRADE = 3     # max you can give in a single trade (per resource)
GIVE_CAP_PER_STEP = 3      # max you can give across all trades in the same step
GIVE_CAP_TOTAL = 24        # max total you can give over an episode (per resource)

# Random events
EVENT_PROB_RESOURCE_LOSS = 0.06   # per agent per step: lose 1 from a random bar
EVENT_PROB_TILE_DEBUFF   = 0.05   # per tile per step: harvesting weaker for 1-2 turns
DEBUFF_MIN_STEPS = 1
DEBUFF_MAX_STEPS = 2
DEBUFF_MULT = 0.5                 # 50% effective while debuffed

# Actions
ACTIONS = [0, 1, 2]  # 0 = left, 1 = right, 2 = interact
ACT_LEFT, ACT_RIGHT, ACT_INTERACT = 0, 1, 2

# ============================
# === Learning Parameters  ===
# ============================
alpha = 0.1
gamma = 0.9

epsilon = 1.0
min_epsilon = 0.02
decay_rate = 0.995
exploration_burst_every = 5000   # briefly increase exploration every N episodes
exploration_burst_level = 0.2

max_episodes = 100000

# Switch between off-policy Q-learning and on-policy SARSA
USE_SARSA = True

# =================
# === Emotions  ===
# =================
EMOTIONS = ["happy", "anxious", "angry", "lonely", "neutral"]
emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}

emotion_states_visual = {
    "happy": "😊",
    "anxious": "😰",
    "angry": "😡",
    "lonely": "😔",
    "neutral": "😐"
}

def emotion_bonus(emotion):
    # Tiny shaping so emotions subtly affect learning without dominating it
    return {
        "happy":   +0.05,
        "anxious": -0.05,
        "angry":   -0.03,
        "lonely":  -0.02,
        "neutral":  0.00
    }[emotion]

# ============================
# === Compositional Comms  ===
# ============================
# Message = (want_slot, trade_slot)
# want_slot ∈ {none, food, water}
# trade_slot ∈ {none, accept, refuse, wanted}
WANT = ["none", "food", "water"]
TRADE = ["none", "accept", "refuse", "wanted"]  # "wanted" = request trade
want_to_idx = {s: i for i, s in enumerate(WANT)}
trade_to_idx = {s: i for i, s in enumerate(TRADE)}

# =================
# === Q-Tables  ===
# =================
# Action head conditions on partner's last composite message (want, trade)
q_act = np.zeros(
    (
        WORLD_SIZE,
        MAX_ENERGY + 1,
        MAX_HYDRATION + 1,
        len(EMOTIONS),
        len(WANT),
        len(TRADE),
        len(ACTIONS),
    ),
    dtype=np.float32,
)

# Signal heads: choose each slot independently (compositional)
q_sig_want = np.zeros(
    (
        WORLD_SIZE,
        MAX_ENERGY + 1,
        MAX_HYDRATION + 1,
        len(EMOTIONS),
        len(WANT),  # we condition on partner's last WANT too (optional)
    ),
    dtype=np.float32,
)
q_sig_trade = np.zeros(
    (
        WORLD_SIZE,
        MAX_ENERGY + 1,
        MAX_HYDRATION + 1,
        len(EMOTIONS),
        len(TRADE),  # and on partner's last TRADE
    ),
    dtype=np.float32,
)

# ==========================
# === Simulation Logging ===
# ==========================
simulation_log = []  # stores final episode trajectory for visualization


# ==========================
# === Utility Functions  ===
# ==========================
def quantize_e(x):
    return int(min(max(int(round(x)), 0), MAX_ENERGY))


def quantize_h(x):
    return int(min(max(int(round(x)), 0), MAX_HYDRATION))


def move_cost(action, is_night):
    base = MOVE_COST if action in (ACT_LEFT, ACT_RIGHT) else IDLE_INTERACT_COST
    return base + (NIGHT_MOVE_PENALTY if is_night and action in (ACT_LEFT, ACT_RIGHT) else 0.0)


def get_new_pos(pos, action):
    if action == ACT_LEFT:
        return max(0, pos - 1)
    if action == ACT_RIGHT:
        return min(WORLD_SIZE - 1, pos + 1)
    return pos  # interact does not move


def update_emotion(energy, hydration, pos, other_pos, event):
    if event == "mutual_aid":
        return "happy"
    elif event == "rejected":
        return "angry"
    elif energy < 3 or hydration < 3:
        return "anxious"
    elif abs(pos - other_pos) > 2:
        return "lonely"
    return "neutral"


def action_to_str(a):
    return ["←", "→", "🤝"][a]


# ============================
# === Policy: Signals First ===
# ============================

def choose_want(pos, energy, hydration, emotion, partner_want, eps):
    e_i = emotion_to_idx[emotion]
    pw_i = want_to_idx[partner_want]
    if random.random() < eps:
        return random.choice(WANT)
    q_vals = q_sig_want[pos, quantize_e(energy), quantize_h(hydration), e_i, pw_i]
    return WANT[int(np.argmax(q_vals))]


def choose_trade(pos, energy, hydration, emotion, partner_trade, eps):
    e_i = emotion_to_idx[emotion]
    pt_i = trade_to_idx[partner_trade]
    if random.random() < eps:
        return random.choice(TRADE)
    q_vals = q_sig_trade[pos, quantize_e(energy), quantize_h(hydration), e_i, pt_i]
    return TRADE[int(np.argmax(q_vals))]


def choose_action_conditional(pos, energy, hydration, emotion, other_want, other_trade, eps):
    e_i = emotion_to_idx[emotion]
    ow_i = want_to_idx[other_want]
    ot_i = trade_to_idx[other_trade]
    if random.random() < eps:
        return random.choice(ACTIONS)
    q_vals = q_act[pos, quantize_e(energy), quantize_h(hydration), e_i, ow_i, ot_i]
    return int(np.argmax(q_vals))


# =============================
# === Bootstraps & Updates  ===
# =============================

def act_bootstrap(new_pos, new_e, new_h, new_emo, other_want_next, other_trade_next, eps):
    e_i = emotion_to_idx[new_emo]
    ow_i = want_to_idx[other_want_next]
    ot_i = trade_to_idx[other_trade_next]
    if USE_SARSA:
        a_next = choose_action_conditional(new_pos, new_e, new_h, new_emo, other_want_next, other_trade_next, eps)
        return q_act[new_pos, quantize_e(new_e), quantize_h(new_h), e_i, ow_i, ot_i, a_next]
    else:
        return np.max(q_act[new_pos, quantize_e(new_e), quantize_h(new_h), e_i, ow_i, ot_i])


def sig_w_bootstrap(new_pos, new_e, new_h, new_emo, partner_want_next, eps):
    e_i = emotion_to_idx[new_emo]
    if USE_SARSA:
        s_next = choose_want(new_pos, new_e, new_h, new_emo, partner_want_next, eps)
        return q_sig_want[new_pos, quantize_e(new_e), quantize_h(new_h), e_i, want_to_idx[s_next]]
    else:
        return np.max(q_sig_want[new_pos, quantize_e(new_e), quantize_h(new_h), e_i])


def sig_t_bootstrap(new_pos, new_e, new_h, new_emo, partner_trade_next, eps):
    e_i = emotion_to_idx[new_emo]
    if USE_SARSA:
        s_next = choose_trade(new_pos, new_e, new_h, new_emo, partner_trade_next, eps)
        return q_sig_trade[new_pos, quantize_e(new_e), quantize_h(new_h), e_i, trade_to_idx[s_next]]
    else:
        return np.max(q_sig_trade[new_pos, quantize_e(new_e), quantize_h(new_h), e_i])


def update_act(prev_pos, prev_e, prev_h, prev_emo, other_want_prev, other_trade_prev, act, reward,
               new_pos, new_e, new_h, new_emo, other_want_next, other_trade_next, terminal, eps):
    e_i = emotion_to_idx[prev_emo]
    ow_i = want_to_idx[other_want_prev]
    ot_i = trade_to_idx[other_trade_prev]
    q = q_act[prev_pos, quantize_e(prev_e), quantize_h(prev_h), e_i, ow_i, ot_i, act]
    target = reward if terminal else reward + gamma * act_bootstrap(new_pos, new_e, new_h, new_emo, other_want_next, other_trade_next, eps)
    q_act[prev_pos, quantize_e(prev_e), quantize_h(prev_h), e_i, ow_i, ot_i, act] += alpha * (target - q)


def update_sig_w(prev_pos, prev_e, prev_h, prev_emo, chosen_want, reward,
                 new_pos, new_e, new_h, new_emo, partner_want_next, terminal, eps):
    e_i = emotion_to_idx[prev_emo]
    cw_i = want_to_idx[chosen_want]
    q = q_sig_want[prev_pos, quantize_e(prev_e), quantize_h(prev_h), e_i, cw_i]
    target = reward if terminal else reward + gamma * sig_w_bootstrap(new_pos, new_e, new_h, new_emo, partner_want_next, eps)
    q_sig_want[prev_pos, quantize_e(prev_e), quantize_h(prev_h), e_i, cw_i] += alpha * (target - q)


def update_sig_t(prev_pos, prev_e, prev_h, prev_emo, chosen_trade, reward,
                 new_pos, new_e, new_h, new_emo, partner_trade_next, terminal, eps):
    e_i = emotion_to_idx[prev_emo]
    ct_i = trade_to_idx[chosen_trade]
    q = q_sig_trade[prev_pos, quantize_e(prev_e), quantize_h(prev_h), e_i, ct_i]
    target = reward if terminal else reward + gamma * sig_t_bootstrap(new_pos, new_e, new_h, new_emo, partner_trade_next, eps)
    q_sig_trade[prev_pos, quantize_e(prev_e), quantize_h(prev_h), e_i, ct_i] += alpha * (target - q)


# ======================
# === Helpers        ===
# ======================

def need_target(energy, hydration):
    if hydration < energy:
        return "water", WATER_POS
    else:
        return "food", FOOD_POS


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def logistic(x):
    # Smooth mapping to 0..1 for reputation
    return 1 / (1 + np.exp(-x))


# ======================
# === Training Loop  ===
# ======================

def train():
    global epsilon
    for ep in range(max_episodes):
        # Initial states
        a_pos, b_pos = 0, WORLD_SIZE - 1
        a_e = MAX_ENERGY
        a_h = MAX_HYDRATION
        b_e = MAX_ENERGY
        b_h = MAX_HYDRATION

        a_emo = "neutral"
        b_emo = "neutral"

        # Last seen partner slots (start as none)
        a_seen_want, a_seen_trade = "none", "none"  # what A last heard from B
        b_seen_want, b_seen_trade = "none", "none"  # what B last heard from A

        # Respawn timers and debuffs
        food_timer = 0
        water_timer = 0
        food_debuff = 0
        water_debuff = 0

        # Reputation & per-episode give caps
        reputation = 0.0  # grows on successful cooperative interactions
        a_given_food_total = 0  # A can give FOOD only (cannot give water)
        b_given_water_total = 0 # B can give WATER only (cannot give food)

        steps = 0
        record = (ep == max_episodes - 1)

        while a_e > 0 and a_h > 0 and b_e > 0 and b_h > 0 and steps < max_steps_per_episode:
            is_night = (steps % (2 * DAY_LEN_STEPS)) >= DAY_LEN_STEPS

            # Store previous
            prev_a_pos, prev_a_e, prev_a_h, prev_a_emo = a_pos, a_e, a_h, a_emo
            prev_b_pos, prev_b_e, prev_b_h, prev_b_emo = b_pos, b_e, b_h, b_emo

            # --- Comms ---
            a_want = choose_want(a_pos, a_e, a_h, a_emo, a_seen_want, epsilon)
            a_trade = choose_trade(a_pos, a_e, a_h, a_emo, a_seen_trade, epsilon)
            b_want = choose_want(b_pos, b_e, b_h, b_emo, b_seen_want, epsilon)
            b_trade = choose_trade(b_pos, b_e, b_h, b_emo, b_seen_trade, epsilon)

            a_comm_cost = 0.02 * ((a_want != "none") + (a_trade != "none"))
            b_comm_cost = 0.02 * ((b_want != "none") + (b_trade != "none"))

            # Hear partner
            a_seen_want, a_seen_trade = b_want, b_trade
            b_seen_want, b_seen_trade = a_want, a_trade

            # --- Action phase ---
            a_act = choose_action_conditional(a_pos, a_e, a_h, a_emo, a_seen_want, a_seen_trade, epsilon)
            b_act = choose_action_conditional(b_pos, b_e, b_h, b_emo, b_seen_want, b_seen_trade, epsilon)

            na_pos = get_new_pos(a_pos, a_act)
            nb_pos = get_new_pos(b_pos, b_act)

            # Head-on swap blocking
            if na_pos == b_pos and nb_pos == a_pos:
                na_pos, nb_pos = a_pos, b_pos

            # Distance shaping (based on need)
            a_need, a_target = need_target(a_e, a_h)
            b_need, b_target = need_target(b_e, b_h)
            a_prev_dist = abs(a_pos - a_target)
            b_prev_dist = abs(b_pos - b_target)
            a_new_dist = abs(na_pos - a_target)
            b_new_dist = abs(nb_pos - b_target)

            a_rew = STEP_REWARD + APPROACH_SHAPING * (a_prev_dist - a_new_dist)
            b_rew = STEP_REWARD + APPROACH_SHAPING * (b_prev_dist - b_new_dist)

            if abs(na_pos - nb_pos) == 1:
                a_rew += 0.05
                b_rew += 0.05

            a_rew += emotion_bonus(prev_a_emo)
            b_rew += emotion_bonus(prev_b_emo)

            a_rew -= a_comm_cost
            b_rew -= b_comm_cost

            # --- Metabolism with grace period ---
            grace_scale = GRACE_SCALE if steps < GRACE_STEPS else 1.0
            a_e = max(0.0, a_e - grace_scale * move_cost(a_act, is_night))
            b_e = max(0.0, b_e - grace_scale * move_cost(b_act, is_night))
            a_h = max(0.0, a_h - grace_scale * THIRST_DRAIN)
            b_h = max(0.0, b_h - grace_scale * THIRST_DRAIN)

            # --- Random events: resource loss ---
            loss_a = False
            loss_b = False
            if random.random() < EVENT_PROB_RESOURCE_LOSS:
                if random.random() < 0.5:
                    a_e = max(0.0, a_e - 1)
                else:
                    a_h = max(0.0, a_h - 1)
                loss_a = True
                a_rew -= 0.05
            if random.random() < EVENT_PROB_RESOURCE_LOSS:
                if random.random() < 0.5:
                    b_e = max(0.0, b_e - 1)
                else:
                    b_h = max(0.0, b_h - 1)
                loss_b = True
                b_rew -= 0.05

            # --- Random events: tile debuffs ---
            # Independent chances for food tile and water tile to be weak for a few steps
            if food_debuff == 0 and random.random() < EVENT_PROB_TILE_DEBUFF:
                food_debuff = random.randint(DEBUFF_MIN_STEPS, DEBUFF_MAX_STEPS)
            if water_debuff == 0 and random.random() < EVENT_PROB_TILE_DEBUFF:
                water_debuff = random.randint(DEBUFF_MIN_STEPS, DEBUFF_MAX_STEPS)

            # =============================
            # === ASYMMETRIC HARVESTING ===
            # =============================
            a_event = "none"
            b_event = "none"

            if na_pos == FOOD_POS and a_act == ACT_INTERACT and food_timer == 0:
                gain = INITIAL_REWARD * (DEBUFF_MULT if food_debuff > 0 else 1.0)
                a_e = min(MAX_ENERGY, a_e + gain)
                food_timer = FOOD_RESPAWN_STEPS
            if nb_pos == WATER_POS and b_act == ACT_INTERACT and water_timer == 0:
                gain = INITIAL_REWARD * (DEBUFF_MULT if water_debuff > 0 else 1.0)
                b_h = min(MAX_HYDRATION, b_h + gain)
                water_timer = WATER_RESPAWN_STEPS

            # Respawn countdowns & debuff countdowns
            food_timer = max(0, food_timer - 1)
            water_timer = max(0, water_timer - 1)
            food_debuff = max(0, food_debuff - 1)
            water_debuff = max(0, water_debuff - 1)

            # --- Signal-shaped rendezvous bonuses ---
            if a_seen_want == "food" and abs(nb_pos - FOOD_POS) < abs(b_pos - FOOD_POS):
                b_rew += 0.05
            if a_seen_want == "water" and abs(nb_pos - WATER_POS) < abs(b_pos - WATER_POS):
                b_rew += 0.05
            if b_seen_want == "food" and abs(na_pos - FOOD_POS) < abs(a_pos - FOOD_POS):
                a_rew += 0.05
            if b_seen_want == "water" and abs(na_pos - WATER_POS) < abs(a_pos - WATER_POS):
                a_rew += 0.05

            # =====================
            # === TRADE LOGIC   ===
            # =====================
            if abs(na_pos - nb_pos) <= 1:
                a_offer = (a_act == ACT_INTERACT)
                b_offer = (b_act == ACT_INTERACT)

                # Intent shaping from trade slot
                if b_seen_trade in ("accept", "wanted") and a_offer:
                    a_rew += 0.04
                if a_seen_trade in ("accept", "wanted") and b_offer:
                    b_rew += 0.04
                if b_seen_trade == "refuse" and a_offer:
                    a_rew -= 0.04
                if a_seen_trade == "refuse" and b_offer:
                    b_rew -= 0.04

                # Penalize offering when the other didn't offer
                if a_offer and not b_offer:
                    a_rew -= FRIENDSHIP_COST
                    a_event = "rejected"
                if b_offer and not a_offer:
                    b_rew -= FRIENDSHIP_COST
                    b_event = "rejected"

                if a_offer and b_offer:
                    # They may exchange: A gives FOOD to B; B gives WATER to A
                    # Caps for this step/trade
                    a_step_cap_left = max(0, GIVE_CAP_PER_STEP)
                    b_step_cap_left = max(0, GIVE_CAP_PER_STEP)

                    # A can give FOOD only -> increases B's ENERGY
                    room_food_for_B = MAX_ENERGY - b_e
                    give_food_from_A = min(
                        BASE_TRADE_CHUNK,
                        GIVE_CAP_PER_TRADE,
                        a_step_cap_left,
                        max(0, GIVE_CAP_TOTAL - a_given_food_total),
                        max(0, int(a_e))
                    )
                    give_food_from_A = min(give_food_from_A, room_food_for_B)

                    # B can give WATER only -> increases A's HYDRATION
                    room_water_for_A = MAX_HYDRATION - a_h
                    give_water_from_B = min(
                        BASE_TRADE_CHUNK,
                        GIVE_CAP_PER_TRADE,
                        b_step_cap_left,
                        max(0, GIVE_CAP_TOTAL - b_given_water_total),
                        max(0, int(b_h))
                    )
                    give_water_from_B = min(give_water_from_B, room_water_for_A)

                    # Reputation-gated success probability (smooth 0.5..0.95)
                    base = 0.5
                    span = 0.45
                    success_prob = base + span * float(logistic(reputation / 3.0))

                    can_trade = (give_food_from_A > 0 and give_water_from_B > 0)
                    if can_trade and random.random() < success_prob:
                        # Execute trade
                        a_e -= give_food_from_A
                        b_e += give_food_from_A
                        a_given_food_total += give_food_from_A

                        b_h -= give_water_from_B
                        a_h += give_water_from_B
                        b_given_water_total += give_water_from_B

                        a_rew += 4.0
                        b_rew += 4.0
                        a_event = b_event = "mutual_aid"
                        reputation += 1.0

                        # Credit stated trade intents
                        if a_seen_trade in ("accept", "wanted"):
                            b_rew += 0.06
                        if b_seen_trade in ("accept", "wanted"):
                            a_rew += 0.06
                    else:
                        a_rew -= FRIENDSHIP_COST
                        b_rew -= FRIENDSHIP_COST
                        a_event = b_event = "rejected"
                        reputation -= 0.5

            # New emotions after events & resource updates
            a_emo_new = update_emotion(a_e, a_h, na_pos, nb_pos, a_event)
            b_emo_new = update_emotion(b_e, b_h, nb_pos, na_pos, b_event)

            # Terminal check BEFORE updates
            a_dead = (a_e <= 0.0 or a_h <= 0.0)
            b_dead = (b_e <= 0.0 or b_h <= 0.0)
            terminal = a_dead or b_dead or (steps + 1 >= max_steps_per_episode)

            # For bootstrapping next-step partner message, use current heard slots as proxy
            a_other_w_next, a_other_t_next = a_seen_want, a_seen_trade
            b_other_w_next, b_other_t_next = b_seen_want, b_seen_trade

            # Update action head (emotion + other-comms aware)
            update_act(
                prev_a_pos, prev_a_e, prev_a_h, prev_a_emo, a_seen_want, a_seen_trade, a_act, a_rew,
                na_pos, a_e, a_h, a_emo_new, a_other_w_next, a_other_t_next, terminal, epsilon
            )
            update_act(
                prev_b_pos, prev_b_e, prev_b_h, prev_b_emo, b_seen_want, b_seen_trade, b_act, b_rew,
                nb_pos, b_e, b_h, b_emo_new, b_other_w_next, b_other_t_next, terminal, epsilon
            )

            # Update signal heads (reward the utility of what you said)
            update_sig_w(
                prev_a_pos, prev_a_e, prev_a_h, prev_a_emo, a_want, a_rew,
                na_pos, a_e, a_h, a_emo_new, b_seen_want, terminal, epsilon
            )
            update_sig_w(
                prev_b_pos, prev_b_e, prev_b_h, prev_b_emo, b_want, b_rew,
                nb_pos, b_e, b_h, b_emo_new, a_seen_want, terminal, epsilon
            )
            update_sig_t(
                prev_a_pos, prev_a_e, prev_a_h, prev_a_emo, a_trade, a_rew,
                na_pos, a_e, a_h, a_emo_new, b_seen_trade, terminal, epsilon
            )
            update_sig_t(
                prev_b_pos, prev_b_e, prev_b_h, prev_b_emo, b_trade, b_rew,
                nb_pos, b_e, b_h, b_emo_new, a_seen_trade, terminal, epsilon
            )

            # Advance state
            a_pos, b_pos = na_pos, nb_pos
            a_emo, b_emo = a_emo_new, b_emo_new

            steps += 1

            # Record for visualization (last episode only)
            if record:
                simulation_log.append((
                    steps, a_pos, b_pos,
                    a_emo, b_emo,
                    action_to_str(a_act), action_to_str(b_act),
                    a_e, a_h, b_e, b_h,
                    a_event, b_event,
                    is_night,
                    a_want, a_trade, b_want, b_trade,
                    a_given_food_total, b_given_water_total,
                    reputation,
                    food_debuff, water_debuff,
                    loss_a, loss_b
                ))

            if terminal:
                break

        # Epsilon schedule
        epsilon = max(min_epsilon, epsilon * decay_rate)
        if exploration_burst_every and ep > 0 and (ep % exploration_burst_every == 0):
            epsilon = max(epsilon, exploration_burst_level)


# ================================
# === Visualization (ASCII)    ===
# ================================

import os
import time
import shutil


def _bar(val, maxval, length=16, fill="█", empty="·"):
    try:
        ratio = 0 if maxval <= 0 else float(max(0, min(maxval, val))) / float(maxval)
    except Exception:
        ratio = 0
    n_fill = int(round(length * ratio))
    return fill * n_fill + empty * (length - n_fill)


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _world_lines(ap, bp, food_debuff, water_debuff):
    # index row
    idx = " ".join(f"{i:2d}" for i in range(WORLD_SIZE))

    # tile row: mark food/water
    tiles = []
    for x in range(WORLD_SIZE):
        if x == FOOD_POS:
            t = "F"
            if food_debuff and food_debuff > 0:
                t += "↓"
        elif x == WATER_POS:
            t = "W"
            if water_debuff and water_debuff > 0:
                t += "↓"
        else:
            t = "."
        tiles.append(f"{t:2s}")
    tile_row = " ".join(tiles)

    # agent row
    cells = ["  "] * WORLD_SIZE
    if ap == bp:
        cells[ap] = "AB"
    else:
        cells[ap] = "A "
        cells[bp] = "B "
    agent_row = " ".join(cells)
    return idx, tile_row, agent_row


def _frame_to_text(frame):
    (
        step, ap, bp, aemo, bemo, aa, ba, ae, ah, be, bh,
        aev, bev, night, a_want, a_trade, b_want, b_trade,
        a_given_food_total, b_given_water_total, reputation,
        food_debuff, water_debuff, loss_a, loss_b,
    ) = frame

    term_w = shutil.get_terminal_size((100, 24)).columns
    hline = "-" * min(term_w, 100)

    daynight = "Night" if night else "Day"
    header = (
        f"Step {step:02d} | {daynight} | Rep: {reputation:+.2f}\n"
        f"A: {aa} ({aev}) | B: {ba} ({bev})"
    )

    idx, tile_row, agent_row = _world_lines(ap, bp, food_debuff, water_debuff)

    a_loss = " ⚠ loss" if loss_a else ""
    b_loss = " ⚠ loss" if loss_b else ""

    a_stats = (
        f"A [{aemo:<8}]  E:{_bar(ae, MAX_ENERGY)} {ae:5.1f}  "
        f"H:{_bar(ah, MAX_HYDRATION)} {ah:5.1f}  "
        f"want:{a_want:<5} trade:{a_trade:<7}{a_loss}"
    )
    b_stats = (
        f"B [{bemo:<8}]  E:{_bar(be, MAX_ENERGY)} {be:5.1f}  "
        f"H:{_bar(bh, MAX_HYDRATION)} {bh:5.1f}  "
        f"want:{b_want:<5} trade:{b_trade:<7}{b_loss}"
    )

    debuffs = []
    if food_debuff and food_debuff > 0:
        debuffs.append(f"Food tile debuffed (↓{food_debuff})")
    if water_debuff and water_debuff > 0:
        debuffs.append(f"Water tile debuffed (↓{water_debuff})")
    deb_line = " | ".join(debuffs) if debuffs else "(no debuffs)"

    totals = (
        f"Totals: A gave FOOD {int(a_given_food_total)}  |  "
        f"B gave WATER {int(b_given_water_total)}"
    )

    lines = [
        hline,
        header,
        hline,
        "Idx : " + idx,
        "Tile: " + tile_row,
        "Pos : " + agent_row,
        hline,
        a_stats,
        b_stats,
        hline,
        deb_line,
        totals,
        hline,
    ]
    return "\n".join(lines)


def visualize_simulation_text(log, interactive=True, fps=5):
    if not log:
        print("(No frames to display.)")
        return

    i = 0
    autoplay = False and not interactive
    delay = 1.0 / max(1, fps)

    while 0 <= i < len(log):
        _clear()
        print(_frame_to_text(log[i]))

        if not interactive and not autoplay:
            # one-shot print for non-interactive
            break

        if autoplay:
            time.sleep(delay)
            i += 1
            continue

        try:
            cmd = input("[Enter]=next  b=back  a=auto  q=quit > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()  # newline after Ctrl-D/C
            break

        if cmd == "":
            i += 1
        elif cmd == "b":
            i = max(0, i - 1)
        elif cmd == "a":
            autoplay = True
        elif cmd == "q":
            break
        else:
            # unknown command, ignore
            pass

    print("\n(end)")

# =====================
# === Entry Point   ===
# =====================
if __name__ == "__main__":
    print("Training...")
    train()
    print("Done training.")

    if len(simulation_log) == 0:
        print("Simulation log is empty. Agents likely failed to survive long enough.")
        print("Try tweaking MOVE_COST/THIRST_DRAIN or bump INITIAL_REWARD and max_steps_per_episode.")
    else:
        print("Visualizing...")
        visualize_simulation_text(simulation_log)
