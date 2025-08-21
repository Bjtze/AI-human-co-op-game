import numpy as np
import random
import os
import time
import shutil
from dataclasses import dataclass, field
import pygame  # used for interactive mode

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
INITIAL_REWARD = 15
BASE_TRADE_CHUNK = 3     # target chunk to exchange if possible
MOVE_COST = 0.3
IDLE_INTERACT_COST = 0.3
THIRST_DRAIN = 0.5
NIGHT_MOVE_PENALTY = 0.1
APPROACH_SHAPING = 0.06
max_steps_per_episode = 60

# Grace period to avoid early deaths during exploration
GRACE_STEPS = 10           # first N steps per episode have reduced drain
GRACE_SCALE = 0.5          # 50% metabolism in grace period

# Trade caps/limits
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
ACTIONS = [0, 1, 2, 3]  # 0 = left, 1 = right, 2 = interact, 3 = chat
ACT_LEFT, ACT_RIGHT, ACT_INTERACT, ACT_CHAT = 0, 1, 2, 3

# =================
# === Chat V2   ===
# =================
# Tokens (internal order) → Labels (UI)
CHAT_MESSAGES = [
    # greetings
    "hello", "hey", "good_morning",
    # requests / signals
    "help", "need_food", "need_water",
    # gratitude / apology
    "thanks", "thank_you", "sorry",
    # coordination
    "group_up", "where_are_you", "on_my_way",
    # safety / world
    "warn_monster",
    # teaching & social
    "teach", "reassure", "joke",
    # negotiation / commitment
    "promise_help", "promise_trade", "negotiate_trade", "later",
    # ack
    "ok", "got_it",
    # negative
    "insult",
]

CHAT_LABELS = {
    "hello":"Hello!", "hey":"Hey!", "good_morning":"Good morning",
    "help":"Help?", "need_food":"I need FOOD", "need_water":"I need WATER",
    "thanks":"Thanks!", "thank_you":"Thank you!", "sorry":"Sorry",
    "group_up":"Group up?", "where_are_you":"Where are you?", "on_my_way":"On my way",
    "warn_monster":"Monster nearby!", "teach":"I can teach you", "reassure":"We’ve got this",
    "joke":"😄 (joke)", "promise_help":"I promise to help", "promise_trade":"I promise to trade",
    "negotiate_trade":"Can we trade?", "later":"Maybe later",
    "ok":"OK", "got_it":"Got it", "insult":"…(insult)",
}

CHAT_GOOD = {"hello","hey","good_morning","thanks","thank_you","sorry","reassure","ok","got_it"}
CHAT_BAD  = {"insult"}

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

max_episodes = 10000

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
WANT = ["none", "food", "water"]
TRADE = ["none", "accept", "refuse", "wanted"]  # "wanted" = request trade
want_to_idx = {s: i for i, s in enumerate(WANT)}
trade_to_idx = {s: i for i, s in enumerate(TRADE)}

# =================
# === Q-Tables  ===
# =================
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

q_sig_want = np.zeros(
    (
        WORLD_SIZE,
        MAX_ENERGY + 1,
        MAX_HYDRATION + 1,
        len(EMOTIONS),
        len(WANT),
    ),
    dtype=np.float32,
)

q_sig_trade = np.zeros(
    (
        WORLD_SIZE,
        MAX_ENERGY + 1,
        MAX_HYDRATION + 1,
        len(EMOTIONS),
        len(TRADE),
    ),
    dtype=np.float32,
)

# Chat Q-table: choose chat message based on state (matches Chat V2 size)
q_chat = np.zeros(
    (
        WORLD_SIZE,
        MAX_ENERGY + 1,
        MAX_HYDRATION + 1,
        len(EMOTIONS),
        len(CHAT_MESSAGES),
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
    return pos  # interact and chat do not move

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def logistic(x):
    # Smooth mapping to 0..1 for reputation
    return 1 / (1 + np.exp(-x))

def role_target(role):
    # For shaping: A heads toward water; B heads toward food
    return WATER_POS if role == "A" else FOOD_POS

def print_full_episode(log):
    if not log:
        print("(No frames to print.)")
        return
    for frame in log:
        print(_frame_to_text(frame))

def print_chat_summary(log):
    from collections import Counter
    a_msgs = [frame[-2] for frame in log if frame[-2]]  # a_chat_msg
    b_msgs = [frame[-1] for frame in log if frame[-1]]  # b_chat_msg

    print("\n=== Chat Summary ===")
    print("A chat messages:")
    for msg, count in Counter(a_msgs).items():
        print(f"  {msg}: {count}")
    print("B chat messages:")
    for msg, count in Counter(b_msgs).items():
        print(f"  {msg}: {count}")
    print("====================\n")

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
    return ["←", "→", "🤝", "💬"][a]

# ============================
# === Policy Helpers       ===
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

def update_chat(prev_pos, prev_e, prev_h, prev_emo, chosen_msg, reward,
                new_pos, new_e, new_h, new_emo, terminal, eps):
    e_i = emotion_to_idx[prev_emo]
    m_i = CHAT_MESSAGES.index(chosen_msg)
    q  = q_chat[prev_pos, quantize_e(prev_e), quantize_h(prev_h), e_i, m_i]
    if USE_SARSA:
        next_msg = choose_chat(new_pos, new_e, new_h, new_emo, eps)
        nm_i = CHAT_MESSAGES.index(next_msg)
        target = reward if terminal else reward + gamma * q_chat[
            new_pos, quantize_e(new_e), quantize_h(new_h), emotion_to_idx[new_emo], nm_i
        ]
    else:
        target = reward if terminal else reward + gamma * np.max(
            q_chat[new_pos, quantize_e(new_e), quantize_h(new_h), emotion_to_idx[new_emo]]
        )
    q_chat[prev_pos, quantize_e(prev_e), quantize_h(prev_h), e_i, m_i] += alpha * (target - q)

# ======================
# === Helpers        ===
# ======================
def need_target(energy, hydration):
    # returns ("water" or "food", target_tile_pos) based on bars
    if hydration < energy:
        return "water", WATER_POS
    else:
        return "food", FOOD_POS

# ======================
# === Chat V2 State  ===
# ======================
@dataclass
class ChatState:
    last_step: dict = field(default_factory=dict)   # msg -> last step said
    owed_thanks_until: int = -1                     # schedule a future "thanks"
    promise_to_give_until: int = -1                 # soft deadline to give after "promise_*"
    last_insult_step: int = -9999

    def record(self, msg, step):
        self.last_step[msg] = step
        if msg == "insult":
            self.last_insult_step = step
        if msg in ("promise_help","promise_trade"):
            self.promise_to_give_until = max(self.promise_to_give_until, step + 3)  # keep promise within ~3 steps

# ======================
# === Chat V2 chooser ==
# ======================
def choose_chat(
    pos, energy, hydration, emotion, eps,
    chat_history=None,
    *,
    step=None,                 # int: current step
    adjacency=False,           # bool: next to partner?
    last_event=None,           # str: e.g. "mutual_aid", "rejected"
    reputation=0.0,            # float: shared rep [-5..+5]
    role=None,                 # "A" or "B"  (A can’t harvest FOOD, B can’t harvest WATER)
    self_perks=None,           # set[str]
    partner_perks=None,        # set[str]
    chat_state=None,           # ChatState
    partner_want=None,         # "none"|"food"|"water"  (optional hint)
    distance_to_partner=None,  # int (optional)
    monster_active=False,      # bool (optional)
    rng=random,
):
    """
    Chat V2:
      - Base Q-values (learned) + heuristic bonuses
      - Richer vocabulary grouped by intent
      - Cooldowns, anti-spam, emotion flavor
      - Context (need, adjacency, monster, promises, negotiation)
    """
    if chat_history is None:   chat_history = []
    if self_perks is None:     self_perks = set()
    if partner_perks is None:  partner_perks = set()

    e_i   = emotion_to_idx[emotion]
    base  = q_chat[pos, quantize_e(energy), quantize_h(hydration), e_i].astype(np.float64)
    scores = base.copy()

    msg_to_i = {m:i for i,m in enumerate(CHAT_MESSAGES)}
    bonus = np.zeros(len(CHAT_MESSAGES), dtype=np.float64)

    said = set(chat_history)
    now  = 0 if step is None else step

    # helpers
    def said_recently(msg, cd):
        if chat_state and msg in chat_state.last_step:
            return now - chat_state.last_step[msg] < cd
        return msg in said

    def boost(msg, v):
        if msg in msg_to_i: bonus[msg_to_i[msg]] += v

    def nerf(msg, v):
        if msg in msg_to_i: bonus[msg_to_i[msg]] -= v

    # groupings
    greetings = ["hello","hey","good_morning"]
    thankses  = ["thanks","thank_you"]
    req_food  = "need_food"
    req_watr  = "need_water"

    # discourage insult
    if "insult" in msg_to_i:
        insult_pen = 0.6
        if emotion == "angry" and last_event == "rejected":
            insult_pen = 0.35
        nerf("insult", insult_pen)

    # greetings early/adjacent, heavy anti-spam
    if adjacency and (step is None or step < 8) and not any(said_recently(g, 40) for g in greetings):
        for g in greetings: boost(g, 0.7)
    else:
        for g in greetings: nerf(g, 0.8)

    # thanks after mutual aid or owed_thanks
    owe = (last_event == "mutual_aid") or (chat_state and chat_state.owed_thanks_until >= now)
    if owe:
        for t in thankses: boost(t, 1.0)
    else:
        for t in thankses:
            if said_recently(t, 8): nerf(t, 0.8)

    # apology after insult/rejection
    if "sorry" in msg_to_i:
        recently_rude = chat_state and now - chat_state.last_insult_step <= 6
        if recently_rude or last_event == "rejected":
            boost("sorry", 0.8)
        elif said_recently("sorry", 8):
            nerf("sorry", 0.6)

    # need & role-aware asks
    need, _ = need_target(energy, hydration)  # "food"|"water"
    if role == "A":   need_other = (need == "food")
    elif role == "B": need_other = (need == "water")
    else:             need_other = (need == "food" or need == "water")

    if need_other:
        urg = 0.9 if (energy < 3 or hydration < 3) else 0.5
        if adjacency:
            if role == "A":
                boost(req_food, urg)
            elif role == "B":
                boost(req_watr, urg)
            else:
                boost("help", urg)
        else:
            if distance_to_partner is not None and distance_to_partner > 1:
                boost("group_up", 0.7)
                boost("where_are_you", 0.4)
            else:
                boost("help", 0.3)
        if role == "A" and said_recently(req_food, 4): nerf(req_food, 0.6)
        if role == "B" and said_recently(req_watr, 4): nerf(req_watr, 0.6)
        if said_recently("help", 4):                    nerf("help", 0.6)

    # monster context
    if monster_active:
        boost("warn_monster", 1.0)
        boost("group_up", 0.8)

    # negotiation / promise (rep-gated)
    rep01 = max(0.0, min(1.0, (reputation + 5.0) / 10.0))  # [-5..+5] → [0..1]
    if adjacency and rep01 >= 0.6:
        boost("negotiate_trade", 0.4)
        boost("promise_help", 0.4)
        if need_other:
            boost("promise_trade", 0.5)
    elif adjacency and 0.3 <= rep01 < 0.6:
        boost("negotiate_trade", 0.25)

    # promise window → de-emphasize idle/jokes, emphasize movement/ok
    if chat_state and chat_state.promise_to_give_until >= now:
        nerf("joke", 0.8)
        nerf("later", 0.6)
        boost("on_my_way", 0.4)
        boost("ok", 0.3)

    # emotion flavors (tiny)
    emo_nudge = {
        "happy":   {"hello": +0.15, "thanks": +0.15, "joke": +0.1},
        "anxious": {"help": +0.15, "group_up": +0.1},
        "angry":   {"insult": +0.05, "negotiate_trade": -0.05},
        "lonely":  {"hello": +0.1, "where_are_you": +0.15, "group_up": +0.15},
        "neutral": {},
    }.get(emotion, {})
    for m, v in emo_nudge.items():
        boost(m, v)

    # anti-repeat
    if chat_history:
        last_msg = chat_history[-1]
        nerf(last_msg, 0.9)

    # final scores
    scores = scores + bonus

    # ε-greedy with softmax exploration
    if rng.random() < eps:
        t = 0.7
        ex = np.exp((scores - np.max(scores)) / t)
        ex = np.clip(ex, 1e-6, None)
        probs = ex / np.sum(ex)
        choice = int(np.random.choice(len(CHAT_MESSAGES), p=probs))
    else:
        choice = int(np.argmax(scores))
    msg = CHAT_MESSAGES[choice]

    # bookkeeping
    if chat_state is not None and step is not None:
        chat_state.record(msg, step)
        if last_event == "mutual_aid":
            chat_state.owed_thanks_until = max(chat_state.owed_thanks_until, step + 2)

    return msg

# ======================
# === Training Loop  ===
# ======================
def train():
    global epsilon, simulation_log
    simulation_log = []
    for ep in range(max_episodes):
        # Initial states
        a_pos, b_pos = 0, WORLD_SIZE - 1
        a_e = float(MAX_ENERGY)
        a_h = float(MAX_HYDRATION)
        b_e = float(MAX_ENERGY)
        b_h = float(MAX_HYDRATION)

        a_emo = "neutral"
        b_emo = "neutral"

        a_perks = random.choice([{"extra_food_space"}, {"rep_boost"}])
        b_perks = random.choice([{"extra_food_space"}, {"rep_boost"}])

        # chat states
        a_chat_state = ChatState()
        b_chat_state = ChatState()

        # Last seen partner slots (start as none)
        a_seen_want, a_seen_trade = "none", "none"
        b_seen_want, b_seen_trade = "none", "none"

        # Respawn timers and debuffs
        food_timer = 0
        water_timer = 0
        food_debuff = 0
        water_debuff = 0

        # Reputation & per-episode give caps
        reputation = 0.0
        a_given_food_total = 0
        b_given_water_total = 0

        # Chat history for structure rewards
        a_chat_history = []
        b_chat_history = []

        steps = 0
        record = (ep == max_episodes - 1)

        while a_e > 0 and a_h > 0 and b_e > 0 and b_h > 0 and steps < max_steps_per_episode:
            a_event = "none"
            b_event = "none"
            is_night = (steps % (2 * DAY_LEN_STEPS)) >= DAY_LEN_STEPS

            # Store previous
            prev_a_pos, prev_a_e, prev_a_h, prev_a_emo = a_pos, a_e, a_h, a_emo
            prev_b_pos, prev_b_e, prev_b_h, prev_b_emo = b_pos, b_e, b_h, b_emo

            # Per-step reward accumulators
            a_rew = STEP_REWARD
            b_rew = STEP_REWARD

            # Per-step trade caps
            step_give_food = 0
            step_give_water = 0

            # --- Comms (signals) ---
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

            # --- Chat phase ---
            a_chat_msg = None
            b_chat_msg = None
            if a_act == ACT_CHAT:
                a_chat_msg = choose_chat(
                    a_pos, a_e, a_h, a_emo, epsilon, a_chat_history,
                    step=steps,
                    adjacency=(abs(a_pos - b_pos) <= 1),
                    last_event="none",
                    reputation=reputation,
                    role="A",
                    self_perks=a_perks, partner_perks=b_perks,
                    chat_state=a_chat_state,
                    partner_want=b_want,
                    distance_to_partner=abs(a_pos - b_pos),
                    monster_active=False,
                )
                a_chat_history.append(a_chat_msg)
            if b_act == ACT_CHAT:
                b_chat_msg = choose_chat(
                    b_pos, b_e, b_h, b_emo, epsilon, b_chat_history,
                    step=steps,
                    adjacency=(abs(a_pos - b_pos) <= 1),
                    last_event="none",
                    reputation=reputation,
                    role="B",
                    self_perks=b_perks, partner_perks=a_perks,
                    chat_state=b_chat_state,
                    partner_want=a_want,
                    distance_to_partner=abs(a_pos - b_pos),
                    monster_active=False,
                )
                b_chat_history.append(b_chat_msg)

            # --- Perk effects ---
            if "extra_food_space" in a_perks:
                a_e = min(MAX_ENERGY + 2, a_e)
            if "extra_food_space" in b_perks:
                b_e = min(MAX_ENERGY + 2, b_e)
            if "rep_boost" in a_perks and a_chat_msg in CHAT_GOOD:
                reputation += 0.2
            if "rep_boost" in b_perks and b_chat_msg in CHAT_GOOD:
                reputation += 0.2

            # Reputation change for chatting
            if a_chat_msg:
                if a_chat_msg in CHAT_GOOD:
                    reputation += 0.5
                    a_rew += 0.2
                    a_event = "chat_good"
                elif a_chat_msg in CHAT_BAD:
                    reputation -= 0.5
                    a_rew -= 0.2
                    a_event = "chat_bad"
            if b_chat_msg:
                if b_chat_msg in CHAT_GOOD:
                    reputation += 0.5
                    b_rew += 0.2
                    b_event = "chat_good"
                elif b_chat_msg in CHAT_BAD:
                    reputation -= 0.5
                    b_rew -= 0.2
                    b_event = "chat_bad"

            # Extra reward for good chat structure
            if a_chat_msg:
                if a_chat_msg in {"hello","hey","good_morning"} and len(a_chat_history) == 1:
                    a_rew += 0.15
                elif a_chat_msg in {"help","thanks","thank_you"} and any(x in {"hello","hey","good_morning"} for x in a_chat_history):
                    a_rew += 0.10
            if b_chat_msg:
                if b_chat_msg in {"hello","hey","good_morning"} and len(b_chat_history) == 1:
                    b_rew += 0.15
                elif b_chat_msg in {"help","thanks","thank_you"} and any(x in {"hello","hey","good_morning"} for x in b_chat_history):
                    b_rew += 0.10

            # Proposed new positions
            na_pos = get_new_pos(a_pos, a_act)
            nb_pos = get_new_pos(b_pos, b_act)

            # Head-on swap blocking
            if na_pos == b_pos and nb_pos == a_pos:
                na_pos, nb_pos = a_pos, b_pos

            # Role-based approach shaping (A->water tile, B->food tile)
            a_target = role_target("A")
            b_target = role_target("B")
            a_prev_dist = abs(a_pos - a_target)
            b_prev_dist = abs(b_pos - b_target)
            a_new_dist = abs(na_pos - a_target)
            b_new_dist = abs(nb_pos - b_target)

            if not a_chat_msg:
                a_rew += APPROACH_SHAPING * (a_prev_dist - a_new_dist)
            if not b_chat_msg:
                b_rew += APPROACH_SHAPING * (b_prev_dist - b_new_dist)

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
            if food_debuff == 0 and random.random() < EVENT_PROB_TILE_DEBUFF:
                food_debuff = random.randint(DEBUFF_MIN_STEPS, DEBUFF_MAX_STEPS)
            if water_debuff == 0 and random.random() < EVENT_PROB_TILE_DEBUFF:
                water_debuff = random.randint(DEBUFF_MIN_STEPS, DEBUFF_MAX_STEPS)

            # --- ASYMMETRIC HARVESTING ---
            # A can harvest WATER only
            if na_pos == WATER_POS and a_act == ACT_INTERACT and water_timer == 0:
                gain = INITIAL_REWARD * (DEBUFF_MULT if water_debuff > 0 else 1.0)
                a_h = min(MAX_HYDRATION, a_h + gain)
                water_timer = WATER_RESPAWN_STEPS

            # B can harvest FOOD only
            if nb_pos == FOOD_POS and b_act == ACT_INTERACT and food_timer == 0:
                gain = INITIAL_REWARD * (DEBUFF_MULT if food_debuff > 0 else 1.0)
                b_e = min(MAX_ENERGY, b_e + gain)
                food_timer = FOOD_RESPAWN_STEPS

            # Respawn countdowns & debuff countdowns
            food_timer = max(0, food_timer - 1)
            water_timer = max(0, water_timer - 1)
            food_debuff = max(0, food_debuff - 1)
            water_debuff = max(0, water_debuff - 1)

            # --- Signal-shaped rendezvous bonuses ---
            if a_seen_want == "food" and abs(nb_pos - FOOD_POS) < abs(b_pos - FOOD_POS):
                b_rew += 0.05
            if b_seen_want == "water" and abs(na_pos - WATER_POS) < abs(a_pos - WATER_POS):
                a_rew += 0.05

            # --- Willing resource sharing logic (ASYMMETRIC) ---
            # A can ONLY GIVE WATER; B can ONLY GIVE FOOD.
            if abs(na_pos - nb_pos) <= 1 and a_act == ACT_INTERACT and b_act == ACT_INTERACT:
                base_prob = 0.2
                max_prob = 0.95
                success_prob = base_prob + (max_prob - base_prob) * min(1.0, reputation / 5.0)

                # If B wants water, A may give water (hydration) — giver gains reputation ONLY
                if b_want == "water":
                    give_water = min(BASE_TRADE_CHUNK, GIVE_CAP_PER_TRADE,
                                     GIVE_CAP_PER_STEP - step_give_water,
                                     GIVE_CAP_TOTAL - 0,
                                     int(a_h))
                    if give_water > 0 and random.random() < success_prob:
                        a_h -= give_water
                        b_h += give_water
                        step_give_water += give_water
                        reputation = clamp(reputation + 1.0, -5.0, 5.0)  # giver A
                        a_event = b_event = "mutual_aid"
                    else:
                        a_event = b_event = "rejected"

                # If A wants food, B may give food (energy) — giver gains reputation ONLY
                if a_want == "food":
                    give_food = min(BASE_TRADE_CHUNK, GIVE_CAP_PER_TRADE,
                                    GIVE_CAP_PER_STEP - step_give_food,
                                    GIVE_CAP_TOTAL - 0,
                                    int(b_e))
                    if give_food > 0 and random.random() < success_prob:
                        b_e -= give_food
                        a_e += give_food
                        step_give_food += give_food
                        reputation = clamp(reputation + 1.0, -5.0, 5.0)  # giver B
                        a_event = b_event = "mutual_aid"
                    else:
                        a_event = b_event = "rejected"

            # clamp reputation
            reputation = clamp(reputation, -5.0, 5.0)

            # ======= RL UPDATES =======
            terminal = (a_e <= 0 or a_h <= 0 or b_e <= 0 or b_h <= 0 or steps+1 >= max_steps_per_episode)

            a_want_next, a_trade_next = a_want, a_trade
            b_want_next, b_trade_next = b_want, b_trade

            a_emo_next = update_emotion(a_e, a_h, na_pos, nb_pos, a_event)
            b_emo_next = update_emotion(b_e, b_h, nb_pos, na_pos, b_event)

            update_sig_w(prev_a_pos, prev_a_e, prev_a_h, prev_a_emo, a_want, a_rew,
                         na_pos, a_e, a_h, a_emo_next, b_want_next, terminal, epsilon)
            update_sig_t(prev_a_pos, prev_a_e, prev_a_h, prev_a_emo, a_trade, a_rew,
                         na_pos, a_e, a_h, a_emo_next, b_trade_next, terminal, epsilon)

            update_sig_w(prev_b_pos, prev_b_e, prev_b_h, prev_b_emo, b_want, b_rew,
                         nb_pos, b_e, b_h, b_emo_next, a_want_next, terminal, epsilon)
            update_sig_t(prev_b_pos, prev_b_e, prev_b_h, prev_b_emo, b_trade, b_rew,
                         nb_pos, b_e, b_h, b_emo_next, a_trade_next, terminal, epsilon)

            update_act(prev_a_pos, prev_a_e, prev_a_h, prev_a_emo, a_seen_want, a_seen_trade, a_act, a_rew,
                       na_pos, a_e, a_h, a_emo_next, b_want_next, b_trade_next, terminal, epsilon)
            update_act(prev_b_pos, prev_b_e, prev_b_h, prev_b_emo, b_seen_want, b_seen_trade, b_act, b_rew,
                       nb_pos, b_e, b_h, b_emo_next, a_want_next, a_trade_next, terminal, epsilon)

            if a_chat_msg:
                update_chat(prev_a_pos, prev_a_e, prev_a_h, prev_a_emo, a_chat_msg, a_rew,
                            na_pos, a_e, a_h, a_emo_next, terminal, epsilon)
            if b_chat_msg:
                update_chat(prev_b_pos, prev_b_e, prev_b_h, prev_b_emo, b_chat_msg, b_rew,
                            nb_pos, b_e, b_h, b_emo_next, terminal, epsilon)

            # Advance to new state
            a_pos, b_pos = na_pos, nb_pos
            a_emo, b_emo = a_emo_next, b_emo_next

            if record:
                simulation_log.append((
                    steps, a_pos, b_pos, a_emo, b_emo, a_act, b_act, a_e, a_h, b_e, b_h,
                    a_event, b_event, is_night, a_want, a_trade, b_want, b_trade,
                    a_given_food_total, b_given_water_total, reputation,
                    food_debuff, water_debuff, loss_a, loss_b,
                    a_chat_msg, b_chat_msg
                ))

            steps += 1

        # Epsilon schedule per episode
        if (ep + 1) % exploration_burst_every == 0:
            epsilon = max(exploration_burst_level, epsilon)
        else:
            epsilon = max(min_epsilon, epsilon * decay_rate)

# ================================
# === Visualization (ASCII)    ===
# ================================
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
    idx = " ".join(f"{i:2d}" for i in range(WORLD_SIZE))
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
        a_chat_msg, b_chat_msg
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

    chat_line = f"A chat: {a_chat_msg or '-'} | B chat: {b_chat_msg or '-'}"

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
        chat_line,
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
            break

        if autoplay:
            time.sleep(delay)
            i += 1
            continue

        try:
            cmd = input("[Enter]=next  b=back  a=auto  q=quit > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
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
            pass

    print("\n(end)")

# =======================
# === Interactive UI  ===
# =======================
def play_interactive_round():
    import pygame, random, re
    pygame.init()
    pygame.key.set_repeat(0)

    # ---------- Window & fonts ----------
    W, H = 1120, 760
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Co-op Survival — You (B) + Bot (A) + Monster")
    clock = pygame.time.Clock()

    font_xs = pygame.font.SysFont(None, 16)
    font_sm = pygame.font.SysFont(None, 20)
    font_md = pygame.font.SysFont(None, 26)
    font_lg = pygame.font.SysFont(None, 34)

    # ---------- Colors ----------
    BG      = (20, 22, 26)
    PANEL   = (36, 39, 46)
    CARD    = (44, 48, 56)
    STROKE  = (70, 74, 82)
    TEXT    = (235, 238, 245)
    MUTED   = (172, 176, 186)
    ACCENT  = (88, 160, 255)
    OK      = (72, 201, 120)
    BAD     = (240, 95, 80)
    BLUE    = (80, 170, 255)
    ORANGE  = (255, 170, 80)
    MONST   = (200, 80, 120)

    # ---------- UI Helpers ----------
    def panel(r):
        pygame.draw.rect(screen, PANEL, r, border_radius=14)
        pygame.draw.rect(screen, STROKE, r, 2, border_radius=14)

    def card(r):
        pygame.draw.rect(screen, CARD, r, border_radius=12)
        pygame.draw.rect(screen, STROKE, r, 1, border_radius=12)

    def draw_bar(x, y, w, h, frac, col=OK):
        frac = max(0.0, min(1.0, frac))
        pygame.draw.rect(screen, (60,62,70), (x, y, w, h), border_radius=6)
        pygame.draw.rect(screen, (95,98,108), (x, y, w, h), 2, border_radius=6)
        if frac > 0:
            pygame.draw.rect(screen, col, (x+2, y+2, int((w-4)*frac), h-4), border_radius=6)

    class Button:
        def __init__(self, rect, label, kind="primary", hotkey=None):
            self.rect  = pygame.Rect(rect)
            self.label = label
            self.kind  = kind  # "primary" | "secondary" | "danger"
            self.down  = False
            self.hotkey = hotkey
        def draw(self):
            base = {"primary": ACCENT, "secondary": (64, 69, 79), "danger": BAD}[self.kind]
            col = tuple(max(0, c-22) for c in base) if self.down else base
            pygame.draw.rect(screen, col, self.rect, border_radius=10)
            pygame.draw.rect(screen, (25, 25, 30), self.rect, 2, border_radius=10)
            txt = font_md.render(self.label, True, TEXT)
            screen.blit(txt, txt.get_rect(center=self.rect.center))
        def handle(self, e):
            clicked = False
            if e.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(e.pos):
                self.down = True
            if e.type == pygame.MOUSEBUTTONUP:
                if self.down and self.rect.collidepoint(e.pos):
                    clicked = True
                self.down = False
            if e.type == pygame.KEYDOWN and self.hotkey and e.key == self.hotkey:
                clicked = True
            return clicked

    class Log:
        def __init__(self, cap=12):
            self.lines = []
            self.cap = cap
            self._last_msg = None
            self._last_ms  = -9999
        def add(self, s):
            now = pygame.time.get_ticks()
            if s == self._last_msg and (now - self._last_ms) < 350:
                return
            self._last_msg, self._last_ms = s, now
            self.lines.append(s)
            if len(self.lines) > self.cap:
                self.lines = self.lines[-self.cap:]
        def draw(self, r):
            card(r)
            y = r.y + 8
            for ln in self.lines:
                screen.blit(font_sm.render(ln, True, TEXT), (r.x+10, y))
                y += 20

    # Scrollable chat with input
    class ChatPanel:
        def __init__(self, rect):
            self.rect = pygame.Rect(rect)
            self.msg_area = pygame.Rect(self.rect.x+10, self.rect.y+10, self.rect.width-20, self.rect.height-70)
            self.input_rect = pygame.Rect(self.rect.x+10, self.rect.bottom-52, self.rect.width-120, 40)
            self.send_btn   = Button((self.rect.right-96, self.rect.bottom-52, 86, 40), "Send", "primary", hotkey=pygame.K_RETURN)
            self.focus = False
            self.input_text = ""
            self.messages = []
            self.scroll = 0
        def add(self, speaker, text):
            self.messages.append((speaker, text))
            self.scroll = 0
        def draw(self):
            panel(self.rect)
            title = font_md.render("Chat", True, TEXT)
            screen.blit(title, (self.rect.x+12, self.rect.y-28))
            card(self.msg_area)
            clip_prev = screen.get_clip()
            screen.set_clip(self.msg_area)
            line_h = 20
            max_lines = self.msg_area.height // line_h
            start = max(0, len(self.messages) - max_lines - self.scroll)
            end   = min(len(self.messages), start + max_lines)
            y = self.msg_area.y + 6
            for (spk, txt) in self.messages[start:end]:
                color = OK if spk == "You" else ACCENT if spk == "A" else TEXT
                line = font_sm.render(f"{spk}: {txt}", True, color)
                screen.blit(line, (self.msg_area.x+8, y))
                y += line_h
            screen.set_clip(clip_prev)
            pygame.draw.rect(screen, (60,62,70), self.input_rect, border_radius=8)
            pygame.draw.rect(screen, (95,98,108), self.input_rect, 2, border_radius=8)
            itxt = self.input_text if self.focus else (self.input_text or "type here…")
            col  = TEXT if self.focus or self.input_text else MUTED
            screen.blit(font_sm.render(itxt, True, col), (self.input_rect.x+8, self.input_rect.y+10))
            self.send_btn.draw()
        def handle_event(self, e):
            sent = None
            if e.type == pygame.MOUSEBUTTONDOWN:
                self.focus = self.input_rect.collidepoint(e.pos)
            if e.type == pygame.MOUSEWHEEL:
                self.scroll = max(0, self.scroll - e.y)
            if e.type == pygame.KEYDOWN and self.focus:
                if e.key == pygame.K_RETURN:
                    if self.input_text.strip():
                        sent = self.input_text.strip()
                        self.input_text = ""
                elif e.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                else:
                    ch = e.unicode
                    if ch and ch.isprintable():
                        self.input_text += ch
            if self.send_btn.handle(e):
                if self.input_text.strip():
                    sent = self.input_text.strip()
                    self.input_text = ""
            return sent

    # ---------- Layout ----------
    top_bar   = pygame.Rect(0, 0, W, 60)

    world_r   = pygame.Rect(20, 80, 680, 240)
    wants_r   = pygame.Rect(20, 330, 680, 110)
    status_r  = pygame.Rect(20, 450, 680, 160)

    right_r   = pygame.Rect(720, 80, 380, 330)     # controls
    chat_r    = pygame.Rect(720, 420, 380, 220)    # chat panel
    log_r     = pygame.Rect(20, 620, 1080, 120)

    # ---------- Controls ----------
    btn_left   = Button((right_r.x+12,  right_r.y+12,  60, 40), "←", "secondary", hotkey=pygame.K_LEFT)
    btn_stay   = Button((right_r.x+80,  right_r.y+12,  70, 40), "Stay", "secondary",
                        hotkey=(pygame.K_PERIOD if hasattr(pygame, "K_PERIOD") else pygame.K_s))
    btn_right  = Button((right_r.x+154, right_r.y+12,  60, 40), "→", "secondary", hotkey=pygame.K_RIGHT)
    btn_share  = Button((right_r.x+220, right_r.y+12, 148, 40), "Interact", "primary", hotkey=pygame.K_SPACE)

    btn_give   = Button((right_r.x+12,  right_r.y+62,  356, 40), "GIVE FOOD", "danger", hotkey=pygame.K_g)

    chat_y = right_r.y + 112
    chat_btns = [
        Button((right_r.x+12,  chat_y,      108, 34), "Hello",     "secondary"),
        Button((right_r.x+124, chat_y,      108, 34), "Thanks",    "secondary"),
        Button((right_r.x+236, chat_y,      132, 34), "Need WATER","secondary"),
        Button((right_r.x+12,  chat_y+40,   108, 34), "Group up?", "secondary"),
        Button((right_r.x+124, chat_y+40,   108, 34), "Sorry",     "secondary"),
        Button((right_r.x+236, chat_y+40,   132, 34), "Insult",    "secondary"),
    ]

    key_help = [
        "Keys: ← → move, SPACE interact, G give food, Enter send chat, MouseWheel scroll chat"
    ]

    # ---------- Game State ----------
    MAX_HP = 10
    a_hp = float(MAX_HP); b_hp = float(MAX_HP)

    a_pos, b_pos = 0, WORLD_SIZE - 1
    a_e = float(MAX_ENERGY); a_h = float(MAX_HYDRATION)
    b_e = float(MAX_ENERGY); b_h = float(MAX_HYDRATION)

    food_timer, water_timer = 0, 0
    reputation = 0.0
    steps = 0
    running = True

    a_emo = "neutral"; b_emo = "neutral"
    a_event = "none";  b_event = "none"

    b_want = "none"

    # Monster state
    MONSTER_SPAWN_PROB   = 0.22
    MONSTER_DAMAGE       = 3
    MONSTER_COOLDOWN     = 4
    MONSTER_REWARD_FOOD  = 4
    MONSTER_REWARD_REP   = 1.0
    monster_active   = False
    monster_defeated = False
    monster_pos      = -1
    monster_cd       = 2

    log = Log(cap=12)
    chat_panel = ChatPanel(chat_r)
    log.add("New: A MONSTER sometimes appears! Stand together on its tile to defeat it.")
    log.add("A harvests WATER only; You harvest FOOD only. Giving raises reputation.")

    # ---------- Bot control nudges from chat ----------
    a_commit_interact = 0     # force A to try to interact (e.g., after agreeing to share)
    a_chase_steps     = 0     # A will move toward B for this many steps

    # ---------- Bot helpers ----------
    def a_choose_action():
        # chat nudges first
        if a_commit_interact > 0:
            return ACT_INTERACT, None
        if a_chase_steps > 0:
            if a_pos < b_pos: return ACT_RIGHT, None
            if a_pos > b_pos: return ACT_LEFT, None
            return ACT_INTERACT, None
        # default policy: go harvest water when possible
        if a_pos < WATER_POS: return ACT_RIGHT, None
        if a_pos > WATER_POS: return ACT_LEFT, None
        if a_pos == WATER_POS and water_timer == 0: return ACT_INTERACT, None
        return ACT_INTERACT, None

    def a_current_want():
        return "food" if a_e < MAX_ENERGY * 0.6 else "none"

    def step_pos(pos, act):
        if act == ACT_LEFT:  return max(0, pos-1)
        if act == ACT_RIGHT: return min(WORLD_SIZE-1, pos+1)
        return pos

    def emo_icon(name):
        try:
            return emotion_states_visual.get(name, "😐")
        except NameError:
            return {"happy":"😊","anxious":"😰","angry":"😡","lonely":"😔","neutral":"😐"}.get(name, "😐")

    # ---- Chat throttle ----
    CHAT_MIN_GAP_MS = 450
    last_chat_tick_A = -999999
    last_chat_tick_B = -999999
    a_spoke_this_step = False
    b_spoke_this_step = False
    def reset_chat_step_flags():
        nonlocal a_spoke_this_step, b_spoke_this_step
        a_spoke_this_step = False
        b_spoke_this_step = False
    def _can_send_chat(is_bot=False):
        now  = pygame.time.get_ticks()
        last = last_chat_tick_A if is_bot else last_chat_tick_B
        spoke = a_spoke_this_step if is_bot else b_spoke_this_step
        return (not spoke) and ((now - last) >= CHAT_MIN_GAP_MS)
    def _send_chat_B(msg):
        nonlocal last_chat_tick_B, b_spoke_this_step, reputation, b_event, b_want
        if not _can_send_chat(False): return
        last_chat_tick_B = pygame.time.get_ticks()
        b_spoke_this_step = True
        chat_panel.add("You", msg)
    def _send_chat_A(msg):
        nonlocal last_chat_tick_A, a_spoke_this_step, reputation, a_event
        if not _can_send_chat(True): return
        last_chat_tick_A = pygame.time.get_ticks()
        a_spoke_this_step = True
        chat_panel.add("A", msg)

    # ---------- Advanced Chat NLU (lightweight) ----------
    INTENTS = [
        "GREET","THANK","APOLOGIZE","INSULT",
        "REQUEST_WATER","GROUP_UP","WARN_MONSTER",
        "OFFER_FOOD","STATUS","SMALLTALK","FAREWELL",
        "UNKNOWN",
    ]
    PATTERNS = {
        "GREET":         r"\b(hi|hello|hey|yo|greetings)\b",
        "THANK":         r"\b(thanks?|thank you|ty|appreciate)\b",
        "APOLOGIZE":     r"\b(sorry|my bad|oops)\b",
        "INSULT":        r"\b(stupid|idiot|dumb|hate you|shut up|insult)\b",
        "REQUEST_WATER": r"(need|want|give|share|some|get|any).*(water|hydration)|\b(thirsty)\b|\bwater\?\b",
        "GROUP_UP":      r"(group|stick|stay).*(up|together)|\bcome (here|to me)\b|\bwith me\b",
        "WARN_MONSTER":  r"\b(monster|danger|threat|watch out|run)\b",
        "OFFER_FOOD":    r"(give|share|have|take).*(food)|\bhere.*food\b|\bfood for you\b",
        "STATUS":        r"(how.*(you|u)|you ok|you okay|status|hp|health)\b",
        "SMALLTALK":     r"(nice|cool|lol|ok|okay|fine|great|good)\b",
        "FAREWELL":      r"\b(bye|later|cya|see you)\b",
    }
    RESPONSES = {
        "GREET":        ["Hi!", "Hello!", "Hey!", "Hey there!"],
        "THANK":        ["You're welcome.", "No problem.", "Anytime.", "Glad to help."],
        "APOLOGIZE":    ["All good.", "No worries.", "It's okay.", "We got this."],
        "INSULT":       ["That won't help.", "Let's stay focused.", "Not cool.", "We still need each other."],
        "REQUEST_WATER":["I can share water if we interact.", "Sure—get close and interact.", "Yes, come adjacent and I'll help."],
        "GROUP_UP":     ["On my way.", "Let's stick together.", "Meet me halfway."],
        "WARN_MONSTER": ["Stay close!", "Move together!", "Watch out—group up!"],
        "OFFER_FOOD":   ["Thanks—interact and I'll take it.", "Okay, ready when adjacent.", "Great, let's trade when close."],
        "STATUS":       [],  # dynamic
        "SMALLTALK":    ["👍", "Ok.", "Yep.", "Got it.", "Cool."],
        "FAREWELL":     ["Stay safe.", "Until next time.", "Bye."],
        "UNKNOWN":      ["Got it.", "Okay.", "Let's survive.", "Understood."],
    }

    class ChatState:
        def __init__(self):
            self.last_step   = {}     # msg -> step
            self.owed_thanks_until = -1
            self.greeted_when_adjacent = False
            self.last_intent_step = {k:-999 for k in INTENTS}
        def mark_intent(self, intent, step):
            self.last_intent_step[intent] = step
        def cooldown_ok(self, intent, step, cd=2):
            return (step - self.last_intent_step.get(intent, -999)) >= cd
        def record(self, msg, step):
            self.last_step[msg] = step
        def schedule_thanks(self, step, extra=2):
            self.owed_thanks_until = max(self.owed_thanks_until, step + extra)

    chat_state = ChatState()

    def parse_intent(text):
        t = text.lower().strip()
        for intent, pat in PATTERNS.items():
            if re.search(pat, t):
                return intent
        return "UNKNOWN"

    def handle_player_chat(user_text):
        """Process B's chat: set wants, tweak rep, push bot reply, and behavior nudges."""
        nonlocal reputation, b_want, a_commit_interact, a_chase_steps
        _send_chat_B(user_text)
        intent = parse_intent(user_text)

        # light rep tweaks
        if intent == "THANK" or intent == "GREET":
            reputation = clamp(reputation + 0.2, -5.0, 5.0)
        elif intent == "INSULT":
            reputation = clamp(reputation - 0.5, -5.0, 5.0)

        # behavior side-effects + reply
        if intent == "REQUEST_WATER":
            b_want = "water"
            a_commit_interact = max(a_commit_interact, 1)   # try to interact next step
            if _can_send_chat(True) and chat_state.cooldown_ok("REQUEST_WATER", steps):
                _send_chat_A(random.choice(RESPONSES["REQUEST_WATER"]))
                chat_state.mark_intent("REQUEST_WATER", steps)

        elif intent == "GROUP_UP":
            a_chase_steps = max(a_chase_steps, 2)
            if _can_send_chat(True) and chat_state.cooldown_ok("GROUP_UP", steps):
                _send_chat_A(random.choice(RESPONSES["GROUP_UP"]))
                chat_state.mark_intent("GROUP_UP", steps)

        elif intent == "WARN_MONSTER":
            if _can_send_chat(True) and chat_state.cooldown_ok("WARN_MONSTER", steps):
                _send_chat_A(random.choice(RESPONSES["WARN_MONSTER"]))
                chat_state.mark_intent("WARN_MONSTER", steps)

        elif intent == "OFFER_FOOD":
            if _can_send_chat(True) and chat_state.cooldown_ok("OFFER_FOOD", steps):
                _send_chat_A(random.choice(RESPONSES["OFFER_FOOD"]))
                chat_state.mark_intent("OFFER_FOOD", steps)

        elif intent == "STATUS":
            if _can_send_chat(True) and chat_state.cooldown_ok("STATUS", steps):
                _send_chat_A(f"My water {int(a_h)}/{MAX_HYDRATION}, energy {int(a_e)}/{MAX_ENERGY}, hp {int(a_hp)}/{MAX_HP}.")
                chat_state.mark_intent("STATUS", steps)

        elif intent in ("GREET","THANK","APOLOGIZE","SMALLTALK","FAREWELL","UNKNOWN"):
            if _can_send_chat(True) and chat_state.cooldown_ok(intent, steps):
                pool = RESPONSES[intent] or RESPONSES["UNKNOWN"]
                _send_chat_A(random.choice(pool))
                chat_state.mark_intent(intent, steps)

    # ---- Simple bot chat policy (event-driven + adjacency greet) ----
    greeted = False
    last_thanks_step = -999
    last_ack_share_step = -999
    last_monster_alert_step = -999

    def bot_chat_policy(*, adjacent, a_gave_water, b_gave_food):
        nonlocal greeted, last_thanks_step, last_ack_share_step, last_monster_alert_step
        if adjacent and not greeted and _can_send_chat(True):
            _send_chat_A("Hello!")
            greeted = True
            return
        if monster_active and not adjacent and (steps - last_monster_alert_step) > 2 and _can_send_chat(True):
            _send_chat_A("Monster nearby — let's group up!")
            last_monster_alert_step = steps
            return
        if a_gave_water and (steps - last_ack_share_step) > 0 and _can_send_chat(True):
            _send_chat_A(random.choice(["Here you go.", "Got you.", "Sharing water now."]))
            last_ack_share_step = steps
            return
        if b_gave_food and (steps - last_thanks_step) > 0 and _can_send_chat(True):
            _send_chat_A(random.choice(["Thanks!", "Appreciate it!", "Much appreciated."]))
            last_thanks_step = steps
            return
        # occasional idle nudge
        if not monster_active and adjacent and _can_send_chat(True) and random.random() < 0.04:
            _send_chat_A(random.choice(["We can trade if needed.", "All good?", "Ready when you are."]))
            return

    # ---- main loop ----
    while (running and a_hp > 0 and b_hp > 0 and a_e > 0 and a_h > 0
           and b_e > 0 and b_h > 0 and steps < max_steps_per_episode):

        # Monster spawn phase
        if not monster_defeated:
            if monster_cd > 0:
                monster_cd -= 1
                monster_active = False
            elif not monster_active:
                if random.random() < MONSTER_SPAWN_PROB:
                    monster_pos = random.randint(0, WORLD_SIZE - 1)
                    monster_active = True
                    log.add(f"👹 Monster appears on tile {monster_pos}!")

        # Draw
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, top_bar)
        title = font_lg.render("Co-op Survival", True, TEXT)
        subtitle = font_sm.render("You are Agent B (Food Getter) • A harvests WATER • Stand together vs MONSTER", True, MUTED)
        screen.blit(title, (20, 10))
        screen.blit(subtitle, (20, 36))
        step_txt = font_md.render(f"Step {steps+1}/{max_steps_per_episode}", True, MUTED)
        screen.blit(step_txt, step_txt.get_rect(right=W-20, centery=top_bar.centery))

        panel(world_r)
        screen.blit(font_md.render("World", True, TEXT), (world_r.x+12, world_r.y-28))
        cell_w = world_r.width // WORLD_SIZE
        for x in range(WORLD_SIZE):
            rx = world_r.x + x*cell_w + 8
            ry = world_r.y + 10
            rw = cell_w - 16
            rh = world_r.height - 20
            color = (80, 80, 88); label = ""
            if x == FOOD_POS:  color, label = ORANGE, "FOOD (B only)"
            if x == WATER_POS: color, label = BLUE,   "WATER (A only)"
            card(pygame.Rect(rx, ry, rw, rh))
            pygame.draw.rect(screen, color, (rx+2, ry+2, rw-4, rh-4), border_radius=10)
            if label:
                screen.blit(font_xs.render(label, True, TEXT), (rx+8, ry+6))

        def draw_agent(px, py, col, tag):
            pygame.draw.circle(screen, col, (px, py), 16)
            t = font_md.render(tag, True, (0,0,0))
            screen.blit(t, t.get_rect(center=(px, py)))
        ax = world_r.x + a_pos*cell_w + cell_w//2
        bx = world_r.x + b_pos*cell_w + cell_w//2
        ay = world_r.y + world_r.height//2 - 18
        by = world_r.y + world_r.height//2 + 18
        draw_agent(ax, ay, (220, 220, 80), "A")
        draw_agent(bx, by, (80, 220, 220), "B")

        if monster_active and not monster_defeated:
            mx = world_r.x + monster_pos*cell_w + cell_w//2
            my = world_r.y + world_r.height//2
            pygame.draw.polygon(screen, MONST, [(mx, my-18), (mx+18, my), (mx, my+18), (mx-18, my)])
            mtxt = font_sm.render("M", True, (0,0,0))
            screen.blit(mtxt, mtxt.get_rect(center=(mx, my)))

        adjacent = abs(a_pos - b_pos) <= 1
        adj_txt = "Adjacent: YES" if adjacent else "Adjacent: NO"
        screen.blit(font_sm.render(adj_txt, True, OK if adjacent else MUTED), (world_r.x+12, world_r.bottom-24))

        panel(wants_r)
        screen.blit(font_md.render("Signals & Status", True, TEXT), (wants_r.x+12, wants_r.y-28))
        inner = pygame.Rect(wants_r.x+10, wants_r.y+10, wants_r.width-20, wants_r.height-20)
        card(inner)
        want = a_current_want()
        want_label = "FOOD" if want == "food" else "Nothing"
        screen.blit(font_sm.render(f"Agent A wants: {want_label}", True, ORANGE if want=="food" else MUTED), (inner.x+12, inner.y+12))
        screen.blit(font_sm.render(f"Your WANT: {'water' if b_want=='water' else 'none'}", True, BLUE if b_want=="water" else MUTED), (inner.x+12, inner.y+40))

        emo_badge = pygame.Rect(inner.right-300, inner.y+8, 134, 60)
        card(emo_badge)
        screen.blit(font_xs.render("Bot emotion", True, MUTED), (emo_badge.x+10, emo_badge.y+6))
        screen.blit(font_md.render(f"{emo_icon(a_emo)}  {a_emo}", True, TEXT), (emo_badge.x+10, emo_badge.y+28))

        mon_badge = pygame.Rect(inner.right-150, inner.y+8, 134, 60)
        card(mon_badge)
        m_status = "Defeated" if monster_defeated else ("Active" if monster_active else "Away")
        m_col = OK if monster_defeated else (BAD if monster_active else MUTED)
        screen.blit(font_xs.render("Monster", True, MUTED), (mon_badge.x+10, mon_badge.y+6))
        screen.blit(font_md.render(m_status, True, m_col), (mon_badge.x+10, mon_badge.y+28))

        panel(status_r)
        screen.blit(font_md.render("Status", True, TEXT), (status_r.x+12, status_r.y-28))
        y0 = status_r.y+8
        screen.blit(font_sm.render("A Health", True, TEXT), (status_r.x+12, y0))
        draw_bar(status_r.x+110, y0+2, 250, 14, a_hp/MAX_HP, col=BAD)
        screen.blit(font_sm.render("A Energy", True, TEXT), (status_r.x+12, y0+28))
        draw_bar(status_r.x+110, y0+30, 250, 14, a_e/MAX_ENERGY)
        screen.blit(font_sm.render("A Hydration", True, TEXT), (status_r.x+12, y0+56))
        draw_bar(status_r.x+110, y0+58, 250, 14, a_h/MAX_HYDRATION)

        screen.blit(font_sm.render("B Health", True, TEXT), (status_r.x+380, y0))
        draw_bar(status_r.x+470, y0+2, 200, 14, b_hp/MAX_HP, col=BAD)
        screen.blit(font_sm.render("B Energy", True, TEXT), (status_r.x+380, y0+28))
        draw_bar(status_r.x+470, y0+30, 200, 14, b_e/MAX_ENERGY)
        screen.blit(font_sm.render("B Hydration", True, TEXT), (status_r.x+380, y0+56))
        draw_bar(status_r.x+470, y0+58, 200, 14, b_h/MAX_HYDRATION)

        screen.blit(font_sm.render("Reputation (giver +1 each gift)", True, TEXT), (status_r.x+12, y0+84))
        draw_bar(status_r.x+300, y0+86, 370, 14, min(1.0, max(0.0, reputation/5.0)))
        screen.blit(font_sm.render(f"{reputation:+.2f}", True, MUTED), (status_r.x+680, y0+82))

        panel(right_r)
        screen.blit(font_md.render("Controls", True, TEXT), (right_r.x+12, right_r.y-28))
        for b in (btn_left, btn_stay, btn_right, btn_share, btn_give, *chat_btns):
            b.draw()
        ykh = right_r.bottom - 22
        for kline in key_help:
            screen.blit(font_xs.render(kline, True, MUTED), (right_r.x+12, ykh))

        chat_panel.draw()
        panel(log_r)
        screen.blit(font_md.render("Log", True, TEXT), (log_r.x+12, log_r.y-28))
        log.draw(pygame.Rect(log_r.x+8, log_r.y+8, log_r.width-16, log_r.height-16))

        pygame.display.flip()

        # Decide A (bot)
        a_act, _ = a_choose_action()

        # Input (you)
        b_act = None
        give_food_now = False
        force_interact_this_frame = False
        sent_from_chatbox = None

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            if btn_left.handle(e):   b_act = ACT_LEFT;  log.add("You moved left.")
            if btn_stay.handle(e):   b_act = ACT_INTERACT; log.add("You chose to stay & interact.")
            if btn_right.handle(e):  b_act = ACT_RIGHT; log.add("You moved right.")
            if btn_share.handle(e):  b_act = ACT_INTERACT; log.add("You chose to Interact.")
            if btn_give.handle(e):   b_act = ACT_INTERACT; give_food_now = True; log.add("Give Food: intent set.")

            # Quick-chat buttons -> go through advanced handler
            if chat_btns[0].handle(e): handle_player_chat("Hello")
            if chat_btns[1].handle(e): handle_player_chat("Thanks!")
            if chat_btns[2].handle(e): handle_player_chat("I need water")
            if chat_btns[3].handle(e): handle_player_chat("Group up?")
            if chat_btns[4].handle(e): handle_player_chat("Sorry")
            if chat_btns[5].handle(e): handle_player_chat("insult")

            sent = chat_panel.handle_event(e)
            if sent:
                sent_from_chatbox = sent

            if e.type == pygame.KEYDOWN and e.key == pygame.K_m:
                if not monster_defeated:
                    monster_pos = random.randint(0, WORLD_SIZE - 1)
                    monster_active = True
                    monster_cd = 0
                    log.add(f"👹 (Debug) Monster forced at tile {monster_pos}")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                running = False

        if sent_from_chatbox:
            handle_player_chat(sent_from_chatbox)
            if any(k in sent_from_chatbox.lower() for k in ("need water", "water please", "i need water", "thirsty")) and adjacent:
                force_interact_this_frame = True
        if force_interact_this_frame and b_act is None:
            b_act = ACT_INTERACT

        if b_act is None:
            clock.tick(60)
            continue

        # reset per-step chat talked flags at start of resolution
        a_event = "none"; b_event = "none"

        # Movement
        na_pos = step_pos(a_pos, a_act)
        nb_pos = step_pos(b_pos, b_act)
        if na_pos == b_pos and nb_pos == a_pos:
            na_pos, nb_pos = a_pos, b_pos
        a_pos, b_pos = na_pos, nb_pos

        # Harvest (asymmetric)
        if a_pos == WATER_POS and a_act == ACT_INTERACT and water_timer == 0:
            a_h = min(MAX_HYDRATION, a_h + INITIAL_REWARD)
            water_timer = WATER_RESPAWN_STEPS
            log.add("A harvested WATER.")
        if b_pos == FOOD_POS and b_act == ACT_INTERACT and food_timer == 0:
            b_e = min(MAX_ENERGY, b_e + INITIAL_REWARD)
            food_timer = FOOD_RESPAWN_STEPS
            log.add("You harvested FOOD.")

        # Trading / Giving
        a_gave_water_this_step = False
        b_gave_food_this_step = False

        if abs(a_pos - b_pos) <= 1 and b_act == ACT_INTERACT:
            if give_food_now:
                give = min(BASE_TRADE_CHUNK, int(b_e))
                if give > 0:
                    b_e -= give
                    a_e += give
                    reputation = clamp(reputation + 1.0, -5.0, 5.0)
                    a_event = b_event = "mutual_aid"
                    b_gave_food_this_step = True
                    log.add(f"★ You GAVE {give} FOOD to A. Reputation +1.")
                    chat_panel.add("System", "You gave A food (+rep).")
                else:
                    log.add("You have no food to give.")
            if b_want == "water":
                base_prob, max_prob = 0.2, 0.95
                success_prob = (1.0 if reputation >= 4.9
                                else base_prob + (max_prob - base_prob) * min(1.0, reputation/5.0))
                max_give_units = int(a_h)
                if max_give_units <= 0:
                    log.add("A wants to help, but has no water to give right now.")
                else:
                    give = min(BASE_TRADE_CHUNK, max_give_units)
                    if random.random() < success_prob:
                        a_h -= give
                        b_h += give
                        reputation = clamp(reputation + 1.0, -5.0, 5.0)
                        a_event = b_event = "mutual_aid"
                        a_gave_water_this_step = True
                        log.add(f"★ A SHARED {give} WATER with you. Reputation +1.")
                        chat_panel.add("System", "A shared water with you (+rep).")
                    else:
                        a_event = "rejected"
                        log.add("A considered it, but declined this time.")

        # Monster resolution
        if monster_active and not monster_defeated:
            if a_pos == monster_pos and b_pos == monster_pos:
                monster_defeated = True
                monster_active = False
                monster_cd = MONSTER_COOLDOWN
                a_e = min(MAX_ENERGY, a_e + MONSTER_REWARD_FOOD)
                b_e = min(MAX_ENERGY, b_e + MONSTER_REWARD_FOOD)
                reputation = clamp(reputation + 2 * MONSTER_REWARD_REP, -5.0, 5.0)
                a_event = b_event = "mutual_aid"
                log.add(f"★ Team-up! Monster defeated. Both gain {MONSTER_REWARD_FOOD} FOOD and +{MONSTER_REWARD_REP} rep each.")
                chat_panel.add("System", "Monster defeated together! Rewards granted.")
            else:
                if a_pos == monster_pos and b_pos != monster_pos:
                    a_hp = max(0.0, a_hp - MONSTER_DAMAGE)
                    log.add("⚠ Monster hit Agent A!")
                if b_pos == monster_pos and a_pos != monster_pos:
                    b_hp = max(0.0, b_hp - MONSTER_DAMAGE)
                    log.add("⚠ Monster hit You!")
                monster_active = False
                monster_cd = MONSTER_COOLDOWN

        # --- Bot chat triggers AFTER outcomes ---
        bot_chat_policy(adjacent=adjacent,
                        a_gave_water=a_gave_water_this_step,
                        b_gave_food=b_gave_food_this_step)

        # Post-chat behavior decay
        if a_commit_interact > 0: a_commit_interact -= 1
        if a_chase_steps   > 0: a_chase_steps   -= 1

        # Respawn & Metabolism
        food_timer  = max(0, food_timer - 1)
        water_timer = max(0, water_timer - 1)
        act_a_used = a_act if a_act in (ACT_LEFT, ACT_RIGHT) else ACT_INTERACT
        act_b_used = b_act if b_act in (ACT_LEFT, ACT_RIGHT) else ACT_INTERACT
        a_e = max(0.0, a_e - (MOVE_COST if act_a_used in (ACT_LEFT, ACT_RIGHT) else IDLE_INTERACT_COST))
        b_e = max(0.0, b_e - (MOVE_COST if act_b_used in (ACT_LEFT, ACT_RIGHT) else IDLE_INTERACT_COST))
        a_h = max(0.0, a_h - THIRST_DRAIN)
        b_h = max(0.0, b_h - THIRST_DRAIN)

        # Emotion update (display next frame)
        try:
            a_emo = update_emotion(a_e, a_h, a_pos, b_pos, a_event)
        except NameError:
            if a_event == "mutual_aid":   a_emo = "happy"
            elif a_event == "rejected":   a_emo = "angry"
            elif a_e < 3 or a_h < 3:      a_emo = "anxious"
            elif abs(a_pos - b_pos) > 2:  a_emo = "lonely"
            else:                         a_emo = "neutral"

        reset_chat_step_flags()
        steps += 1
        clock.tick(60)

    pygame.quit()
    print("Interactive round finished.")

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
        print("Printing full last episode...")
        print_full_episode(simulation_log)
        print_chat_summary(simulation_log)
        print("Launching interactive round (you are agent B)...")
        play_interactive_round()

