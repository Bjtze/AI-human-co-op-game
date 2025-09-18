import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame
import sys

# --- Environment vars ---
WORLD_SIZE = 5
MAX_ENERGY = 12
MAX_HYDRATION = 12

FOOD_POS = 2
WATER_POS = 4
INITIAL_REWARD = 6
FOOD_RESPAWN = 3
WATER_RESPAWN = 3

STEP_REWARD = -0.2
THIRST_DRAIN = 0.4
MOVE_COST = 0.3

# Actions
ACTIONS = [0, 1, 2, 3, 4, 5, 6]  # left, right, interact, ask_water, ask_food, give_food, give_water
ACT_LEFT, ACT_RIGHT, ACT_INTERACT, ACT_ASK_WATER, ACT_ASK_FOOD, ACT_GIVE_FOOD, ACT_GIVE_WATER = 0, 1, 2, 3, 4, 5, 6

PERSONALITY_TRAITS = ["generosity", "emotionality", "sociability", "risk_taking"]
EMOTIONS = ["happy", "anxious", "angry", "lonely", "neutral", "sad"]
emo_to_idx = {e: i for i, e in enumerate(EMOTIONS)}

# For interactive round 
EMO_COLORS = {
    "neutral": (200, 200, 200),
    "happy": (255, 223, 0),
    "anxious": (100, 149, 237),
    "angry": (255, 69, 0),
    "lonely": (160, 32, 240),
    "sad": (70, 130, 180),
}

personality_A = {
    "generosity": random.uniform(0.5, 1),
    "emotionality": random.uniform(-0.3, 1),
    "sociability": random.uniform(-0.3, 1),
    "risk_taking": random.uniform(-1, 1)
}

personality_B = {
    "generosity": random.uniform(0.5, 1),
    "emotionality": random.uniform(-0.3, 1),
    "sociability": random.uniform(-0.3, 1),
    "risk_taking": random.uniform(-1, 1)
}

CHAT_TEMPLATES = {
    ("greet", "neutral"): ["Hello.", "Hi there."],

    ("request", "anxious"): ["Can you help me?", "I really need help…"],
    ("request", "neutral"): ["Mind sharing?", "Could you help me?"],

    ("offer", "happy"): ["I’ll share some with you!", "Here, take mine!"],
    ("offer", "neutral"): ["I can give you some.", "You can have a bit."],

    ("refuse", "sad"): ["Sorry, I don’t have enough.", "I can’t right now…"],
    ("refuse", "angry"): ["Not now.", "Leave me alone."],

    ("thanks", "happy"): ["Thanks a lot!", "Much appreciated!"],
}

# --- Social transfer tuning ---
REQUIRE_ASK_TO_GIVE = True         # set False to allow altruistic giving
GIVE_COOLDOWN_STEPS = 2            # min steps between gives by same agent/resource

GIVE_MIN_TRANSFER = 0.5            # smallest chunk
GIVE_MAX_TRANSFER = 1.0            # largest chunk per step

RESERVE_ENERGY = 4.0               # giver keeps at least this much energy
RESERVE_HYDRATION = 4.0            # giver keeps at least this much hydration

NEED_THRESHOLD_ENERGY = 2.0        # only give if receiver is this far from max
NEED_THRESHOLD_HYDRATION = 2.0



def gen_chat(agent, intent, emotion="neutral"):
    # pick a template based on (intent, emotion), else fall back to neutral
    options = CHAT_TEMPLATES.get((intent, emotion)) or CHAT_TEMPLATES.get((intent, "neutral"))
    if not options:
        options = [f"{intent.title()}."]
    msg = random.choice(options)
    add_chat(agent, intent, emotion, msg)


def emotion_bonus(emotion):
    return {
        "happy":   +0.2,
        "anxious": -0.3,
        "angry":   -0.4,
        "lonely":  -0.1,
        "neutral":  0.0,
        "sad":     -0.3
    }[emotion]

def update_emotion(current_emo, e, h, pos, other_pos, event):
    if event == "helped": 
        if current_emo == "sad" or current_emo == "angry": return "neutral"
        else: return "happy"
    if event == "ignored":
        if random.random() < 0.5: return "sad"
        else: return "angry"
    if e < MAX_ENERGY*0.3 or h < MAX_HYDRATION*0.3: return "anxious"
    if abs(pos - other_pos) > 2: return "lonely"
    return "neutral"

def encode_state(a_pos, a_e, a_h, b_pos, b_e, b_h,
                 rep_a, rep_b, emo_a, emo_b,
                 trust_A_to_B, trust_B_to_A):
    a_pos_vec = np.zeros(WORLD_SIZE); a_pos_vec[a_pos] = 1
    b_pos_vec = np.zeros(WORLD_SIZE); b_pos_vec[b_pos] = 1
    emo_vec_a = np.zeros(len(EMOTIONS)); emo_vec_a[emo_to_idx[emo_a]] = 1
    emo_vec_b = np.zeros(len(EMOTIONS)); emo_vec_b[emo_to_idx[emo_b]] = 1

    return np.concatenate([
        a_pos_vec, [a_e/MAX_ENERGY, a_h/MAX_HYDRATION],
        b_pos_vec, [b_e/MAX_ENERGY, b_h/MAX_HYDRATION],
        [rep_a/10.0, rep_b/10.0],
        emo_vec_a, emo_vec_b,
        [trust_A_to_B/5.0, trust_B_to_A/5.0],
    ])

STATE_DIM = len(encode_state(
    0, MAX_ENERGY, MAX_HYDRATION,
    WORLD_SIZE-1, MAX_ENERGY, MAX_HYDRATION,
    0, 0, "neutral", "neutral",
    0, 0
))



# --- Q net ---
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x): return self.net(x)

qnet_A = QNet(STATE_DIM, len(ACTIONS))
qnet_B = QNet(STATE_DIM, len(ACTIONS))
target_A = QNet(STATE_DIM, len(ACTIONS)); target_A.load_state_dict(qnet_A.state_dict())
target_B = QNet(STATE_DIM, len(ACTIONS)); target_B.load_state_dict(qnet_B.state_dict())

optim_A = optim.Adam(qnet_A.parameters(), lr=1e-4)
optim_B = optim.Adam(qnet_B.parameters(), lr=1e-4)

buffer_A = deque(maxlen=50000)
buffer_B = deque(maxlen=50000)

def train_step(buffer, qnet, target_qnet, optim_q, batch_size=64, gamma=0.99):
    if len(buffer) < batch_size: return None
    batch = random.sample(buffer, batch_size)
    s,a,r,s2,d = zip(*batch)
    s  = torch.tensor(s, dtype=torch.float32)
    a  = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
    r  = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
    s2 = torch.tensor(s2, dtype=torch.float32)
    d  = torch.tensor(d, dtype=torch.float32).unsqueeze(1)
    qvals = qnet(s).gather(1,a)
    with torch.no_grad():
        q_next = target_qnet(s2).max(1, keepdim=True)[0]
        target = r + gamma * (1-d) * q_next
    loss = nn.SmoothL1Loss()(qvals, target)
    optim_q.zero_grad(); loss.backward(); optim_q.step()
    return loss.item()


def choose_action(state, eps, qnet, emotion, trust_for_other, personality):
    if random.random() < eps:
        return random.choice(ACTIONS)

    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        qvals = qnet(s).detach().cpu().numpy()[0]


    # Emotion bias
    if emotion == "angry":
        qvals[ACT_GIVE_FOOD] -= 0.5
        qvals[ACT_GIVE_WATER] -= 0.5
    if emotion == "happy":
        qvals[ACT_GIVE_FOOD] += 0.3
        qvals[ACT_GIVE_WATER] += 0.3


    # Trust bias (long-term memory)
    qvals[ACT_GIVE_WATER] += trust_for_other * 0.2
    qvals[ACT_GIVE_FOOD] += trust_for_other * 0.2

        # Personality: generosity makes GIVE more attractive
    qvals[ACT_GIVE_FOOD] += personality["generosity"] * 0.3
    qvals[ACT_GIVE_WATER] += personality["generosity"] * 0.3

    # Personality: sociability makes ASK more attractive
    qvals[ACT_ASK_WATER] += personality["sociability"] * 0.2
    qvals[ACT_ASK_FOOD] += personality["sociability"] * 0.2

    # Personality: risk-taking influences movement
    qvals[ACT_LEFT]  += 0.1 * personality["risk_taking"]
    qvals[ACT_RIGHT] += 0.1 * personality["risk_taking"]

    # Personality: emotionality amplifies emotion effects
    qvals += (qvals * personality["emotionality"] * 0.05)


    return int(np.argmax(qvals))



# --- Training loop ---
def train_agent(episodes=300):
    epsilon, min_eps, decay = 1.0, 0.05, 0.999
    target_update = 20

    # For plotting
    lossA, lossB, rewA, rewB = [], [], [], []
    rep_history_A, rep_history_B = [], []
    survival_length = []

    emo_counts_A = {e: [] for e in EMOTIONS}
    emo_counts_B = {e: [] for e in EMOTIONS}

    smooth_loss_A, smooth_loss_B = 0.0, 0.0
    beta = 0.95

    for ep in range(episodes):
        # --- Reset episode ---
        a_pos, b_pos = 0, WORLD_SIZE-1
        a_e, b_e = MAX_ENERGY, MAX_ENERGY
        a_h, b_h = MAX_HYDRATION, MAX_HYDRATION
        food_timer, water_timer = 0, 0

        trust_A_to_B = 0.0   # how much A trusts B
        trust_B_to_A = 0.0   # how much B trusts A


        rep_a, rep_b = 0.0, 0.0
        emo_A, emo_B = "neutral", "neutral"

        ep_reward_A, ep_reward_B = 0, 0
        steps = 0

        ep_emo_A = {e:0 for e in EMOTIONS}
        ep_emo_B = {e:0 for e in EMOTIONS}


        # --- Every episode ---
        while a_e>0 and b_e>0 and a_h>0 and b_h>0 and steps<100:
            state = encode_state(a_pos, a_e, a_h, b_pos, b_e, b_h,
                     rep_a, rep_b, emo_A, emo_B,
                     trust_A_to_B, trust_B_to_A)

            act_A = choose_action(state, epsilon, qnet_A, emo_A, trust_A_to_B, personality_A)
            act_B = choose_action(state, epsilon, qnet_B, emo_B, trust_B_to_A, personality_B)

            rew_A, rew_B = STEP_REWARD, STEP_REWARD
            asked_A, asked_B = False, False

            if act_A == ACT_LEFT:   a_pos = max(0, a_pos-1); a_e -= MOVE_COST
            if act_A == ACT_RIGHT:  a_pos = min(WORLD_SIZE-1, a_pos+1); a_e -= MOVE_COST
            if act_B == ACT_LEFT:   b_pos = max(0, b_pos-1); b_e -= MOVE_COST
            if act_B == ACT_RIGHT:  b_pos = min(WORLD_SIZE-1, b_pos+1); b_e -= MOVE_COST

            if act_A == ACT_INTERACT and a_pos == WATER_POS and water_timer == 0:
                a_h = min(MAX_HYDRATION, a_h + INITIAL_REWARD)
                rew_A += 1.0 * (1 + rep_a*0.2)
                water_timer = WATER_RESPAWN
            if act_A == ACT_INTERACT and a_pos == FOOD_POS and food_timer == 0:
                a_e = min(MAX_ENERGY, a_e + INITIAL_REWARD)
                rew_A += 1.0 * (1 + rep_a*0.2)
                food_timer = FOOD_RESPAWN

            if act_B == ACT_INTERACT and b_pos == FOOD_POS and food_timer == 0:
                b_e = min(MAX_ENERGY, b_e + INITIAL_REWARD)
                rew_B += 1.0 * (1 + rep_b*0.2)
                food_timer = FOOD_RESPAWN
            if act_B == ACT_INTERACT and b_pos == WATER_POS and water_timer == 0:
                b_h = min(MAX_HYDRATION, b_h + INITIAL_REWARD)
                rew_B += 1.0 * (1 + rep_b*0.2)
                water_timer = WATER_RESPAWN

            adjacent = abs(a_pos - b_pos) <= 1

            # === Asking ===
            if act_A == ACT_ASK_WATER and adjacent:
                asked_A = True; rew_A += 0.2
                trust_B_to_A += 0.5
            if act_A == ACT_ASK_FOOD and adjacent:
                asked_A = True; rew_A += 0.2
                trust_B_to_A += 0.5

            if act_B == ACT_ASK_WATER and adjacent:
                asked_B = True; rew_B += 0.2
                trust_A_to_B += 0.5
            if act_B == ACT_ASK_FOOD and adjacent:
                asked_B = True; rew_B += 0.2
                trust_A_to_B += 0.5


            # === Giving from A to B ===
            if act_A == ACT_GIVE_WATER and adjacent and a_h > 2:
                transfer = min(2, a_h); a_h -= transfer
                b_h = min(MAX_HYDRATION, b_h + transfer)
                rep_a += 1.0; rep_b += 0.3
                rew_A += 1.0 + rep_a*0.2; rew_B += 0.5
                trust_B_to_A += 0.5
                if emo_B == "anxious":
                    rep_a += 0.5

            elif act_A == ACT_GIVE_FOOD and adjacent and a_e > 2:
                transfer = min(2, a_e); a_e -= transfer
                b_e = min(MAX_ENERGY, b_e + transfer)
                rep_a += 1.0; rep_b += 0.3
                rew_A += 1.0 + rep_a*0.2; rew_B += 0.5
                trust_B_to_A += 0.5
                if emo_B == "anxious":
                    rep_a += 0.5

            elif asked_B and adjacent and act_A not in [ACT_GIVE_FOOD, ACT_GIVE_WATER]:
                rep_a -= 0.5; rew_A -= 0.5
                trust_A_to_B -= 1


            # === Giving from B to A ===
            if act_B == ACT_GIVE_WATER and adjacent and b_h > 2:
                transfer = min(2, b_h); b_h -= transfer
                a_h = min(MAX_HYDRATION, a_h + transfer)
                rep_b += 1.0; rep_a += 0.3
                rew_B += 1.0 + rep_b*0.2; rew_A += 0.5
                trust_A_to_B += 0.5
                if emo_A == "anxious":
                    rep_b += 0.5

            elif act_B == ACT_GIVE_FOOD and adjacent and b_e > 2:
                transfer = min(2, b_e); b_e -= transfer
                a_e = min(MAX_ENERGY, a_e + transfer)
                rep_b += 1.0; rep_a += 0.3
                rew_B += 1.0 + rep_b*0.2; rew_A += 0.5
                trust_A_to_B += 0.5
                if emo_A == "anxious":
                    rep_b += 0.5

            elif asked_A and adjacent and act_B not in [ACT_GIVE_FOOD, ACT_GIVE_WATER]:
                rep_b -= 0.5; rew_B -= 0.5
                trust_B_to_A -= 1



            emo_A = update_emotion(
                emo_A, a_e, a_h, a_pos, b_pos,
                "helped" if act_A in [ACT_GIVE_FOOD, ACT_GIVE_WATER] else
                "ignored" if (act_B in [ACT_ASK_FOOD, ACT_ASK_WATER] and act_A not in [ACT_GIVE_FOOD, ACT_GIVE_WATER]) else "none"
            )
            emo_B = update_emotion(
                emo_B, b_e, b_h, b_pos, a_pos,
                "helped" if act_B in [ACT_GIVE_FOOD, ACT_GIVE_WATER] else
                "ignored" if (act_A in [ACT_ASK_FOOD, ACT_ASK_WATER] and act_B not in [ACT_GIVE_FOOD, ACT_GIVE_WATER]) else "none"
            )


            rew_A += emotion_bonus(emo_A)
            rew_B += emotion_bonus(emo_B)

            ep_emo_A[emo_A] += 1
            ep_emo_B[emo_B] += 1

            a_e = max(0, a_e - 0.05); b_e = max(0, b_e - 0.05)
            a_h = max(0, a_h - THIRST_DRAIN); b_h = max(0, b_h - THIRST_DRAIN)

            rep_a *= 0.995; rep_b *= 0.995

            food_timer = max(0, food_timer-1); water_timer = max(0, water_timer-1)


            next_state = encode_state(a_pos, a_e, a_h, b_pos, b_e, b_h,
                                    rep_a, rep_b, emo_A, emo_B,
                                    trust_A_to_B, trust_B_to_A)

            done = (a_e<=0 or a_h<=0 or b_e<=0 or b_h<=0 or steps>=99)

            buffer_A.append((state, act_A, np.tanh(rew_A), next_state, done))
            buffer_B.append((state, act_B, np.tanh(rew_B), next_state, done))

            if steps % 4 == 0:
                lA = train_step(buffer_A, qnet_A, target_A, optim_A)
                lB = train_step(buffer_B, qnet_B, target_B, optim_B)
                if lA:
                    smooth_loss_A = beta * smooth_loss_A + (1-beta) * lA
                    lossA.append(smooth_loss_A / (1-beta**(len(lossA)+1)))
                if lB:
                    smooth_loss_B = beta * smooth_loss_B + (1-beta) * lB
                    lossB.append(smooth_loss_B / (1-beta**(len(lossB)+1)))

            ep_reward_A += rew_A; ep_reward_B += rew_B
            steps += 1
        

        survival_length.append(steps)



        epsilon = max(min_eps, epsilon*decay)
        if ep % target_update == 0:
            target_A.load_state_dict(qnet_A.state_dict())
            target_B.load_state_dict(qnet_B.state_dict())

        rewA.append(ep_reward_A); rewB.append(ep_reward_B)
        rep_history_A.append(rep_a); rep_history_B.append(rep_b)

        for e in EMOTIONS:
            emo_counts_A[e].append(ep_emo_A[e]/steps)
            emo_counts_B[e].append(ep_emo_B[e]/steps)

        print(f"Ep {ep:03d} | RewA {ep_reward_A:.2f} | RewB {ep_reward_B:.2f} | "
              f"RepA {rep_a:.1f} | RepB {rep_b:.1f} | eps {epsilon:.2f}")

    return lossA, lossB, rewA, rewB, rep_history_A, rep_history_B, emo_counts_A, emo_counts_B, survival_length

def run_training(episodes=300):
    lA, lB, rA, rB, repA, repB, emoA, emoB, survival_length = train_agent(episodes=episodes)

    # --- Plot Loss ---
    plt.figure(figsize=(10,5))
    plt.plot(lA, label="Loss A")
    plt.plot(lB, label="Loss B")
    plt.legend(); plt.title("Loss over time"); plt.xlabel("Training steps"); plt.ylabel("Loss")
    plt.show()

    # --- Plot Rewards ---
    plt.figure(figsize=(10,5))
    plt.plot(rA, label="Reward A")
    plt.plot(rB, label="Reward B")
    plt.legend(); plt.title("Episode Rewards"); plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.show()

    # --- Plot Reputation ---
    plt.figure(figsize=(10,5))
    plt.plot(repA, label="Reputation A")
    plt.plot(repB, label="Reputation B")
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.legend(); plt.title("Reputation per Episode"); plt.xlabel("Episode"); plt.ylabel("Reputation")
    plt.show()

    # --- Plot Emotions  ---
    plt.figure(figsize=(12,6))
    for e in EMOTIONS:
        plt.plot(emoA[e], label=f"A {e}")
    plt.title("Agent A Emotions (fraction per episode)")
    plt.xlabel("Episode"); plt.ylabel("Fraction of steps")
    plt.legend(); plt.show()

    plt.figure(figsize=(12,6))
    for e in EMOTIONS:
        plt.plot(emoB[e], label=f"B {e}")
    plt.title("Agent B Emotions (fraction per episode)")
    plt.xlabel("Episode"); plt.ylabel("Fraction of steps")
    plt.legend(); plt.show()

    # --- Plot Survival Length ---
    plt.plot(survival_length, label="Survival Length (steps)")
    plt.title("Survival Length")
    plt.xlabel("Episode"); plt.ylabel("Steps survived")
    plt.legend(); plt.show()

    return lA, lB, rA, rB, repA, repB, emoA, emoB, survival_length



TILE_SIZE = 80
WIDTH = WORLD_SIZE * TILE_SIZE
HEIGHT = 300
CHAT_HISTORY = []
MAX_CHAT = 8


def add_chat(agent, intent, emotion, msg):
    CHAT_HISTORY.append({
        "agent": agent,
        "intent": intent,
        "emotion": emotion,
        "text": msg
    })
    if len(CHAT_HISTORY) > MAX_CHAT:
        CHAT_HISTORY.pop(0)

def draw_bar(screen, x, y, value, max_value, width, height, color, label=""):
    # Background (empty bar)
    pygame.draw.rect(screen, (60, 60, 60), (x, y, width, height))
    # Filled portion
    fill_width = int((value / max_value) * width)
    pygame.draw.rect(screen, color, (x, y, fill_width, height))
    # Outline
    pygame.draw.rect(screen, (200, 200, 200), (x, y, width, height), 2)

    if label:
        font = pygame.font.SysFont("consolas", 16)
        text = font.render(f"{label}: {value:.1f}/{max_value}", True, (255,255,255))
        screen.blit(text, (x, y - 20))



def draw_world(screen, a_pos, b_pos, a_e, a_h, b_e, b_h,
               emo_A, emo_B, rep_a, rep_b, trust_A, trust_B,
               food_timer, water_timer, act_A=None, act_B=None):
    screen.fill((25,25,25))

    # --- World grid ---
    for x in range(int(WORLD_SIZE)):
        rect = pygame.Rect(int(x*TILE_SIZE), 100, int(TILE_SIZE), int(TILE_SIZE))
        pygame.draw.rect(screen, (60,60,60), rect, 0)
        pygame.draw.rect(screen, (120,120,120), rect, 2)

    # --- Food ---
    if food_timer == 0:
        pygame.draw.rect(screen, (0,200,0),
                         (int(FOOD_POS*TILE_SIZE+20), 120, 40, 40))
    else:
        pygame.draw.rect(screen, (0,80,0),
                         (int(FOOD_POS*TILE_SIZE+20), 120, 40, 40))

    # --- Water ---
    if water_timer == 0:
        pygame.draw.circle(screen, (0,150,255),
                           (int(WATER_POS*TILE_SIZE+TILE_SIZE//2), 140), 20)
    else:
        pygame.draw.circle(screen, (0,60,120),
                           (int(WATER_POS*TILE_SIZE+TILE_SIZE//2), 140), 20)

    # --- Agents ---
    # Player A (the human)
    pygame.draw.circle(screen, EMO_COLORS["neutral"],
                       (int(a_pos*TILE_SIZE+TILE_SIZE//2), 190), 28)
    pygame.draw.circle(screen, (0,0,0),
                       (int(a_pos*TILE_SIZE+TILE_SIZE//2), 190), 28, 2)

    # Player B (the AI)
    pygame.draw.circle(screen, EMO_COLORS[emo_B],
                       (int(b_pos*TILE_SIZE+TILE_SIZE//2), 230), 28)
    pygame.draw.circle(screen, (0,0,0),
                       (int(b_pos*TILE_SIZE+TILE_SIZE//2), 230), 28, 2)

    # --- HUD text (top left) ---
    # --- HUD bars for Agent A ---
    draw_bar(screen, 10, 10, a_e, MAX_ENERGY, 200, 15, (0,200,0), "A Energy")
    draw_bar(screen, 10, 40, a_h, MAX_HYDRATION, 200, 15, (0,150,255), "A Hydration")

    # --- HUD bars for Agent B ---
    draw_bar(screen, 10, 80, b_e, MAX_ENERGY, 200, 15, (0,200,0), "B Energy")
    draw_bar(screen, 10, 110, b_h, MAX_HYDRATION, 200, 15, (0,150,255), "B Hydration")

    # --- Chat Box (right side) ---
    chat_rect = pygame.Rect(int(WIDTH+20), 20, 300, int(HEIGHT-40))
    pygame.draw.rect(screen, (40,40,40), chat_rect)
    pygame.draw.rect(screen, (200,200,200), chat_rect, 2)

    y_offset = 30
    font_chat = pygame.font.SysFont("consolas", 16)
    for line in CHAT_HISTORY[-MAX_CHAT:]:
        speaker, intent, emotion, msg = line["agent"], line["intent"], line["emotion"], line["text"]
        color = (255,220,180) if speaker=="A" else (180,220,255)

        emo_color = EMO_COLORS.get(emotion, (200,200,200))
        pygame.draw.rect(screen, emo_color, (WIDTH+22, y_offset+4, 6, 12))
        chat_text = font_chat.render(f"{speaker}: {msg}", True, color)
        screen.blit(chat_text, (WIDTH+32, y_offset))
        y_offset += 20


    pygame.display.flip()



def interactive_pygame(max_steps=50, epsilon=0.1):
    pygame.init()
    screen_width = WIDTH + 500   # world + chat + buttons
    screen = pygame.display.set_mode((screen_width, HEIGHT))
    clock = pygame.time.Clock()

    # --- Buttons (right column) ---
    BUTTONS = [
        {"label": "Ask Food",  "action": ACT_ASK_FOOD},
        {"label": "Ask Water", "action": ACT_ASK_WATER},
        {"label": "Give Food", "action": ACT_GIVE_FOOD},
        {"label": "Give Water","action": ACT_GIVE_WATER},
    ]
    button_w, button_h = 160, 40
    button_start_x, button_start_y = WIDTH + 320, 60  # right of chat

    def draw_buttons():
        font = pygame.font.SysFont("consolas", 18)
        rects = []
        # Sidebar background for buttons
        button_panel = pygame.Rect(WIDTH+300, 0, 200, HEIGHT)
        pygame.draw.rect(screen, (40, 40, 40), button_panel)

        for i, btn in enumerate(BUTTONS):
            rect = pygame.Rect(
                button_start_x,
                button_start_y + i * (button_h + 15),
                button_w,
                button_h
            )
            pygame.draw.rect(screen, (90, 90, 90), rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 2)
            text = font.render(btn["label"], True, (255, 255, 255))
            screen.blit(text, (rect.x + 10, rect.y + 10))
            rects.append((rect, btn["action"]))
        return rects
    

        # --- Transfer helper: returns (giver_new, receiver_new, moved) ---
    def do_transfer(giver_amt, receiver_amt, receiver_max,
                    min_keep_for_giver=0.0,   # reserve
                    min_chunk=0.5,            # don't "give" dust
                    max_chunk=2.0):           # cap per step
        room = max(0.0, receiver_max - receiver_amt)          # receiver need
        can_spare = max(0.0, giver_amt - min_keep_for_giver)  # giver margin
        raw = min(room, can_spare, max_chunk)
        moved = raw if raw >= min_chunk else 0.0
        if moved > 0.0:
            giver_amt -= moved
            receiver_amt = min(receiver_max, receiver_amt + moved)
        return giver_amt, receiver_amt, moved


    # --- Reset world ---
    a_pos, b_pos = 0, WORLD_SIZE-1
    a_e, b_e = MAX_ENERGY, MAX_ENERGY
    a_h, b_h = MAX_HYDRATION, MAX_HYDRATION
    rep_a, rep_b, trust_A, trust_B = 0.0, 0.0, 0.0, 0.0
    emo_A, emo_B = "neutral", "neutral"
    food_timer, water_timer = 0, 0

    last_give = {
        "A_FOOD": -999, "A_WATER": -999,
        "B_FOOD": -999, "B_WATER": -999
    }

    for step in range(max_steps):
        if step == max_steps - 1:
            print("Game Over: Maximum steps reached!")

        # === Player input (Agent A) ===
        act_A = None
        waiting = True
        while waiting:
            draw_world(screen, a_pos, b_pos,
                       a_e, a_h, b_e, b_h,
                       emo_A, emo_B,
                       rep_a, rep_b,
                       trust_A, trust_B,
                       food_timer, water_timer)

            rects = draw_buttons()
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:  act_A = ACT_LEFT; waiting = False
                    if event.key == pygame.K_RIGHT: act_A = ACT_RIGHT; waiting = False
                    if event.key == pygame.K_UP:    act_A = ACT_INTERACT; waiting = False
                    if event.key == pygame.K_DOWN:  act_A = None; waiting = False  # could add "rest" here
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    for rect, action in rects:
                        if rect.collidepoint(mx, my):
                            act_A = action
                            waiting = False

        # === Agent B decides ===
        state = encode_state(a_pos, a_e, a_h, b_pos, b_e, b_h,
                             rep_a, rep_b, emo_A, emo_B,
                             trust_A, trust_B)
        act_B = choose_action(state, epsilon, qnet_B, emo_B, trust_B, personality_B)

        # === Game logic ===
        rew_A, rew_B = STEP_REWARD, STEP_REWARD
        asked_A, asked_B = False, False

        # --- Movement ---
        if act_A == ACT_LEFT:   a_pos = max(0, a_pos-1); a_e -= MOVE_COST
        if act_A == ACT_RIGHT:  a_pos = min(WORLD_SIZE-1, a_pos+1); a_e -= MOVE_COST
        if act_B == ACT_LEFT:   b_pos = max(0, b_pos-1); b_e -= MOVE_COST
        if act_B == ACT_RIGHT:  b_pos = min(WORLD_SIZE-1, b_pos+1); b_e -= MOVE_COST

        # --- Interact (↑ key) ---
        if act_A == ACT_INTERACT and a_pos == FOOD_POS and food_timer == 0:
            a_e = min(MAX_ENERGY, a_e + INITIAL_REWARD)
            rew_A += 1.0 * (1 + rep_a*0.2)
            food_timer = FOOD_RESPAWN
        if act_A == ACT_INTERACT and a_pos == WATER_POS and water_timer == 0:
            a_h = min(MAX_HYDRATION, a_h + INITIAL_REWARD)
            rew_A += 1.0 * (1 + rep_a*0.2)
            water_timer = WATER_RESPAWN
        if act_B == ACT_INTERACT and b_pos == FOOD_POS and food_timer == 0:
            b_e = min(MAX_ENERGY, b_e + INITIAL_REWARD)
            rew_B += 1.0 * (1 + rep_b*0.2)
            food_timer = FOOD_RESPAWN
        if act_B == ACT_INTERACT and b_pos == WATER_POS and water_timer == 0:
            b_h = min(MAX_HYDRATION, b_h + INITIAL_REWARD)
            rew_B += 1.0 * (1 + rep_b*0.2)
            water_timer = WATER_RESPAWN

        # --- Ask (buttons) ---
        adjacent = abs(a_pos - b_pos) <= 1

        did_A_give = False
        did_B_give = False
        a_asked_now = (act_A in [ACT_ASK_FOOD, ACT_ASK_WATER]) and adjacent
        b_asked_now = (act_B in [ACT_ASK_FOOD, ACT_ASK_WATER]) and adjacent

        if a_asked_now:
            asked_A = True; rew_A += 0.2; trust_B += 0.5
            gen_chat("A", "request", emo_A)
        if b_asked_now:
            asked_B = True; rew_B += 0.2; trust_A += 0.5
            gen_chat("B", "request", emo_B)

        # helpers
        def clamp_transfer(need, giver_amount, reserve, tmin, tmax):
            # only transfer what's needed, never below reserve, within [tmin, tmax]
            max_allowed = max(0.0, giver_amount - reserve)
            amt = min(need, max_allowed, tmax)
            return max(0.0, amt if amt >= tmin else 0.0)

        step_idx = step  # for cooldown

        # --- A gives FOOD ---
        if act_A == ACT_GIVE_FOOD and adjacent:
            # gating: require ask or clear need unless disabled
            receiver_need = max(0.0, MAX_ENERGY - b_e)
            can_give = (not REQUIRE_ASK_TO_GIVE and receiver_need >= NEED_THRESHOLD_ENERGY) or asked_B
            cd_ok = (step_idx - last_give["A_FOOD"] >= GIVE_COOLDOWN_STEPS)
            if can_give and cd_ok:
                transfer = clamp_transfer(receiver_need, a_e, RESERVE_ENERGY,
                                          GIVE_MIN_TRANSFER, GIVE_MAX_TRANSFER)
                if transfer > 0:
                    a_e -= transfer; b_e = min(MAX_ENERGY, b_e + transfer)
                    rep_a += 1.0; rep_b += 0.3
                    rew_A += 1.0 + rep_a*0.2; rew_B += 0.5; trust_A += 0.5
                    gen_chat("A", "offer", emo_A)
                    add_chat("SYS", "info", "neutral",
                             f"A→B FOOD {transfer:.1f} (A_E {a_e:.1f} → B_E {b_e:.1f})")
                    did_A_give = True; last_give["A_FOOD"] = step_idx
                else:
                    gen_chat("A", "refuse", emo_A)
                    add_chat("SYS", "warn", "neutral", "A kept FOOD reserve; no transfer.")
            else:
                gen_chat("A", "refuse", emo_A)
                add_chat("SYS", "warn", "neutral", "A didn’t meet ask/need/cooldown for FOOD.")

        # --- A gives WATER ---
        if act_A == ACT_GIVE_WATER and adjacent:
            receiver_need = max(0.0, MAX_HYDRATION - b_h)
            can_give = (not REQUIRE_ASK_TO_GIVE and receiver_need >= NEED_THRESHOLD_HYDRATION) or asked_B
            cd_ok = (step_idx - last_give["A_WATER"] >= GIVE_COOLDOWN_STEPS)
            if can_give and cd_ok:
                transfer = clamp_transfer(receiver_need, a_h, RESERVE_HYDRATION,
                                          GIVE_MIN_TRANSFER, GIVE_MAX_TRANSFER)
                if transfer > 0:
                    a_h -= transfer; b_h = min(MAX_HYDRATION, b_h + transfer)
                    rep_a += 1.0; rep_b += 0.3
                    rew_A += 1.0 + rep_a*0.2; rew_B += 0.5; trust_A += 0.5
                    gen_chat("A", "offer", emo_A)
                    add_chat("SYS", "info", "neutral",
                             f"A→B WATER {transfer:.1f} (A_H {a_h:.1f} → B_H {b_h:.1f})")
                    did_A_give = True; last_give["A_WATER"] = step_idx
                else:
                    gen_chat("A", "refuse", emo_A)
                    add_chat("SYS", "warn", "neutral", "A kept WATER reserve; no transfer.")
            else:
                gen_chat("A", "refuse", emo_A)
                add_chat("SYS", "warn", "neutral", "A didn’t meet ask/need/cooldown for WATER.")

        # snub penalty if B asked and A didn’t give
        if asked_B and adjacent and not did_A_give and act_A not in [ACT_GIVE_FOOD, ACT_GIVE_WATER]:
            rep_a -= 0.5; rew_A -= 0.5; trust_A -= 1
            gen_chat("A", "refuse", emo_A)
            add_chat("SYS", "info", "neutral", "A ignored B’s request this step.")

        # --- B gives FOOD ---
        if act_B == ACT_GIVE_FOOD and adjacent:
            receiver_need = max(0.0, MAX_ENERGY - a_e)
            can_give = (not REQUIRE_ASK_TO_GIVE and receiver_need >= NEED_THRESHOLD_ENERGY) or asked_A
            cd_ok = (step_idx - last_give["B_FOOD"] >= GIVE_COOLDOWN_STEPS)
            if can_give and cd_ok:
                transfer = clamp_transfer(receiver_need, b_e, RESERVE_ENERGY,
                                          GIVE_MIN_TRANSFER, GIVE_MAX_TRANSFER)
                if transfer > 0:
                    b_e -= transfer; a_e = min(MAX_ENERGY, a_e + transfer)
                    rep_b += 1.0; rep_a += 0.3
                    rew_B += 1.0 + rep_b*0.2; rew_A += 0.5; trust_B += 0.5
                    gen_chat("B", "offer", emo_B)
                    add_chat("SYS", "info", "neutral",
                             f"B→A FOOD {transfer:.1f} (B_E {b_e:.1f} → A_E {a_e:.1f})")
                    did_B_give = True; last_give["B_FOOD"] = step_idx
                else:
                    gen_chat("B", "refuse", emo_B)
                    add_chat("SYS", "warn", "neutral", "B kept FOOD reserve; no transfer.")
            else:
                gen_chat("B", "refuse", emo_B)
                add_chat("SYS", "warn", "neutral", "B didn’t meet ask/need/cooldown for FOOD.")

        # --- B gives WATER ---
        if act_B == ACT_GIVE_WATER and adjacent:
            receiver_need = max(0.0, MAX_HYDRATION - a_h)
            can_give = (not REQUIRE_ASK_TO_GIVE and receiver_need >= NEED_THRESHOLD_HYDRATION) or asked_A
            cd_ok = (step_idx - last_give["B_WATER"] >= GIVE_COOLDOWN_STEPS)
            if can_give and cd_ok:
                transfer = clamp_transfer(receiver_need, b_h, RESERVE_HYDRATION,
                                          GIVE_MIN_TRANSFER, GIVE_MAX_TRANSFER)
                if transfer > 0:
                    b_h -= transfer; a_h = min(MAX_HYDRATION, a_h + transfer)
                    rep_b += 1.0; rep_a += 0.3
                    rew_B += 1.0 + rep_b*0.2; rew_A += 0.5; trust_B += 0.5
                    gen_chat("B", "offer", emo_B)
                    add_chat("SYS", "info", "neutral",
                             f"B→A WATER {transfer:.1f} (B_H {b_h:.1f} → A_H {a_h:.1f})")
                    did_B_give = True; last_give["B_WATER"] = step_idx
                else:
                    gen_chat("B", "refuse", emo_B)
                    add_chat("SYS", "warn", "neutral", "B kept WATER reserve; no transfer.")
            else:
                gen_chat("B", "refuse", emo_B)
                add_chat("SYS", "warn", "neutral", "B didn’t meet ask/need/cooldown for WATER.")

        # snub penalty if A asked and B didn’t give
        if asked_A and adjacent and not did_B_give and act_B not in [ACT_GIVE_FOOD, ACT_GIVE_WATER]:
            rep_b -= 0.5; rew_B -= 0.5; trust_B -= 1
            gen_chat("B", "refuse", emo_B)
            add_chat("SYS", "info", "neutral", "B ignored A’s request this step.")


        # --- Ask (button actions) ---
        adjacent = abs(a_pos - b_pos) <= 1
        did_A_give = False
        did_B_give = False

        if act_A in [ACT_ASK_FOOD, ACT_ASK_WATER] and adjacent:
            asked_A = True; rew_A += 0.2; trust_B += 0.5
            gen_chat("A", "request", emo_A)
        if act_B in [ACT_ASK_FOOD, ACT_ASK_WATER] and adjacent:
            asked_B = True; rew_B += 0.2; trust_A += 0.5
            gen_chat("B", "request", emo_B)

        # Early visibility if not adjacent
        if (act_A in [ACT_GIVE_FOOD, ACT_GIVE_WATER] or
            act_B in [ACT_GIVE_FOOD, ACT_GIVE_WATER]) and not adjacent:
            add_chat("SYS", "warn", "neutral", "Give failed: not adjacent.")

        # --- A gives FOOD ---
        if act_A == ACT_GIVE_FOOD and adjacent:
            old_a, old_b = a_e, b_e
            a_e, b_e, moved = do_transfer(
                a_e, b_e, MAX_ENERGY,
                min_keep_for_giver=2.0,  # RESERVE_ENERGY
                min_chunk=0.5,
                max_chunk=2.0
            )
            if moved > 0:
                rep_a += 1.0; rep_b += 0.3
                rew_A += 1.0 + rep_a*0.2; rew_B += 0.5; trust_A += 0.5
                gen_chat("A", "offer", emo_A)
                add_chat("SYS", "info", "neutral",
                         f"A→B FOOD {moved:.1f} (A_E {old_a:.1f}→{a_e:.1f}, B_E {old_b:.1f}→{b_e:.1f})")
                did_A_give = True
            else:
                gen_chat("A", "refuse", emo_A)
                add_chat("SYS", "warn", "neutral",
                         f"A kept reserve or B full (A_E={old_a:.1f}, B_E={old_b:.1f}).")

        # --- A gives WATER ---
        if act_A == ACT_GIVE_WATER and adjacent:
            old_a, old_b = a_h, b_h
            a_h, b_h, moved = do_transfer(
                a_h, b_h, MAX_HYDRATION,
                min_keep_for_giver=2.0,  # RESERVE_HYDRATION
                min_chunk=0.5,
                max_chunk=2.0
            )
            if moved > 0:
                rep_a += 1.0; rep_b += 0.3
                rew_A += 1.0 + rep_a*0.2; rew_B += 0.5; trust_A += 0.5
                gen_chat("A", "offer", emo_A)
                add_chat("SYS", "info", "neutral",
                         f"A→B WATER {moved:.1f} (A_H {old_a:.1f}→{a_h:.1f}, B_H {old_b:.1f}→{b_h:.1f})")
                did_A_give = True
            else:
                gen_chat("A", "refuse", emo_A)
                add_chat("SYS", "warn", "neutral",
                         f"A kept reserve or B full (A_H={old_a:.1f}, B_H={old_b:.1f}).")

        # Penalty if B asked but A didn’t give
        if asked_B and adjacent and not did_A_give and act_A not in [ACT_GIVE_FOOD, ACT_GIVE_WATER]:
            rep_a -= 0.5; rew_A -= 0.5; trust_A -= 1
            gen_chat("A", "refuse", emo_A)
            add_chat("SYS", "info", "neutral", "A ignored B’s request this step.")

        # --- B gives FOOD ---
        if act_B == ACT_GIVE_FOOD and adjacent:
            old_b, old_a = b_e, a_e
            b_e, a_e, moved = do_transfer(
                b_e, a_e, MAX_ENERGY,
                min_keep_for_giver=2.0,
                min_chunk=0.5,
                max_chunk=2.0
            )
            if moved > 0:
                rep_b += 1.0; rep_a += 0.3
                rew_B += 1.0 + rep_b*0.2; rew_A += 0.5; trust_B += 0.5
                gen_chat("B", "offer", emo_B)
                add_chat("SYS", "info", "neutral",
                         f"B→A FOOD {moved:.1f} (B_E {old_b:.1f}→{b_e:.1f}, A_E {old_a:.1f}→{a_e:.1f})")
                did_B_give = True
            else:
                gen_chat("B", "refuse", emo_B)
                add_chat("SYS", "warn", "neutral",
                         f"B kept reserve or A full (B_E={old_b:.1f}, A_E={old_a:.1f}).")

        # --- B gives WATER ---
        if act_B == ACT_GIVE_WATER and adjacent:
            old_b, old_a = b_h, a_h
            b_h, a_h, moved = do_transfer(
                b_h, a_h, MAX_HYDRATION,
                min_keep_for_giver=2.0,
                min_chunk=0.5,
                max_chunk=2.0
            )
            if moved > 0:
                rep_b += 1.0; rep_a += 0.3
                rew_B += 1.0 + rep_b*0.2; rew_A += 0.5; trust_B += 0.5
                gen_chat("B", "offer", emo_B)
                add_chat("SYS", "info", "neutral",
                         f"B→A WATER {moved:.1f} (B_H {old_b:.1f}→{b_h:.1f}, A_H {old_a:.1f}→{a_h:.1f})")
                did_B_give = True
            else:
                gen_chat("B", "refuse", emo_B)
                add_chat("SYS", "warn", "neutral",
                         f"B kept reserve or A full (B_H={old_b:.1f}, A_H={old_a:.1f}).")

        # Penalty if A asked but B didn’t give
        if asked_A and adjacent and not did_B_give and act_B not in [ACT_GIVE_FOOD, ACT_GIVE_WATER]:
            rep_b -= 0.5; rew_B -= 0.5; trust_B -= 1
            gen_chat("B", "refuse", emo_B)
            add_chat("SYS", "info", "neutral", "B ignored A’s request this step.")


        # --- Emotions ---
        emo_A = update_emotion(
            emo_A, a_e, a_h, a_pos, b_pos,
            "helped" if act_A in [ACT_GIVE_FOOD, ACT_GIVE_WATER] else
            "ignored" if (act_B in [ACT_ASK_FOOD, ACT_ASK_WATER] and act_A not in [ACT_GIVE_FOOD, ACT_GIVE_WATER]) else "none"
        )
        emo_B = update_emotion(
            emo_B, b_e, b_h, b_pos, a_pos,
            "helped" if act_B in [ACT_GIVE_FOOD, ACT_GIVE_WATER] else
            "ignored" if (act_A in [ACT_ASK_FOOD, ACT_ASK_WATER] and act_B not in [ACT_GIVE_FOOD, ACT_GIVE_WATER]) else "none"
        )
        rew_A += emotion_bonus(emo_A)
        rew_B += emotion_bonus(emo_B)

        # --- Metabolism ---
        a_e = max(0, a_e - 0.05)
        b_e = max(0, b_e - 0.05)
        a_h = max(0, a_h - THIRST_DRAIN)
        b_h = max(0, b_h - THIRST_DRAIN)

        rep_a *= 0.995; rep_b *= 0.995
        food_timer = max(0, food_timer-1)
        water_timer = max(0, water_timer-1)

        # Redraw after actions
        draw_world(screen, a_pos, b_pos,
                   a_e, a_h, b_e, b_h,
                   emo_A, emo_B,
                   rep_a, rep_b,
                   trust_A, trust_B,
                   food_timer, water_timer,
                   act_A=act_A, act_B=act_B)
        draw_buttons()
        pygame.display.flip()

        if a_e <= 0 or a_h <= 0 or b_e <= 0 or b_h <= 0:
            add_chat("SYS", "Game Over: one agent collapsed.")
            break

        clock.tick(30)



# === Entry Point ===
if __name__ == "__main__":
    run_training(episodes=700)
    interactive_pygame(max_steps=30, epsilon=0.1)
