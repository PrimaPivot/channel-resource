import pygame
import numpy as np
import random
import math

# --- 常量设置 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
BG_COLOR = (5, 5, 20)

# 状态一 (随机粒子)
STATE1_PARTICLE_COUNT = 8000
STATE1_PARTICLE_COLOR = (200, 200, 255)
STATE1_PARTICLE_RADIUS = 1

# 状态二 (黏菌模拟)
STATE2_AGENT_COUNT = 4000
STATE2_AGENT_COLOR = (220, 255, 220)
STATE2_MOVE_SPEED = 1
STATE2_TURN_SPEED = 0.2
STATE2_SENSOR_ANGLE = math.pi / 6  # 30 degrees
STATE2_SENSOR_DISTANCE = 10
STATE2_DECAY_RATE = 0.98
STATE2_DIFFUSE_RATE = 0.5 # Not used in this simplified version, but can be added

# 状态三 (生命游戏)
STATE3_CELL_SIZE = 4
GRID_WIDTH = SCREEN_WIDTH // STATE3_CELL_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // STATE3_CELL_SIZE
GRID_COLOR = (30, 30, 60)
ALIVE_COLOR = (255, 255, 220)
FPS = 20

# --- 初始化 Pygame ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Type 1,2,3 to switch mode | Type G,R,P in mode 3 to seed | Press Q to quit")
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

# --- 状态管理 ---
current_state = 1

# --- 状态一：粒子 ---
particles = np.random.rand(STATE1_PARTICLE_COUNT, 4)  # x, y, vx, vy
particles[:, 0] *= SCREEN_WIDTH
particles[:, 1] *= SCREEN_HEIGHT
particles[:, 2] = (particles[:, 2] - 0.5) * 2
particles[:, 3] = (particles[:, 3] - 0.5) * 2

# --- 状态二：黏菌 ---
class Agent:
    def __init__(self):
        self.pos = pygame.Vector2(random.uniform(0, SCREEN_WIDTH), random.uniform(0, SCREEN_HEIGHT))
        self.angle = random.uniform(0, 2 * math.pi)

    def update(self, trail_map):
        # 感知
        sensor_center_pos = self.pos + pygame.Vector2(math.cos(self.angle), math.sin(self.angle)) * STATE2_SENSOR_DISTANCE
        sensor_left_pos = self.pos + pygame.Vector2(math.cos(self.angle - STATE2_SENSOR_ANGLE), math.sin(self.angle - STATE2_SENSOR_ANGLE)) * STATE2_SENSOR_DISTANCE
        sensor_right_pos = self.pos + pygame.Vector2(math.cos(self.angle + STATE2_SENSOR_ANGLE), math.sin(self.angle + STATE2_SENSOR_ANGLE)) * STATE2_SENSOR_DISTANCE

        def get_trail_value(pos):
            x, y = int(pos.x), int(pos.y)
            if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT: return trail_map[y, x]
            return 0

        val_c = get_trail_value(sensor_center_pos)
        val_l = get_trail_value(sensor_left_pos)
        val_r = get_trail_value(sensor_right_pos)

        # 转向
        if val_c > val_l and val_c > val_r:
            pass # 保持方向
        elif val_l > val_r:
            self.angle -= STATE2_TURN_SPEED
        elif val_r > val_l:
            self.angle += STATE2_TURN_SPEED

        # 移动
        self.pos += pygame.Vector2(math.cos(self.angle), math.sin(self.angle)) * STATE2_MOVE_SPEED

        # 边界处理
        if self.pos.x < 0 or self.pos.x >= SCREEN_WIDTH or self.pos.y < 0 or self.pos.y >= SCREEN_HEIGHT:
            self.pos.x = random.uniform(0, SCREEN_WIDTH)
            self.pos.y = random.uniform(0, SCREEN_HEIGHT)
            self.angle = random.uniform(0, 2 * math.pi)

    def draw(self, surface): pygame.draw.circle(surface, STATE2_AGENT_COLOR, self.pos, 1)

agents = [Agent() for _ in range(STATE2_AGENT_COUNT)]
trail_map = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.float32)

# --- 状态三：生命游戏 ---
grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int8)

def place_pattern(pattern_name, offset_x, offset_y):
    global grid
    patterns = {
        "gosper_glider_gun": [
            (5, 1), (5, 2), (6, 1), (6, 2), (5, 11), (6, 11), (7, 11), (4, 12), (8, 12), (3, 13), (9, 13),
            (3, 14), (9, 14), (6, 15), (4, 16), (8, 16), (5, 17), (6, 17), (7, 17), (6, 18), (3, 21),
            (4, 21), (5, 21), (3, 22), (4, 22), (5, 22), (2, 23), (6, 23), (1, 25), (2, 25), (6, 25),
            (7, 25), (3, 35), (4, 35), (3, 36), (4, 36)
        ],
        "r_pentomino": [(2, 1), (1, 2), (2, 2), (2, 3), (3, 3)],
        "pulsar": [
            (2,4),(2,5),(2,6), (2,10),(2,11),(2,12),
            (4,2),(5,2),(6,2), (10,2),(11,2),(12,2),
            (4,7),(5,7),(6,7), (10,7),(11,7),(12,7),
            (4,9),(5,9),(6,9), (10,9),(11,9),(12,9),
            (7,4),(7,5),(7,6), (7,10),(7,11),(7,12),
            (9,4),(9,5),(9,6), (9,10),(9,11),(9,12),
            (12,4),(12,5),(12,6), (12,10),(12,11),(12,12),
        ]
    }
    for y, x in patterns.get(pattern_name, []): grid[(y + offset_y) % GRID_HEIGHT, (x + offset_x) % GRID_WIDTH] = 1

def update_life_grid():
    global grid
    new_grid = grid.copy()
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            # 计算邻居数量 (使用numpy的切片和sum会更快，但这里为了清晰)
            neighbors = np.sum(grid[y-1:y+2, x-1:x+2]) - grid[y, x]

            if grid[y, x] == 1 and (neighbors < 2 or neighbors > 3):
                new_grid[y, x] = 0
            elif grid[y, x] == 0 and neighbors == 3:
                new_grid[y, x] = 1
    grid = new_grid

# --- 主循环 ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                current_state = 1
            if event.key == pygame.K_2:
                current_state = 2
                # 重置史莱姆菌模拟
                agents = [Agent() for _ in range(STATE2_AGENT_COUNT)]
                trail_map.fill(0)
            if event.key == pygame.K_3:
                current_state = 3
                grid.fill(0)
                place_pattern("gosper_glider_gun", 5, 5) # 默认生成滑翔者枪
            if current_state == 3:
                if event.key == pygame.K_g:
                    place_pattern("gosper_glider_gun", random.randint(0, GRID_WIDTH-40), random.randint(0, GRID_HEIGHT-40))
                if event.key == pygame.K_r:
                    place_pattern("r_pentomino", GRID_WIDTH // 2, GRID_HEIGHT // 2)
                if event.key == pygame.K_p:
                    place_pattern("pulsar", GRID_WIDTH // 2 - 7, GRID_HEIGHT // 2 - 7)

    screen.fill(BG_COLOR)

    if current_state == 1:
        # 更新粒子
        particles[:, 0] = (particles[:, 0] + particles[:, 2]) % SCREEN_WIDTH
        particles[:, 1] = (particles[:, 1] + particles[:, 3]) % SCREEN_HEIGHT
        # 绘制粒子 (使用BLEND_ADD使亮点更亮)
        for p in particles:
            pygame.draw.circle(screen, STATE1_PARTICLE_COLOR, (p[0], p[1]), STATE1_PARTICLE_RADIUS, 0)
            # screen.set_at((int(p[0]), int(p[1])), STATE1_PARTICLE_COLOR) # 像素点画法

    elif current_state == 2:
        # 绘制信息素轨迹图
        trail_surf = pygame.surfarray.make_surface(np.clip(trail_map.T * 255, 0, 255))
        trail_surf.set_palette([(0,0,0), (20,50,20), (40,100,40), (180,255,180)]) # 绿色调色板
        trail_surf.set_colorkey((0,0,0))
        screen.blit(trail_surf, (0, 0))

        # 更新和绘制智能体
        for agent in agents:
            agent.update(trail_map)
            # 在智能体位置留下信息素
            ix, iy = int(agent.pos.x), int(agent.pos.y)
            if 0 <= ix < SCREEN_WIDTH and 0 <= iy < SCREEN_HEIGHT:
                trail_map[iy, ix] = 1.0 #min(1.0, trail_map[iy, ix] + 0.1)

        # 信息素衰退和扩散（简化版，只做衰退）
        trail_map *= STATE2_DECAY_RATE

    elif current_state == 3:
        update_life_grid()
        # 绘制生命游戏
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if grid[y, x] == 1:
                    pygame.draw.rect(screen, ALIVE_COLOR, (x * STATE3_CELL_SIZE, y * STATE3_CELL_SIZE, STATE3_CELL_SIZE-1, STATE3_CELL_SIZE-1))

    else:
        current_state = 1

    pygame.display.flip()
    clock.tick(FPS if current_state == 3 else 60)

pygame.quit()