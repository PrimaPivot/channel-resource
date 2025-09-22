#  !/usr/bin/env python
#  -*- coding:utf-8 -*-

#  ==============================================
#  ·
#  · Author: PrimaPivot
#  ·
#  · Filename: 002-slime.py
#  ·
#  · COPYRIGHT 2025
#  ·
#  · 模拟黏菌生成路径的过程
#  ·
#  ==============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

# --- 1. 配置参数 ---
class Config:
    WIDTH, HEIGHT = 600, 450
    N_AGENTS = 5000
    ITERATIONS = 300
    MOVE_SPEED = 3.
    TURN_SPEED = 0.3
    SENSOR_ANGLE = 0.5
    SENSOR_DISTANCE = 10.0
    DEPOSIT_AMOUNT = 0.1
    DECAY_FACTOR = 0.98
    BLUR_SIGMA = 0.5
    CHEMOTAXIS_WEIGHT = 0.2
    FOOD_PHEROMONE = 1.0
    REPOP_INTERVAL = 50
    DEATH_DISTANCE_THRESHOLD = WIDTH / 3.0
    RANDOM_SEED = 42
    DOCKING_RADIUS = 5.0
    SATIATED_DISTANCE = WIDTH / 3.5

# --- 2. 定义智能体 (Agent) ---
class Agent:
    def __init__(self, x, y, heading):
        self.pos = np.array([x, y], dtype=float)
        self.heading = heading
        self.state = 'satiated'
        self.last_eaten_at = self.pos.copy()
        self.distance_traveled_since_eat = 0.0

    def move(self):
        self.pos += Config.MOVE_SPEED * np.array([np.cos(self.heading), np.sin(self.heading)])

        bounce = False
        # 检查左右边界
        if self.pos[0] <= 0:
            self.pos[0] = 0
            self.heading = np.pi - self.heading
            bounce = True
        elif self.pos[0] >= Config.WIDTH - 1:
            self.pos[0] = Config.WIDTH - 1
            self.heading = np.pi - self.heading
            bounce = True

        # 检查上下边界
        if self.pos[1] <= 0:
            self.pos[1] = 0
            self.heading = -self.heading # 等效于 2*pi - heading
            bounce = True
        elif self.pos[1] >= Config.HEIGHT - 1:
            self.pos[1] = Config.HEIGHT - 1
            self.heading = -self.heading
            bounce = True

        if bounce: self.heading += np.random.uniform(-0.1, 0.1) # 增加轻微扰动防止卡住

        if self.state == 'satiated':
            self.distance_traveled_since_eat += Config.MOVE_SPEED

    def sense_and_steer(self, pheromone_map, obstacle_map, stations):
        if self.state == 'hungry':
            for station in stations:
                if np.linalg.norm(self.pos - station) < Config.DOCKING_RADIUS:
                    self.pos = station.copy()
                    self.state = 'satiated'
                    self.last_eaten_at = station.copy()
                    self.distance_traveled_since_eat = 0.0
                    self.heading = np.random.uniform(0, 2 * np.pi)
                    return

        if self.state == 'satiated' and self.distance_traveled_since_eat > Config.SATIATED_DISTANCE:
            self.state = 'hungry'

        angle_fwd = self.heading
        angle_left = self.heading - Config.SENSOR_ANGLE
        angle_right = self.heading + Config.SENSOR_ANGLE
        sensor_fwd = self.pos + Config.SENSOR_DISTANCE * np.array([np.cos(angle_fwd), np.sin(angle_fwd)])
        sensor_left = self.pos + Config.SENSOR_DISTANCE * np.array([np.cos(angle_left), np.sin(angle_left)])
        sensor_right = self.pos + Config.SENSOR_DISTANCE * np.array([np.cos(angle_right), np.sin(angle_right)])

        for sensor_pos in [sensor_fwd, sensor_left, sensor_right]:
            ix, iy = int(sensor_pos[0]), int(sensor_pos[1])
            if not (0 <= ix < Config.WIDTH and 0 <= iy < Config.HEIGHT and obstacle_map[iy, ix] == 0):
                self.heading = np.random.uniform(0, 2 * np.pi); return

        c_fwd = pheromone_map[int(sensor_fwd[1]), int(sensor_fwd[0])]
        c_left = pheromone_map[int(sensor_left[1]), int(sensor_left[0])]
        c_right = pheromone_map[int(sensor_right[1]), int(sensor_right[0])]

        pheromone_steer_angle = self.heading
        if c_fwd > c_left and c_fwd > c_right: pass
        elif c_left > c_right: pheromone_steer_angle -= Config.TURN_SPEED
        elif c_right > c_left: pheromone_steer_angle += Config.TURN_SPEED
        else: pheromone_steer_angle += np.random.uniform(-Config.TURN_SPEED, Config.TURN_SPEED)

        chemotaxis_target = None
        if self.state == 'hungry':
            # --- 寻找目标时，排除上一个吃过的站点 ---
            min_dist = float('inf')
            target_station = None
            for station in stations:
                # 关键判断：不能是上一个吃过的站点
                if not np.array_equal(station, self.last_eaten_at):
                    dist = np.linalg.norm(self.pos - station)
                    if dist < min_dist:
                        min_dist = dist
                        target_station = station
            # 如果因为某种原因没找到（比如只有两个站点来回），就随机选一个
            if target_station is None:
                 options = [s for s in stations if not np.array_equal(s, self.last_eaten_at)]
                 target_station = options[np.random.randint(len(options))]
            chemotaxis_target = target_station
        else: # 'satiated'
            repulsion_vec = self.pos - self.last_eaten_at
            if np.linalg.norm(repulsion_vec) < 1e-5:
                chemotaxis_target = self.pos + np.array([np.cos(self.heading), np.sin(self.heading)])
            else:
                chemotaxis_target = self.pos + repulsion_vec

        chemotaxis_steer_angle = np.arctan2(chemotaxis_target[1] - self.pos[1], chemotaxis_target[0] - self.pos[0])

        pheromone_vec = np.array([np.cos(pheromone_steer_angle), np.sin(pheromone_steer_angle)])
        chemotaxis_vec = np.array([np.cos(chemotaxis_steer_angle), np.sin(chemotaxis_steer_angle)])
        final_vec = (1 - Config.CHEMOTAXIS_WEIGHT) * pheromone_vec + Config.CHEMOTAXIS_WEIGHT * chemotaxis_vec
        self.heading = np.arctan2(final_vec[1], final_vec[0])

# --- 3. 主仿真函数 ---
def run_simulation():
    np.random.seed(Config.RANDOM_SEED)
    pheromone_map = np.zeros((Config.HEIGHT, Config.WIDTH))
    obstacle_map = np.zeros((Config.HEIGHT, Config.WIDTH))
    # obstacle_map[int(Config.HEIGHT*0.8):, int(Config.WIDTH*0.7):] = 1
    obstacle_map[int(Config.HEIGHT*0.6):, int(Config.WIDTH*0.7):] = 1
    cx, cy, r = int(Config.WIDTH*0.6), int(Config.HEIGHT*0.5), 30
    y, x = np.ogrid[-cy:Config.HEIGHT-cy, -cx:Config.WIDTH-cx]
    mask = x*x + y*y <= r*r
    obstacle_map[mask] = 1
    stations = []
    center_x, center_y = Config.WIDTH / 2, Config.HEIGHT / 2
    radius_x, radius_y = Config.WIDTH / 3, Config.HEIGHT / 3
    num_stations = 10

    for i in range(num_stations):
        stations.append(np.array([
            center_x + radius_x * np.cos(2 * np.pi * i / num_stations),
            center_y + radius_y * np.sin(2 * np.pi * i / num_stations)
        ]))
    stations.append(np.array([center_x, center_y]))
    agents = [
        Agent(s[0], s[1], np.random.uniform(0, 2 * np.pi))
        for _ in range(Config.N_AGENTS // len(stations) + 1) for s in stations
    ]
    agents = np.random.choice(agents, Config.N_AGENTS, replace=False).tolist()
    plt.ion()
    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(np.zeros((Config.HEIGHT, Config.WIDTH, 3)))

    for i in range(Config.ITERATIONS):
        for agent in agents:
            agent.sense_and_steer(pheromone_map, obstacle_map, stations)
            agent.move()
            ix, iy = int(agent.pos[0]), int(agent.pos[1])
            if 0 <= ix < Config.WIDTH and 0 <= iy < Config.HEIGHT: pheromone_map[iy, ix] += Config.DEPOSIT_AMOUNT
        pheromone_map *= Config.DECAY_FACTOR
        pheromone_map = gaussian_filter(pheromone_map, sigma=Config.BLUR_SIGMA)
        for sx, sy in stations:
            ix, iy = int(sx), int(sy)
            pheromone_map[iy, ix] = max(pheromone_map[iy, ix], Config.FOOD_PHEROMONE)
        if i > 0 and i % Config.REPOP_INTERVAL == 0:
            survivors = [
                agent for agent in agents
                if min([np.linalg.norm(agent.pos - s) for s in stations]) < Config.DEATH_DISTANCE_THRESHOLD
            ]
            num_dead = Config.N_AGENTS - len(survivors)
            if num_dead > 0:
                new_agents_pool = [
                    Agent(s[0], s[1], np.random.uniform(0, 2 * np.pi))
                    for _ in range(num_dead // len(stations) + 2) for s in stations
                ]
                new_agents = np.random.choice(new_agents_pool, num_dead, replace=False).tolist()
                agents = survivors + new_agents

        if i % 5 == 0:
            ax.set_title(f"Slime Network (Iteration: {i}/{Config.ITERATIONS}) | Agent Number: {len(agents)}")
            p_max = np.percentile(pheromone_map, 99.9)
            if p_max < 1e-5: p_max = 1e-5
            norm_pheromone = np.clip(pheromone_map / p_max, 0, 1)
            display_map = np.stack([norm_pheromone]*3, axis=-1)
            display_map[obstacle_map > 0] = [0.4, 0, 0]
            for sx, sy in stations:
                ix, iy = int(sx), int(sy)
                size = 4
                display_map[
                    max(0,iy-size):min(Config.HEIGHT,iy+size+1), max(0,ix-size):min(Config.WIDTH,ix+size+1)
                ] = [0, 0.9, 0.1]
            im.set_data(display_map)
            plt.pause(0.001)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()
