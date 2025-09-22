#  !/usr/bin/env python
#  -*- coding:utf-8 -*-

#  ==============================================
#  ·
#  · Author: PrimaPivot
#  ·
#  · Filename: 001-dropout.py
#  ·
#  · COPYRIGHT 2025
#  ·
#  · 呈现dropout对过拟合的影响
#  ·
#  ==============================================

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from functools import partial

# --- 0. 参数设置 ---
LEARNING_RATE = 5e-3
EPOCHS = 300
HIDDEN_SIZE = 256
WIDTH = 3
activation = jax.nn.relu # tanh sigmoid mish # relu 容易说，sigmoid可以呈现更丰富的现象
n_samples = 500
noise = 0.30

# --- 1. 数据准备 ---
X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

# --- 2. 模型定义：为处理【单个数据点】而设计 ---
class MLP(eqx.Module):
    layers: list
    use_dropout: bool
    dropout_rate: float

    def __init__(self, key, in_size, out_size, hidden_size, width, use_dropout=False, dropout_rate=0.5):
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        keys = jax.random.split(key, width + 1)
        self.layers = []
        self.layers.append(eqx.nn.Linear(in_size, hidden_size, key=keys[0]))
        for i in range(width - 1):
            self.layers.append(eqx.nn.Linear(hidden_size, hidden_size, key=keys[i+1]))
        self.layers.append(eqx.nn.Linear(hidden_size, out_size, key=keys[-1]))

    def __call__(self, x, key, training=False):
        keys = jax.random.split(key, len(self.layers) - 1)
        for i, layer in enumerate(self.layers[:-1]):
            x = activation(layer(x))
            if self.use_dropout and training:
                x = eqx.nn.Dropout(self.dropout_rate)(x, key=keys[i])
        x = self.layers[-1](x)
        return x.squeeze()

# --- 3. 辅助函数 ---
@eqx.filter_jit
def loss_fn(model, x_batch, y_batch, key):
    pred_y_batch = jax.vmap(model, in_axes=(0, None))(x_batch, key)
    return optax.sigmoid_binary_cross_entropy(pred_y_batch, y_batch).mean()

@eqx.filter_jit
def compute_accuracy(model, x_batch, y_batch, key):
    pred_y_batch = jax.vmap(model, in_axes=(0, None))(x_batch, key)
    pred_labels = (jax.nn.sigmoid(pred_y_batch) > 0.5).astype(jnp.int32)
    return jnp.mean(pred_labels == y_batch)

# --- 4. 训练步骤 ---
@eqx.filter_jit
def train_step(model, opt_state, x_batch, y_batch, key):
    def step_loss(params, static, x_batch, y_batch, key):
        model = eqx.combine(params, static)
        dropout_keys = jax.random.split(key, x_batch.shape[0])
        train_model_call = partial(model, training=True)
        pred_y_batch = jax.vmap(train_model_call, in_axes=(0, 0))(x_batch, dropout_keys)
        return optax.sigmoid_binary_cross_entropy(pred_y_batch, y_batch).mean()

    params, static = eqx.partition(model, eqx.is_array)
    grads = jax.grad(step_loss)(params, static, x_batch, y_batch, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state

# --- 5. 可视化函数 ---
def plot_decision_boundary(ax, model, key, title):
    ax.clear()
    h = .05
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_data = jnp.c_[xx.ravel(), yy.ravel()]

    eval_model_call = partial(model, training=False)
    Z = jax.vmap(eval_model_call, in_axes=(0, None))(grid_data, key)
    Z = (jax.nn.sigmoid(Z) > 0.5).reshape(xx.shape)

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.contourf(xx, yy, Z, cmap=cm_bright, alpha=0.6)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, s=25, alpha=0.7, label="Train")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, marker='x', s=40, label="Test")
    ax.set_title(title, fontsize=12)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.legend(loc='lower center')

def plot_learning_curves(ax, history, title):
    ax.clear()
    ax.plot(history["train_acc"], label=f"Train Acc (Final: {history['train_acc'][-1]:.3f})", color="blue")
    ax.plot(history["test_acc"], label=f"Test Acc (Final: {history['test_acc'][-1]:.3f})", color="red", linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='center')
    ax.grid(True)
    ax.set_ylim(0.4, 1.05)

# --- 6. 主训练循环 ---
key = jax.random.PRNGKey(0)
key, key_p, key_m = jax.random.split(key, 3)

model_perfectionist = MLP(key_p, 2, 1, HIDDEN_SIZE, WIDTH, use_dropout=False)
model_essentialist = MLP(key_m, 2, 1, HIDDEN_SIZE, WIDTH, use_dropout=True, dropout_rate=0.5)

optimizer = optax.adam(LEARNING_RATE)
opt_state_p = optimizer.init(eqx.filter(model_perfectionist, eqx.is_array))
opt_state_m = optimizer.init(eqx.filter(model_essentialist, eqx.is_array))

history_p = {"train_acc": [], "test_acc": []}
history_m = {"train_acc": [], "test_acc": []}

plt.ion()
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Perfectionist(No Dropout) vs. Essentialist(Dropout): A Tale of Two Strategies", fontsize=16)
plt.pause(3)  # 暂停几秒钟，方便后期剪辑

for epoch in tqdm(range(EPOCHS), desc="Training Models"):
    key, step_key_p, step_key_m, eval_key = jax.random.split(key, 4)

    model_perfectionist, opt_state_p = train_step(model_perfectionist, opt_state_p, X_train, y_train, step_key_p)
    model_essentialist, opt_state_m = train_step(model_essentialist, opt_state_m, X_train, y_train, step_key_m)

    # if epoch % 10 == 0 or epoch == EPOCHS - 1:
    train_acc_p = compute_accuracy(model_perfectionist, X_train, y_train, eval_key)
    test_acc_p = compute_accuracy(model_perfectionist, X_test, y_test, eval_key)
    train_acc_m = compute_accuracy(model_essentialist, X_train, y_train, eval_key)
    test_acc_m = compute_accuracy(model_essentialist, X_test, y_test, eval_key)

    history_p["train_acc"].append(train_acc_p)
    history_p["test_acc"].append(test_acc_p)
    history_m["train_acc"].append(train_acc_m)
    history_m["test_acc"].append(test_acc_m)

    plot_decision_boundary(axes[0, 0], model_perfectionist, eval_key, "The 'Perfectionist' (No Dropout)")
    plot_learning_curves(axes[1, 0], history_p, "Perfectionist's Learning Curve")

    plot_decision_boundary(axes[0, 1], model_essentialist, eval_key, "The 'Essentialist' (With Dropout)")
    plot_learning_curves(axes[1, 1], history_m, "Essentialist's Learning Curve")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.draw()

    if epoch == 0:
        plt.pause(3)  # 展示数据初始情况
    else:
        plt.pause(0.1)

plt.ioff()
print("\n--- Training Complete ---")
print(f"Final Test Accuracy (Perfectionist): {history_p['test_acc'][-1]:.4f}")
print(f"Final Test Accuracy (Essentialist): {history_m['test_acc'][-1]:.4f}")
plt.show()
