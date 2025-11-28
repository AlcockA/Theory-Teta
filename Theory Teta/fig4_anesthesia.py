# fig4_anesthesia.py
import numpy as np, matplotlib.pyplot as plt

N = 100
L = 50.0
dx = L/N
x = np.linspace(-L/2, L/2, N)
X,Y = np.meshgrid(x,x)
R = np.sqrt(X**2 + Y**2)

alpha_base = 0.4*(R/20)**2 - 1.1

# Начальное бодрствование
Psi = 0.8 * np.exp(-(R/15)**2) * np.exp(1j*4*np.arctan2(Y,X))

def step(alpha):
    lap = (np.roll(Psi,1,0)+np.roll(Psi,-1,0)+np.roll(Psi,1,1)+np.roll(Psi,-1,1)-4*Psi)/(dx**2)
    return Psi + 0.05*(lap - (alpha + np.abs(Psi)**2)*Psi)

# 1. Бодрствование
for _ in range(800): Psi = step(alpha_base)

# 2. Анестезия
for _ in range(600): Psi = step(alpha_base + 2.2)   # alpha > 0 → Ψ → 0

# 3. Восстановление
for _ in range(800): Psi = step(alpha_base)

fig, ax = plt.subplots(1,4,figsize=(18,4))
states = ["Бодрствование", "Начало анестезии", "Глубокая анестезия", "Восстановление"]
images = []

# Сохраняем 4 состояния
Psi_save = []
for alpha in [alpha_base, alpha_base+1.0, alpha_base+2.2, alpha_base]:
    for _ in range(300): Psi = step(alpha)
    Psi_save.append(np.abs(Psi).copy())

for i, img in enumerate(Psi_save):
    ax[i].imshow(img, cmap='inferno', vmin=0, vmax=1.1)
    ax[i].set_title(states[i], fontsize=14)
    ax[i].axis('off')

plt.suptitle("Теория Θ — Коллапс и восстановление сознания при анестезии", fontsize=18)
plt.tight_layout()
plt.savefig('FIG4_ANESTHESIA.png', dpi=500)
plt.show()