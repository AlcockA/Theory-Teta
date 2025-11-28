# fig2_emergence_ULTIMATE.py  ← КОПИРУЙ ЭТОТ И ТОЛЬКО ЭТОТ
import numpy as np
import matplotlib.pyplot as plt

N = 100
L = 50.0
dx = L / N
dt = 0.02

beta = 1.0
D = 1.0

x = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)

alpha_map = 0.6 * (R / (L/4))**2 - 1.0

Psi = 0.05 * (np.random.randn(N, N) + 1j*np.random.randn(N, N))

def laplacian(Z):
    return (np.roll(Z,1,0) + np.roll(Z,-1,0) +
            np.roll(Z,1,1) + np.roll(Z,-1,1) - 4*Z) / (dx**2)

# Кадры, которые хотим сохранить
target_steps = [0, 300, 700, 1200, 2200]
saved_frames = []

current_step = 0

# Сразу сохраняем t=0
saved_frames.append((np.abs(Psi).copy(), np.angle(Psi).copy()))

print("Запуск симуляции...")

while len(saved_frames) < 5:
    lap = laplacian(Psi)
    Psi += dt * (D * lap - (alpha_map + beta * np.abs(Psi)**2) * Psi)
    Psi += 0.01 * (np.random.randn(N,N) + 1j*np.random.randn(N,N))

    current_step += 1

    # Сохраняем кадр, когда достигли нужного времени
    if current_step >= target_steps[len(saved_frames)]:
        saved_frames.append((np.abs(Psi).copy(), np.angle(Psi).copy()))
        print(f"  Сохранён кадр t ≈ {current_step}")

# Рисуем 5 колонок
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
times = [0, 300, 700, 1200, 2200]

for i in range(5):
    amp = saved_frames[i][0]
    phase = saved_frames[i][1]

    axes[0,i].imshow(amp, cmap='inferno', vmin=0, vmax=1.1)
    axes[0,i].set_title(f"t = {times[i]}", fontsize=14)
    axes[0,i].axis('off')

    mask = amp < 0.12
    phase_masked = np.ma.array(phase, mask=mask)
    axes[1,i].imshow(phase_masked, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1,i].set_title(f"t = {times[i]}", fontsize=14)
    axes[1,i].axis('off')

axes[0,0].set_ylabel("Интенсивность |Ψ|", fontsize=15)
axes[1,0].set_ylabel("Квалиа — фаза Θ", fontsize=15)

plt.suptitle("Теория Θ v2.0 — Рождение сознания из хаоса\nСпонтанное нарушение U(1)-симметрии", 
             fontsize=18, y=0.98)
plt.tight_layout()
plt.savefig('FIG2_BIRTH_OF_CONSCIOUSNESS.png', dpi=500, bbox_inches='tight')
plt.show()

print("ГОТОВО! Файл FIG2_BIRTH_OF_CONSCIOUSNESS.png сохранён.")