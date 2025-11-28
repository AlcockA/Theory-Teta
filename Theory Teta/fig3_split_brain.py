# fig3_split_brain_v2_1.py
import numpy as np
import matplotlib.pyplot as plt

# --- ПАРАМЕТРЫ МОДЕЛИ ---
N = 140                                     # Размер сетки
L = 100.0
dx = L / N
dt = 0.05                                   # Шаг времени (уменьшен для стабильности)

# Координаты и карта потенциала Alpha
x = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, x)

# Две ямы активности — два полушария
R_left  = np.sqrt((X + 30)**2 + Y**2)
R_right = np.sqrt((X - 30)**2 + Y**2)
# Alpha < 0 в ямах, Alpha > 0 вне
alpha_map = 0.5 * ((R_left/25)**2 + (R_right/25)**2) - 1.8 

# Начальное поле — слабый шум (спонтанное нарушение симметрии)
Psi = 0.08 * (np.random.randn(N,N) + 1j*np.random.randn(N,N))

# --- ФУНКЦИЯ ЛАПЛАСИАНА ---
def laplacian(Z):
    # Лапласиан с периодическими условиями
    return (np.roll(Z,1,0) + np.roll(Z,-1,0) +
            np.roll(Z,1,1) + np.roll(Z,-1,1) - 4*Z) / (dx**2)

# --- ПАРАМЕТРЫ РАЗРЕЗА ---
center_line_index = N // 2
barrier_width = 2                           # Толщина разреза в пикселях (2-3 достаточно)

print(f"Запуск симуляции Split-brain (N={N}x{N})...")

# --- ЭВОЛЮЦИЯ ---
for i in range(8000): # Увеличиваем шаги для гарантированной стабилизации фаз
    lap = laplacian(Psi)
    
    # ⚡️ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Имитация коллозотомии
    # Обнуляем горизонтальную диффузию (связь) через центральную линию.
    # Это создает физический барьер, не давая фазам выровняться.
    lap[:, center_line_index - barrier_width : center_line_index + barrier_width] = 0.0

    # Уравнение TDGL: dPsi/dt = D*Laplacian(Psi) - (alpha + |Psi|^2)*Psi
    # Увеличиваем D (коэффициент 0.9 → 1.2) для более быстрого упорядочивания внутри полушарий
    Psi += dt * (1.2 * lap - (alpha_map + np.abs(Psi)**2) * Psi)
    
    if i % 1000 == 0:
        print(f"  Шаг {i}/8000")

# --- ВИЗУАЛИЗАЦИЯ ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 1. Амплитуда |Psi| (Интенсивность сознания)
im1 = ax1.imshow(np.abs(Psi), cmap='inferno', vmin=0, vmax=1.1, origin='lower')
ax1.set_title("Split-brain: Два независимых центра A(x)", fontsize=16)
ax1.axis('off')

# 2. Фаза Theta (Содержание сознания / Доменная стенка)
mask = np.abs(Psi) < 0.15 # Скрываем шум там, где нет активности
phase_masked = np.ma.array(np.angle(Psi), mask=mask)
im2 = ax2.imshow(phase_masked, cmap='hsv', vmin=-np.pi, vmax=np.pi, origin='lower')
ax2.set_title("Фаза Θ: Доменная стенка между независимыми субъектами", fontsize=16)
ax2.axis('off')

plt.suptitle("Теория Θ v2.1 — Успешная Split-brain симуляция", fontsize=20)
plt.tight_layout()
plt.savefig('FIG3_SPLIT_BRAIN_V2_1.png', dpi=500, bbox_inches='tight')
plt.show()

print("Готово! Сохранено как FIG3_SPLIT_BRAIN_V2_1.png")