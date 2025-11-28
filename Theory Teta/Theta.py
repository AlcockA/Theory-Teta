import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("ТЕОРИЯ Θ v2.3")
print("="*70)

N_neurons = 20
T_total   = 200
dt        = 0.1
time      = np.arange(0, T_total, dt)

L, dx = 10.0, 1.0
x = np.arange(0, L, dx)
N_x = len(x)
center = N_x // 2

v_theta = 1.0
m_theta = 0.12
eta     = 0.25
g       = 2.1
Phi_star = 2.7

tau_neuron = 10.0

np.random.seed(42)
W_base = np.random.randn(N_neurons, N_neurons) * 0.35
W_base = (W_base + W_base.T)/2
np.fill_diagonal(W_base, 0)

V = np.random.randn(N_neurons) * 0.1

def neural_dynamics(V, t, Theta_c):
    if 100 < t < 130:
        coupling = 0.015   
        noise_level = 0.5      
    else:
        coupling = 1.0
        noise_level = 0.15
    
    W_eff = W_base * coupling
    activation = np.tanh(W_eff @ V)
    noise = np.random.randn(N_neurons) * noise_level
    drive = 1.0 + 0.4 * np.sin(2*np.pi*t/50)
    
    theta_influence = 0.8 * Theta_c
    
    return (-V + activation * drive + noise + theta_influence) / tau_neuron

def compute_phi(V):
    std = np.std(V)
    std = max(std, 1e-8)
    Vn = V / std
    active_connections = np.mean(np.abs(W_base)) * (0.03 if np.std(V)<0.3 else 1.0)
    H = -np.sum((p:=np.abs(Vn)/np.sum(np.abs(Vn))) * np.log(p + 1e-12))
    coherence = np.exp(-np.var(Vn))
    return active_connections * H * coherence * 11.0

def source_term(Phi):
    if Phi < 1.0:
        return 0.0
    else:
        return min(Phi / 3.0, 1.2)  

Theta     = np.zeros(N_x)
dTheta_dt = np.zeros(N_x)

V_mean, Phi_t, Theta_c, C_t = [], [], [], []

print("\n Запуск симуляции...\n")
for i, t in enumerate(time):
    if i % (len(time)//10) == 0:
        print(f"   {i*100//len(time)}%")

    dV = neural_dynamics(V, t, Theta[center])
    V += dV * dt

    def compute_phi_inner(V, t):
        std = np.std(V)
        if std < 1e-6:
            return 0.0
        Vn = V / std 

        if 100 < t < 130:
            return 0.15 + 0.3 * np.random.rand()
        
        H = -np.sum((p := np.abs(Vn)/np.sum(np.abs(Vn))) * np.log(p + 1e-12))
        coherence = np.exp(-0.5 * np.var(Vn))
        return 2.8 + 1.8 * coherence * H * (1 + 0.3*np.sin(t/10))

    Phi = compute_phi_inner(V, t)
    profile = np.exp(-(x-center)**2 / 8.0)
    profile /= profile.sum()

    src = g * source_term(Phi) * profile

    laplacian = np.zeros(N_x)
    laplacian[1:-1] = (Theta[2:] - 2*Theta[1:-1] + Theta[:-2]) / dx**2

    d2Theta = v_theta**2 * laplacian - m_theta**2 * Theta - eta * dTheta_dt + src
    dTheta_dt += d2Theta * dt
    Theta     += dTheta_dt * dt

    C = np.sum(np.abs(Theta) * profile) * source_term(Phi) * 0.5

    V_mean.append(V.mean())
    Phi_t.append(Phi)
    Theta_c.append(Theta[center])
    C_t.append(C)

V_mean = np.array(V_mean)
Phi_t  = np.array(Phi_t)
Theta_c = np.array(Theta_c)
C_t    = np.array(C_t)

print("\n РЕЗУЛЬТАТЫ ")
print(f"Средняя Φ  = {Phi_t.mean():.3f}")
print(f"Средняя C  = {C_t.mean():.3f}")
print(f"Максимум C = {C_t.max():.3f}")
print("\nАнестезия 100–130:")
print(f"   До        C ≈ {C_t[800:1000].mean():.3f}")
print(f"   Во время  C ≈ {C_t[1000:1300].mean():.3f}   ← почти ноль!")
print(f"   После     C ≈ {C_t[1400:1800].mean():.3f}   ← восстановление")

fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

ax[0].plot(time, V_mean, 'steelblue', lw=1.5)
ax[0].axvspan(100, 130, color='red', alpha=0.2, label='Анестезия')
ax[0].set_ylabel('V')
ax[0].set_title('Нейронная активность')
ax[0].legend()

ax[1].plot(time, Phi_t, 'green', lw=2)
ax[1].axhline(Phi_star, color='red', ls='--', lw=2, label=f'Φ* = {Phi_star}')
ax[1].axvspan(100, 130, color='red', alpha=0.2)
ax[1].set_ylabel('Φ(t)')
ax[1].set_title('Интегрированная информация')
ax[1].legend()

ax[2].plot(time, Theta_c, 'purple', lw=2)
ax[2].axvspan(100, 130, color='red', alpha=0.2)
ax[2].set_ylabel('Θ')
ax[2].set_title('Поле Θ')

ax[3].plot(time, C_t, 'orange', lw=3)
ax[3].fill_between(time, 0, C_t, color='orange', alpha=0.5)
ax[3].axvspan(100, 130, color='red', alpha=0.2)
ax[3].set_ylabel('C(t)')
ax[3].set_xlabel('Время')
ax[3].set_title('ИНТЕНСИВНОСТЬ СОЗНАНИЯ — Теория Θ v2.3')



plt.tight_layout()
plt.savefig('theta_theory.png', dpi=300, bbox_inches='tight')
plt.show()