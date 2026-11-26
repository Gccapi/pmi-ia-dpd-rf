import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import firwin, upfirdn, welch
import matplotlib.pyplot as plt

# Fixer les seed pour la reproduct
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# GÉNÉRATION DU SIGNAL RF (16-QAM suréchantillonné)
# ==========================================
def generate_rf_signal(num_symbols=6000, sps=4, num_taps=33, rms_level=0.48):
    # Symboles 16-QAM normalisés
    alphabet = np.array([-3, -1, 1, 3])
    re = np.random.choice(alphabet, num_symbols)
    im = np.random.choice(alphabet, num_symbols)
    symbols = (re + 1j * im) / np.sqrt(10.0)
    
    # Filtre de mise en forme (bande limitée)
    rrc_filter = firwin(num_taps, cutoff=1.0 / sps, window=('kaiser', 5.0))
    rrc_filter = rrc_filter / np.sum(rrc_filter) * sps
    
    # Suréchantillonnage et filtrage
    signal = upfirdn(rrc_filter, symbols, up=sps)
    
    # Normalisation de la puissance d'entrée (proche zone de compression)
    signal = signal / np.sqrt(np.mean(np.abs(signal)**2)) * rms_level
    delay = (num_taps - 1) // 2
    return signal, symbols, delay

# ==========================================
# MODÈLE D'AMPLIFICATEUR NON-LINÉAIRE AVEC MÉMOIRE
# ==========================================
def power_amplifier(x):
    """Modèle polynomial avec mémoire (ordres 1, 3, 5 + 2 taps de mémoire)"""
    x_m1 = np.pad(x, (1, 0))[:-1]
    x_m2 = np.pad(x, (2, 0))[:-2]
    
    # Gain linéaire G = 1.0 + 0.05j, compression ordre 3 et 5 + déphasage AM/PM
    y = (1.0 + 0.05j) * x + (0.08 - 0.03j) * x_m1 - (0.02 + 0.01j) * x_m2 \
        + (-0.28 + 0.14j) * x * (np.abs(x)**2) \
        + (-0.06 + 0.03j) * x_m1 * (np.abs(x_m1)**2) \
        + (0.05 - 0.04j) * x * (np.abs(x)**4)
    return y

LINEAR_GAIN = 1.0 + 0.05j

# ==========================================
# EXTRACTION DE FEATURES & BLOC IA (RVTDNN)
# ==========================================
def extract_features(sig, memory_depth=3):
    """Crée les entrées [I, Q, |x|, |x|^2] sur une fenêtre temporelle"""
    features = []
    for m in range(memory_depth):
        s_m = np.pad(sig, (m, 0))[:-m] if m > 0 else sig
        env = np.abs(s_m)
        features.extend([s_m.real, s_m.imag, env, env**2])
    return np.stack(features, axis=1).astype(np.float32)

class LightweightDPD(nn.Module):
    def __init__(self, in_features=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 24),
            nn.Tanh(),
            nn.Linear(24, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        # Initialisation proche de 0 (mais non nulle) pour laisser passer le gradient dès le début
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, x):
        return x[:, :2] + self.net(x)

def apply_dpd(model, sig, memory_depth=3):
    model.eval()
    feats = torch.tensor(extract_features(sig, memory_depth))
    with torch.no_grad():
        out = model(feats).numpy()
    return out[:, 0] + 1j * out[:, 1]

# ==========================================
# TRAINING (Indirect Learning Architecture)
# ==========================================
MEMORY_DEPTH = 3
x_train, _, _ = generate_rf_signal(num_symbols=12000, rms_level = 0.68)
y_train = power_amplifier(x_train) / LINEAR_GAIN  # Sortie PA normalisée

# En ILA : Entrée du NN = sortie PA normalisée | Cible = entrée PA
X_tensor = torch.tensor(extract_features(y_train, MEMORY_DEPTH))
Y_tensor = torch.tensor(np.stack([x_train.real, x_train.imag], axis=1).astype(np.float32))

loader = DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=512, shuffle=True)
model = LightweightDPD(in_features=4 * MEMORY_DEPTH)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
criterion = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"Nombre total de paramètres du modèle DPD : {total_params}")


for epoch in range(150):
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
    scheduler.step()

# ==========================================
# TEST ET MÉTRIQUES RF (NMSE, PSD, Constellation)
# ==========================================
x_test, syms_test, delay = generate_rf_signal(num_symbols=5000, rms_level = 0.48)

# Cas 1 : Sans DPD
y_no_dpd = power_amplifier(x_test) / LINEAR_GAIN

# Cas 2 : Avec IA-DPD en amont du PA
x_predistorted = apply_dpd(model, x_test, MEMORY_DEPTH)
y_with_dpd = power_amplifier(x_predistorted) / LINEAR_GAIN

def compute_nmse(ref, est):
    # On ignore les bords liés au filtre
    r, e = ref[100:-100], est[100:-100]
    return 10 * np.log10(np.sum(np.abs(r - e)**2) / np.sum(np.abs(r)**2))

print(f"NMSE sans DPD : {compute_nmse(x_test, y_no_dpd):.2f} dB")
print(f"NMSE avec IA-DPD : {compute_nmse(x_test, y_with_dpd):.2f} dB")

# --- Visualisation ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 1. Densité Spectrale de Puissance (PSD)
for sig, label, col in [(x_test, 'Signal Idéal', 'black'),
                        (y_no_dpd, 'PA sans DPD', 'red'),
                        (y_with_dpd, 'PA avec IA-DPD', 'green')]:
    f, Pxx = welch(sig, fs=4.0, nperseg=1024, return_onesided=False)
    axes[0].plot(np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(Pxx)), label=label, color=col, alpha=0.85)
axes[0].set_title("Densité Spectrale de Puissance (PSD)")
axes[0].set_xlabel("Fréquence normalisée")
axes[0].set_ylabel("Puissance (dB/Hz)")
axes[0].set_ylim([-90, 5])
axes[0].grid(True)
axes[0].legend()

# 2. Courbe AM/AM (Linéarité)
axes[1].scatter(np.abs(x_test[::10]), np.abs(y_no_dpd[::10]), s=2, color='red', alpha=0.4, label='Sans DPD (Saturation)')
axes[1].scatter(np.abs(x_test[::10]), np.abs(y_with_dpd[::10]), s=2, color='green', alpha=0.4, label='Avec IA-DPD (Linéarisé)')
axes[1].plot([0, 1.3], [0, 1.3], 'k--', label='Idéal')
axes[1].set_title("Caractéristique AM/AM")
axes[1].set_xlabel("Amplitude d'entrée |x|")
axes[1].set_ylabel("Amplitude de sortie |y|/G")
axes[1].grid(True)
axes[1].legend()

# 3. Constellation 16-QAM (échantillonnée au rythme symbole)
rx_no_dpd = y_no_dpd[delay::4][:len(syms_test)]
rx_with_dpd = y_with_dpd[delay::4][:len(syms_test)]
axes[2].scatter(rx_no_dpd.real[:1000], rx_no_dpd.imag[:1000], s=8, color='red', alpha=0.5, label='Sans DPD')
axes[2].scatter(rx_with_dpd.real[:1000], rx_with_dpd.imag[:1000], s=8, color='green', alpha=0.6, label='Avec IA-DPD')
axes[2].set_title("Constellation 16-QAM en sortie")
axes[2].set_xlabel("En phase (I)")
axes[2].set_ylabel("Quadrature (Q)")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()