# Bilan Technique — Version 1 (V1.0 à V1.2)
**Projet Master (PMI) :** Apport de l'IA dans les systèmes embarqués radiofréquences  
**Axe d'étude :** Linéarisation par Prédistorsion Numérique (DPD) assistée par IA pour amplificateur de puissance (PA)  

---

## 1. Objectif de la V1
Valider la faisabilité algorithmique de bout en bout (*Proof of Concept*) en simulation Python : démontrer qu'un réseau de neurones ultra-léger, compatible avec les contraintes mémoire d'un système embarqué (FPGA/SDR), peut corriger les non-linéarités et les effets de mémoire d'un amplificateur de puissance RF.

---

## 2. Architecture de simulation et modèle IA retenus

### Chaîne RF simulée (`dpd_ai.py`)
* **Signal d'entrée :** Modulation **16-QAM** suréchantillonnée ($\times 4$) et mise en forme par un filtre FIR de Nyquist (fenêtre de Kaiser, 33 coefficients) afin d'observer la remontée spectrale hors-bande (*spectral regrowth*).
* **Amplificateur de puissance (PA) :** Modèle polynomial complexe avec mémoire d'ordre 5 et profondeur de mémoire $M=2$ ($x[n], x[n-1], x[n-2]$), reproduisant la compression d'amplitude (AM/AM), la distorsion de phase (AM/PM) et l'inertie thermique/électrique du composant.

### Bloc IA : RVTDNN Résiduel
Pour minimiser le coût calculatoire, l'architecture choisie est un **RVTDNN** (*Real-Valued Time-Delay Neural Network*) résiduel :
* **Vecteur d'entrée (12 features) :** $[I(n), Q(n), |x(n)|, |x(n)|^2]$ sur une fenêtre glissante de 3 échantillons ($n, n-1, n-2$). L'injection explicite de l'enveloppe $|x(n)|$ permet au réseau de capturer les non-linéarités sans multiplier les couches.
* **Topologie :** MLP compact `12 -> 24 -> 16 -> 2` avec activations `Tanh` et connexion résiduelle ($\hat{x}[n] = x[n] + \text{NN}(x[n])$), soit **746 paramètres** au total (~2,98 Ko en `float32`).
* **Stratégie d'apprentissage :** *Indirect Learning Architecture* (ILA). Le réseau est entraîné en post-inverse (entrée = sortie normalisée du PA $y[n]/G$, cible = entrée du PA $x[n]$) puis copié en amont du PA pour l'inférence.

---

## 3. Journal d'expérimentation et résolution des verrous techniques

| Version | Configuration testée | NMSE sans DPD | NMSE avec IA-DPD | Gain | Diagnostic et enseignement technique |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **V1.0** | MLP direct (`Tanh`, 60 époques, RMS 0.65) | -16,09 dB | -22,97 dB | +6,88 dB | Le réseau doit reconstruire tout le signal : remontée d'un plancher de bruit à -45 dB sur la PSD et dispersion des coins de la constellation. |
| **V1.1** | Architecture résiduelle (`LeakyReLU`, init à 0) | -16,09 dB | -23,90 dB | +7,81 dB | L'initialisation à zéro strict freine les premiers gradients et `LeakyReLU` génère des harmoniques haute fréquence sur le spectre. |
| **V1.2** | Résiduel `Tanh`, init faible (`1e-3`), *Cosine Annealing* + **Calibration dynamique ILA** | **-22,60 dB** | **-44,24 dB** | **+21,64 dB** | **Succès.** L'entraînement sur une plage élargie (RMS 0.68 vs test à RMS 0.48) supprime l'extrapolation hors zone de saturation. |

### Point clé identifié en V1.2 : la calibration de plage dynamique en ILA
En V1.0/V1.1, l'amplificateur écrasait les crêtes du signal d'entraînement au-delà de $|y| \approx 0,85$. Lors du test, les 4 sommets extérieurs de la constellation 16-QAM atteignaient $|x| \approx 1,25$, forçant le réseau à extrapoler dans une zone jamais vue. En entraînant le modèle avec une puissance RMS supérieure (`0.68`) à celle du signal nominal (`0.48`), la sortie saturée $y_{\text{train}}$ couvre 100 % de la dynamique d'entrée $x_{\text{test}}$, faisant chuter l'erreur globale à **-44,24 dB**.

---

## 4. Résultats visuels (V1.2)
* **Densité Spectrale de Puissance (PSD) :** Réduction de plus de 18 dB des lobes secondaires adjacents (passant de -36 dB à environ -55 dB).
* **Caractéristique AM/AM :** Redressement complet de la courbe de compression sur la diagonale idéale.
* **Constellation 16-QAM :** Recentrage précis des 16 symboles, y compris les 4 symboles de crête aux extrémités.

*(Voir la capture associée : `fig_v1_2_results.png`)*

---

## 5. Limites de la V1 et objectifs de la V2
La V1 valide la performance algorithmique en virgule flottante (`FP32`) sur un modèle d'amplificateur statique. Pour répondre aux contraintes d'un système embarqué (SWaP), la **V2** s'articulera autour de trois axes :
1. **V2.1 — Comparaison critique :** Évaluer notre IA-DPD face à une DPD mathématique classique (Polynôme à mémoire résolu par moindres carrés), notamment en cas de dérive thermique de l'amplificateur.
2. **V2.2 — Contraintes matérielles FPGA :** Simuler la quantification des poids et activations (`FP32` $\rightarrow$ `Int8`, `Int6`, `Int4`) pour mesurer l'impact sur la linéarité RF et l'empreinte mémoire.
3. **V2.3 — Bilan énergétique (SWaP) :** Calculer les métriques RF standards (**ACPR**, **EVM**) et le coût matériel exact (**MACs/échantillon**, estimation en Watts à fréquence d'échantillonnage réelle).