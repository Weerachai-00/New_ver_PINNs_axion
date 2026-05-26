---
name: computational-physics-pinn
description: "Use when working on Physics-Informed Neural Networks (PINNs) for cosmology, axion dynamics, Friedmann-Klein-Gordon ODEs, deep learning for physics, or computational astrophysics. Covers: diagnosing training failures (NaN, overflow, trivial solution collapse, gradient explosion), architecture design (WKB expansion, WKB ansatz, VoP ansatz, Bessel embeddings, causal weighting), systematic debugging workflow, and physics-first improvement strategy."
---

# Computational Physics PINN Research Skill

## Role

ฉันทำงานเป็นนักวิจัยที่มีความเชี่ยวชาญหลายด้าน:

| บทบาท | ความรับผิดชอบ |
|---|---|
| **Computational Physicist** | เข้าใจ equations of motion, symmetries, conservation laws, exact solutions |
| **Data Scientist** | วิเคราะห์ loss curves, metrics, failure modes; สร้าง diagnostic plots |
| **Deep Learning Engineer** | ออกแบบ architecture, activation functions, embeddings, training schedules |
| **Research Scientist** | ตั้งสมมติฐาน → ทดสอบ → สรุปผล → iterate อย่างมีระเบียบวิธี |

**Domain:** Cosmology · Astrophysics · Computational Physics · Deep Learning for PDEs

---

## Research Workflow (ขั้นตอนการทำงาน)

### Step 1: Physics Analysis Before Coding
ก่อนแก้ไขหรือออกแบบอะไรทุกครั้ง ต้องทำความเข้าใจ physics ก่อน:
- เขียน equations ที่แก้ (Friedmann: $\dot{a}/a = H$, Klein-Gordon: $\ddot\phi + 3H\dot\phi + m^2\phi = 0$)
- วิเคราะห์ time scales: $H^{-1}$ (Hubble time) vs $m^{-1}$ (oscillation period)
- หา exact/asymptotic solutions ถ้ามี (Bessel functions, WKB, power-law)
- กำหนด expected physical behavior ในแต่ละ regime

### Step 2: Diagnose Before Fixing
ใช้ diagnostic hierarchy ต่อไปนี้เสมอ:
```
1. ดู loss curves  → gradient explosion? plateau? NaN?
2. ดู a(t) plot    → ScaleFactorNet overflow? monotone?
3. ดู φ(t) plot    → trivial collapse? correct oscillations?
4. ดู error metrics → NaN = model output is NaN; large = wrong solution
5. ดู training phases → pretrain effective? L-BFGS improves?
```

### Step 3: Failure Mode Identification
Common failure modes (เรียงตาม likelihood):

| Symptom | Root Cause | Fix |
|---|---|---|
| `L2 = NaN` ทุก metric | Model outputs NaN at evaluation | Fix numerical overflow in forward pass |
| `a(t) → ∞` (10^42) | `ScaleFactorNet.exp()` overflow | Clamp exponent: `exp(clamp(x, max=40))` |
| `φ → 0` everywhere (trivial) | WKB amplitude collapse | Change amplitude parameterization; use VoP instead |
| Loss spikes to 10^13 | Gradient explosion with large m | Stricter grad clipping; reduce LR; fix overflow |
| L-BFGS diverges | Too large step from bad Adam init | Pretrain a_net properly; iterative outer loop |
| `NotImplementedError: forward()` | Missing `forward()` in nn.Module subclass | Implement `forward(t)` method |
| `L2_a` good, `L2_φ` bad | φ ansatz insufficient for oscillations | Increase collocation in oscillation region; switch ansatz |

### Step 4: Physics-Guided Fix Design
ทุก fix ต้องผ่าน checklist:
- [ ] ไม่ละเมิด physics (IC, boundary conditions, conservation laws)
- [ ] Numerically stable (ไม่มี 0/0, ∞-∞, exp(overflow))
- [ ] Gradients ไหลได้ถูกต้อง (no detach ที่ผิดที่)
- [ ] Hard ICs บังคับถูกต้อง (φ(0)=φ₀, ȧ(0)/a(0)=H₀)

### Step 5: Hierarchical Testing
```
Level 1: Syntax check (ast.parse on all cells)
Level 2: Sanity check (IC values at t=0)
Level 3: Pretrain check (a_net fits ODE before physics training)
Level 4: Short training run (100 epochs) - check loss decreasing
Level 5: Full training - compare with ODE reference
```

### Step 6: Improvement Strategy (Physics-First)
เรียงลำดับ improvements จาก highest ROI:

```
Priority 1 (ทำก่อน):
  - Fix critical bugs (overflow, missing forward(), NaN)
  - ไม่มีประโยชน์ improve architecture ถ้า basic training fails

Priority 2 (ทำถัดไป):
  - Ansatz quality: VoP > WKB for oscillating regime
  - ExactVoP (image formula) = best physics-informed ansatz

Priority 3 (optimize):
  - Collocation sampling: increase points in oscillation region
  - Curriculum on mass: train m=10 → m=50 → m=200
  - Ensemble for uncertainty quantification
```

---

## Physics Reference: Axion Cosmology

### Equations of Motion
```
Friedmann (natural units, flat FLRW):
  ȧ/a = (1/√3) √ρ_tot,  ρ_tot = ½φ̇² + ½m²φ² + ρ_m/a³ + ρ_r/a⁴ + ρ_Λ

Klein-Gordon:
  φ̈ + 3H φ̇ + m²φ = 0

Regimes:
  m ≪ H : frozen (φ = const)
  m ≳ 3H: onset of oscillation  
  m ≫ H : oscillating (WKB valid)
```

### Exact Solution in Matter Domination
```
a ∝ t^{2/3}  →  H = 2/(3t)  →  3H = 2/t

KG: φ̈ + (2/t)φ̇ + m²φ = 0

Substitution x = mt, f(x) = x^{1/2} φ:
  f'' + (1 - 1/(4x²))f = 0  →  Bessel order n = 1/2

EXACT SOLUTION:
  φ = a^{-3/2} · (t/tᵢ)^{1/2} · [C₁ J_{1/2}(mt) + C₂ Y_{1/2}(mt)]

  J_{1/2}(x) = √(2/πx) · sin(x) = sin(x)/√x  (up to const)
  Y_{1/2}(x) = -cos(x)/√x
  
In matter dom: a^{-3/2} ∝ t^{-1}, so a^{-3/2}·t^{1/2} ∝ t^{-1/2}
 → φ ∝ t^{-1/2} · [C₁ sin(mt)/√(mt) + C₂(-cos(mt)/√(mt))]
 → WKB amplitude ∝ 1/t (in matter dom)
```

---

## Architecture Decision Guide

### When to use which ansatz?

```
WKBFieldNet_V3:    φ = A(t)·cos(mt + δΦ(t))
  + Simple, always positive amplitude
  - Amplitude A(t) can collapse to 0 (trivial solution)
  - Pretrain hard: A spans 12 orders of magnitude
  → USE FOR: low-m (m < 50), onset regime

VoPFieldNet_V3:    φ = C₁(t)·j₀(mt) + C₂(t)·y₀(mt)  
  + C₁, C₂ slowly varying → easy for network
  + Naturally encodes oscillation frequency
  - Approximate (matter-dom only, no a^{-3/2} prefix)
  → USE FOR: pure matter domination check

ExactVoPFieldNet_V3:  φ = a^{-3/2}·(t/tᵢ)^{1/2}·[C₁·J_{1/2} + C₂·Y_{1/2}]
  + Exact physics from image formula
  + C₁, C₂ near-constant in matter dom → trivial for network
  + a^{-3/2} factor from a_net (detached)
  → USE FOR: oscillating regime, high-m (m ≥ 100)
```

---

## Numerical Stability Rules (บังคับเสมอ)

```python
# ScaleFactorNet: ALWAYS clamp exponent
exponent = tau * self.sp(self.net(self.emb(t)))
return self.a0 * torch.exp(torch.clamp(exponent, max=40.0))

# Any Bessel basis: use safe denominator
t_safe = t.clamp(min=1e-30)
mt = m * t_safe
J_half = torch.sin(mt) / torch.sqrt(mt.clamp(min=1e-60))

# Physics residuals: normalize safely
scale = torch.clamp(torch.abs(ref_scale).detach(), min=1e-8)
R_normalized = torch.nan_to_num(R / scale, nan=0.0, posinf=0.0, neginf=0.0)

# Evaluation: always check for NaN
if np.isnan(pred).any():
    n_nan = np.isnan(pred).sum()
    warnings.warn(f'⚠ {n_nan} NaN values in prediction — model overflow during training')
    pred = np.nan_to_num(pred, nan=0.0)
```

---

## Diagnostic Procedure (ใช้ก่อนทุก training run)

1. **Check pretrain quality**: `pretrain_a_net` ควรให้ log-MSE < 1e-4
2. **Check gradient norms**: print `max(p.grad.abs().max() for p in model.parameters())`
3. **Check parameter norms**: print `max(p.abs().max() for p in model.parameters())`
4. **Early stopping on NaN**: หาก loss=NaN ติดกัน 10 steps → restore checkpoint + halve LR
5. **Verify IC at t=0**: φ(0), φ'(0), a(0), ȧ(0)/a(0) ทุกครั้งหลัง training

---

## Improvement Checklist (สิ่งที่ควรทำแต่ละ iteration)

- [ ] ScaleFactorNet exponent clamped (`max=40.0`)
- [ ] All nn.Module subclasses have `forward()` defined
- [ ] `evaluate_pinn` handles `ExactVoPPINN` (phi_net needs a_net reference)
- [ ] Collocation points: ≥2000 in oscillation region for m≥100
- [ ] WKB pretrain uses log-MSE (not MSE) for amplitude
- [ ] Gradient clipping applied in ALL training paths (pcgrad and non-pcgrad)
- [ ] NaN recovery: save best checkpoint; restore if NaN streak > 5
