# Debugging Guide for Training Loss Spikes

## What We Added

### 1. Correlation Loss Debugging
**File**: `training/loss_functions.py` (lines 74-117)

**What to look for in logs:**
- 🔴 `CorrelationLoss: pred contains NaN/Inf` - Model output is broken
- ⚠️  `CorrelationLoss: X predictions have zero std` - Model outputting constant values
- ⚠️  `CorrelationLoss: High loss X.XX` - Correlation loss > 2.0 (should be ≤ 2.0)

### 2. Loss Component Breakdown
**File**: `training/loss_functions.py` (lines 195-217)

**What to look for in logs:**
- 🔴 `TOTAL LOSS IS NaN/Inf!` - Critical failure, shows which component broke
- 🔶 `HIGH LOSS DETECTED: X.XX` - Loss > 5.0, shows breakdown of all components
  - Check which component is large: EEG, Corr, Spiking, or Reg

### 3. Gradient Monitoring
**File**: `training/trainer.py` (lines 262-285)

**What to look for in logs:**
- 🔴 `NaN/Inf gradient in <layer_name>` - Gradient explosion in specific layer
- 🔶 `Large gradient detected: X.XX` - Gradient > 10.0 before clipping
- ⚠️  `Clipped gradient norm: X.XX → 1.0` - Gradient > 5.0 after backward

### 4. Model Output Checks
**File**: `training/trainer.py` (lines 244-253)

**What to look for in logs:**
- 🔴 `Model output contains NaN/Inf!` - Forward pass producing invalid values
- ⚠️  `Large model output: max=X.XX` - Output values > 10.0 (should be ~[-3, 3])

## How to Diagnose Loss Spikes

### Scenario 1: Loss jumps to 28.76
**Look for this sequence in logs:**
1. Before spike: Check last good batch's loss components
2. At spike: Look for 🔶 `HIGH LOSS DETECTED` - which component is huge?
3. After spike: Did gradients explode? Look for 🔶 `Large gradient detected`

**Likely culprits:**
- **Correlation loss > 2.0**: Model output became constant or anti-correlated
- **EEG loss > 5.0**: Model output very different from input
- **Gradient > 10.0**: Unstable training, need lower learning rate

### Scenario 2: Gradual loss increase
**Look for:**
- ⚠️  `Neurons not spiking (rate=0.0000)` - Dead neurons
- ⚠️  `X predictions have zero std` - Constant outputs
- ⚠️  `Clipped gradient norm: X.XX` - Gradients consistently high

**Likely cause:**
- Learning rate too high
- Temporal windowing breaking gradient flow
- Spiking loss forcing all neurons silent

### Scenario 3: NaN loss
**Look for this order:**
1. 🔴 `Model output contains NaN/Inf` - Forward pass broke
2. 🔴 `CorrelationLoss: pred contains NaN` - Loss computation broke
3. 🔴 `NaN/Inf gradient in <layer>` - Backward pass broke

**Immediate action:**
- Batch is skipped automatically
- Check which layer has NaN gradient
- Likely numerical instability in that layer

## What Log Messages Mean

### Good Training:
```
Loss components: EEG=0.45, Corr=0.95, Spiking=0.03, Reg=0.001, Total=1.43
```
- EEG loss: 0.3-0.6 (good reconstruction)
- Corr loss: 0.8-1.2 (decent correlation)
- Spiking: 0.01-0.05 (neurons spiking at ~10% rate)
- Total: 1.0-2.0 (healthy range)

### Warning Signs:
```
🔶 HIGH LOSS DETECTED: 8.52 | EEG=0.50, Corr=1.85, Spiking=0.03, Reg=0.001
```
- Correlation loss 1.85 is high (but not catastrophic)
- Model outputs are weakly correlated with inputs
- May improve with training

### Critical Failure:
```
🔴 TOTAL LOSS IS NaN/Inf!
   EEG loss: 0.4523
   Correlation loss: nan
   Simulated EEG range: [nan, nan]
```
- Correlation loss computation failed
- Model outputs contain NaN
- Training will skip this batch

## Quick Reference

| Loss Value | Status | Action |
|-----------|---------|--------|
| 1.0-2.0 | ✅ Normal | Continue training |
| 2.0-5.0 | ⚠️  High | Monitor, check components |
| 5.0-10.0 | 🔶 Very High | Investigate which component |
| > 10.0 | 🔴 Critical | Look for NaN/gradient explosion |

| Gradient Norm | Status | Action |
|--------------|---------|--------|
| < 1.0 | ✅ Normal | Good, no clipping |
| 1.0-5.0 | ⚠️  Moderate | Clipped but okay |
| 5.0-10.0 | 🔶 High | Frequent clipping, consider lower LR |
| > 10.0 | 🔴 Exploding | Reduce learning rate |

## What to Report

When you see a loss spike, capture:
1. **Batch number** where spike occurred
2. **Loss components** from 🔶 `HIGH LOSS DETECTED` message
3. **Gradient info** from 🔶 `Large gradient detected` message
4. **Any 🔴 error messages** before the spike
5. **Previous batch loss** (for comparison)

Example report:
```
Spike at batch 38:
- Previous: Total=1.65
- Spike: Total=28.76 | EEG=0.52, Corr=1.95, Spiking=0.04
- Gradient: 45.23 (clipped to 1.0)
- Error: "Large model output: max=15.23"
→ Model output exploded, causing correlation loss spike
```
