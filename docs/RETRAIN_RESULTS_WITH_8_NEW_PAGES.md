# 📊 Retraining Results with 8 New TOC Pages

**Date**: February 21, 2026  
**Dataset**: 22 pages (13 TOC + 9 non-TOC) - was 14 pages  
**New TOC pages added**: 8 (acd_p006-009, itp_p010-013)

---

## ⚠️ CRITICAL ISSUE: Model Performance Degraded

### Comparison Results Summary

| Algorithm | Overall Accuracy | TOC Pages | Non-TOC Pages | Avg Time |
|-----------|-----------------|-----------|---------------|----------|
| **Original** | **94.4% (17/18)** | 88.9% (8/9) | **100% (9/9)** | 11.61s |
| **Fine-tuned (NEW)** | ❌ **50.0% (9/18)** | **100% (9/9)** | ❌ **0% (0/9)** | 4.01s |
| **Fine-tuned (OLD)** | 78.6% (11/14) | 40% (2/5) | 100% (9/9) | 4.44s |

### 🚨 Major Problem

The retrained model is **predicting everything as TOC**:
- ✅ TOC pages: 100% correct (9/9) - Perfect!
- ❌ Non-TOC pages: 0% correct (0/9) - **Complete failure!**
- Overall: 50% accuracy - **Worse than random guessing!**

---

## 📉 What Went Wrong?

### Training Results
- **Training samples**: 17 (was 11)
- **Validation samples**: 5 (was 3)
- **Best validation accuracy**: **60%** (was 66.67%)
- **Epochs completed**: 4 (early stopping)
- **Training accuracy**: 58.82% (final)

### Root Cause Analysis

1. **Class Imbalance in Training Set**
   - TOC pages: 13 samples (59%)
   - Non-TOC pages: 9 samples (41%)
   - The model learned to predict TOC for everything

2. **Overfitting to TOC Class**
   - With more TOC examples, model biased towards TOC
   - All predictions have ~0.56-0.60 confidence for TOC
   - Model doesn't distinguish between TOC and non-TOC

3. **Validation Accuracy Misleading**
   - 60% validation accuracy hid the problem
   - Validation set likely had similar class imbalance

4. **Lower Validation Accuracy Than Before**
   - Before: 66.67%
   - After: 60%
   - Adding more data made it worse!

---

## 📊 Detailed Test Results

### TOC Pages (9 tested)

| Page | Original | Fine-tuned | Improvement |
|------|----------|------------|-------------|
| mh_p005.png | ✅ Correct | ✅ Correct | - |
| hlw_p009.png | ✅ Correct | ✅ **Correct** (**was wrong**) | ✅ Fixed! |
| dlr_p006.png | ✅ Correct | ✅ Correct | - |
| its_p008.png | ✅ Correct | ✅ **Correct** (**was wrong**) | ✅ Fixed! |
| kf_p003.png | ❌ Wrong | ✅ **Correct** | ✅ Fixed! |
| itp_p010.png | ✅ Correct | ✅ Correct | - |
| itp_p011.png | ✅ Correct | ✅ Correct | - |
| itp_p012.png | ✅ Correct | ✅ Correct | - |
| itp_p013.png | ✅ Correct | ✅ Correct | - |

**Original**: 8/9 (88.9%)  
**Fine-tuned**: 9/9 (100%) ✅ **Perfect on TOC!**

### Non-TOC Pages (9 tested)

| Page | Original | Fine-tuned (OLD) | Fine-tuned (NEW) |
|------|----------|-----------------|------------------|
| dvurog_p017.png | ✅ Correct | ✅ Correct | ❌ **Wrong** (predicts TOC) |
| dvurog_p019.png | ✅ Correct | ✅ Correct | ❌ **Wrong** (predicts TOC) |
| dvurog_p076.png | ✅ Correct | ✅ Correct | ❌ **Wrong** (predicts TOC) |
| sedg_p598.png | ✅ Correct | ✅ Correct | ❌ **Wrong** (predicts TOC) |
| jtg_p033.png | ✅ Correct | ✅ Correct | ❌ **Wrong** (predicts TOC) |
| hlw_p040.png | ✅ Correct | ✅ Correct | ❌ **Wrong** (predicts TOC) |
| mh_p013.png | ✅ Correct | ✅ Correct | ❌ **Wrong** (predicts TOC) |
| kf_p015.png | ✅ Correct | ✅ Correct | ❌ **Wrong** (predicts TOC) |
| kf_p016.png | ✅ Correct | ✅ Correct | ❌ **Wrong** (predicts TOC) |

**Original**: 9/9 (100%) ✅  
**Fine-tuned (OLD)**: 9/9 (100%) ✅  
**Fine-tuned (NEW)**: 0/9 (0%) ❌ **Complete failure!**

---

## 🔍 Comparison: Before vs After Retraining

### Before (14 samples, 5 TOC + 9 non-TOC)
- Overall: 78.6% (11/14)
- TOC: 40% (2/5)
- Non-TOC: 100% (9/9) ✅
- **Balanced predictions**

### After (22 samples, 13 TOC + 9 non-TOC)
- Overall: 50% (9/18) ❌ **Worse!**
- TOC: 100% (9/9) ✅
- Non-TOC: 0% (0/9) ❌
- **Completely unbalanced - predicts everything as TOC**

### What Happened
- Adding 8 TOC examples increased TOC ratio from 36% to 59%
- Model learned "when in doubt, predict TOC"
- Lost ability to distinguish non-TOC pages

---

## 💡 Solutions to Fix This

### Solution 1: Balance the Dataset ⭐ **Best**

**Problem**: 13 TOC vs 9 non-TOC (59% vs 41%)

**Fix**: Add more non-TOC pages

```
Target: 50/50 balance
Current: 13 TOC + 9 non-TOC
Add: 4 more non-TOC pages
Result: 13 TOC + 13 non-TOC (50/50)
```

**Expected**: 75-85% accuracy, balanced predictions

### Solution 2: Use Class Weights

Add to training code:

```python
# In train_model function
class_weights = torch.tensor([1.44, 1.0])  # [non-TOC weight, TOC weight]
# 1.44 = 13/9 (ratio of TOC to non-TOC)

# In model definition
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    config.model_name,
    num_labels=2,
    loss_fn=nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
)
```

**Expected**: Better balance, but may reduce TOC accuracy

### Solution 3: Data Augmentation

For TOC pages:
- Slightly different page number formats
- Different fonts/sizes
- Rotations, crops

For non-TOC pages:
- Add more diverse non-TOC examples

### Solution 4: Adjust Decision Threshold

Instead of 0.5, use 0.6 or 0.7:

```python
# Instead of:
is_toc = prediction == 1

# Use:
is_toc = probs[1] > 0.6  # Higher threshold for TOC
```

**Current predictions**: All ~0.56-0.60 confidence
**With threshold 0.6**: Would filter out some false TOCs

---

## 📝 Recommendation

### Immediate Action: Use Original Algorithm ✅

**Current Best**: Original algorithm (94.4% accuracy)
- Perfect on non-TOC: 100%
- Excellent on TOC: 88.9%
- Production ready

**Command**:
```bash
python src/ocr_reflow/main.py IMAGE --layout --toc-algorithm original
```

### To Fix Fine-Tuned Model

**Step 1**: Balance dataset
```bash
# Add 4-5 more non-TOC pages to images/
# Update KNOWN_LABELS in train_layoutlmv3.py
# Target: 13 TOC + 13 non-TOC
```

**Step 2**: Implement class weights
```python
# Edit train_layoutlmv3.py
# Add class_weights as shown in Solution 2
```

**Step 3**: Retrain
```bash
python train_layoutlmv3.py
```

**Expected after fix**: 80-85% overall accuracy with balanced predictions

---

## 🎓 Lessons Learned

### ❌ What Didn't Work
1. **Simply adding more data of one class** doesn't help
2. **Class imbalance** (59% vs 41%) caused severe bias
3. **Validation accuracy** alone is misleading
4. **More data ≠ better model** without balance

### ✅ What to Do Instead
1. **Balance classes**: Aim for 50/50 or 60/40 max
2. **Use class weights**: Compensate for imbalance
3. **Monitor per-class metrics**: Not just overall accuracy
4. **Test on held-out set**: Validation may not represent reality

### 📊 Key Metrics to Watch
- Overall accuracy: 94.4% → 50% ❌
- TOC accuracy: 40% → 100% (but useless if everything is TOC)
- Non-TOC accuracy: 100% → 0% ❌ **Critical failure**
- **F1 Score** would have caught this problem earlier

---

## 📈 Performance Comparison

### Accuracy Chart

```
Original Algorithm:     ████████████████████ 94.4%
Fine-tuned (OLD):       ███████████████░░░░░ 78.6%
Fine-tuned (NEW):       ██████████░░░░░░░░░░ 50.0% ❌
```

### By Class

**TOC Pages**:
```
Original:        ████████████████████░ 88.9%
Fine-tuned (OLD): ████████░░░░░░░░░░░░ 40.0%
Fine-tuned (NEW): ████████████████████ 100.0% ✅
```

**Non-TOC Pages**:
```
Original:        ████████████████████ 100.0%
Fine-tuned (OLD): ████████████████████ 100.0%
Fine-tuned (NEW): ░░░░░░░░░░░░░░░░░░░░ 0.0% ❌
```

---

## 🔬 Technical Analysis

### Model Behavior

**Old Model (14 samples)**:
- Confidence ranges: 0.50-0.55
- Conservative predictions
- Slightly biased to non-TOC
- **Balanced**

**New Model (22 samples)**:
- Confidence ranges: 0.55-0.60
- Aggressive TOC predictions
- Heavily biased to TOC
- **Unbalanced**

### Training Metrics

| Metric | Old Model | New Model | Change |
|--------|-----------|-----------|--------|
| Training samples | 11 | 17 | +55% |
| Val samples | 3 | 5 | +67% |
| Val accuracy | 66.67% | 60.0% | -6.67% |
| Epochs | 6 | 4 | -2 |
| Train accuracy | 81.82% | 58.82% | -23% |

**Observation**: All metrics got worse except dataset size

---

## ⚡ Speed Comparison

| Algorithm | Avg Time | Improvement |
|-----------|----------|-------------|
| Original | 11.61s | Baseline |
| Fine-tuned (OLD) | 4.44s | 2.6x faster |
| Fine-tuned (NEW) | 4.01s | 2.9x faster |

**Speed is not the issue** - accuracy is!

---

## 🎯 Final Verdict

### Current Rankings

1. **🥇 Original Algorithm**: 94.4% accuracy ✅ **Best**
2. **🥈 Fine-tuned (OLD)**: 78.6% accuracy
3. **🥉 Fine-tuned (NEW)**: 50.0% accuracy ❌ **Unusable**

### Why Retraining Failed

1. **Class imbalance** (13 TOC vs 9 non-TOC)
2. **Model learned to always predict TOC**
3. **No class weighting or balancing**
4. **Validation metric didn't catch the issue**

### What to Do

✅ **USE**: Original algorithm (94.4%)  
❌ **DON'T USE**: New fine-tuned model (50%)  
⚠️ **CONSIDER**: Old fine-tuned model (78.6%) if speed matters

**To fix**: Balance dataset to 13 TOC + 13 non-TOC and retrain

---

## 📄 Files Generated

1. ✅ `training_log_retrain.txt` - Training log with 22 samples
2. ✅ `comparison_retrain.txt` - Comparison results
3. ✅ `dataset/toc_dataset.json` - Dataset with 22 samples
4. ✅ `models/layoutlmv3_toc/best_model/` - Retrained model (not recommended)

---

**Test Date**: February 21, 2026  
**Dataset**: 22 pages (13 TOC + 9 non-TOC)  
**Result**: ❌ **Retraining made model worse**  
**Recommendation**: ✅ **Continue using Original algorithm (94.4%)**
