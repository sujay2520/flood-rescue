import rasterio
import numpy as np
import glob
import os
from scipy.ndimage import median_filter, binary_opening

dest = 'data/ground_truth/sen1floods11_pakistan'
s1_files = sorted(glob.glob(os.path.join(dest, '*_S1Hand.tif')))

print("=" * 115)
print("CHECK 1: EXACT FP / PRECISION BREAKDOWN ACROSS ALL 11 SEN1FLOODS11 PAKISTAN CHIPS")
print("=" * 115)
print(f"{'Chip Identifier':<20} | {'Valid Px':>8} | {'GT Water':>8} | {'Pred Px':>8} | {'TP':>8} | {'FP':>8} | {'FN':>8} | {'Prec %':>7} | {'Rec %':>7} | {'IoU %':>7} | {'F1 %':>7}")
print("-" * 115)

total_tp = total_fp = total_fn = total_tn = 0
chip_stats = []

for s1_path in s1_files:
    chip_id = os.path.basename(s1_path).replace('_S1Hand.tif', '')
    lbl_path = os.path.join(dest, chip_id + '_LabelHand.tif')
    if not os.path.exists(lbl_path):
        continue
    with rasterio.open(s1_path) as src:
        vv = src.read(1).astype(np.float32)
    with rasterio.open(lbl_path) as src:
        lbl = src.read(1)
    
    valid = ((lbl == 0) | (lbl == 1)) & np.isfinite(vv)
    gt = (lbl == 1) & valid
    
    vpx = vv[valid]
    if len(vpx) < 50:
        continue
        
    counts, edges = np.histogram(vpx, bins=256, range=(float(np.nanmin(vpx)), float(np.nanmax(vpx))))
    centers = (edges[:-1] + edges[1:]) / 2.0
    w1 = np.cumsum(counts).astype(float)
    w2 = counts.sum() - w1
    m1 = np.cumsum(counts * centers) / (w1 + 1e-9)
    m2_cum = np.cumsum((counts * centers)[::-1])[::-1]
    m2 = m2_cum / (w2 + 1e-9)
    var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    otsu_th = float(centers[np.argmax(var)])
    effective_th = float(np.clip(otsu_th, -22.0, -11.0))
    
    cleaned_vv = np.nan_to_num(vv, nan=0.0)
    filt = median_filter(cleaned_vv, size=3)
    pred = binary_opening(filt < effective_th, structure=np.ones((2,2))) & valid
    
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & (~gt)))
    fn = int(np.sum((~pred) & gt))
    tn = int(np.sum((~pred) & (~gt)))
    
    total_tp += tp
    total_fp += fp
    total_fn += fn
    total_tn += tn
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    chip_stats.append({
        'chip': chip_id, 'valid': valid.sum(), 'gt_water': gt.sum(), 'pred': pred.sum(),
        'tp': tp, 'fp': fp, 'fn': fn, 'prec': prec, 'rec': rec, 'iou': iou, 'f1': f1, 'th': effective_th
    })
    
    print(f"{chip_id:<20} | {valid.sum():>8,} | {gt.sum():>8,} | {pred.sum():>8,} | {tp:>8,} | {fp:>8,} | {fn:>8,} | {prec*100:>6.2f}% | {rec*100:>6.2f}% | {iou*100:>6.2f}% | {f1*100:>6.2f}%")

print("-" * 115)
pooled_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
pooled_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
pooled_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
pooled_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0

print(f"{'POOLED 11 CHIPS':<20} | {total_tp+total_fp+total_fn+total_tn:>8,} | {total_tp+total_fn:>8,} | {total_tp+total_fp:>8,} | {total_tp:>8,} | {total_fp:>8,} | {total_fn:>8,} | {pooled_prec*100:>6.2f}% | {pooled_rec*100:>6.2f}% | {pooled_iou*100:>6.2f}% | {pooled_f1*100:>6.2f}%")
print("=" * 115)
