from shapely.geometry import box
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from sklearn.metrics import auc
def iou(a, b):
    a = np.array(a)
    b = np.array(b)
    
    x1 = a[0] - a[2]/2
    y1 = a[1] - a[3]/2
    X1 = a[0] + a[2]/2
    Y1 = a[1] + a[3]/2
    
    x2 = b[0] - b[2]/2
    y2 = b[1] - b[3]/2
    X2 = b[0] + b[2]/2
    Y2 = b[1] + b[3]/2
    
    b1 = box(x1, y1, X1, Y1)
    b2 = box(x2, y2, X2, Y2)
    
    i = b1.intersection(b2).area
    u = b1.union(b2).area
    
    return i/u if u > 0 else 0

def AP(prec, rec, method='auc'):
    if method == 'voc11':
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11
        return ap
    elif method == 'coco101':
        ap = 0
        for t in np.arange(0, 1.01, 0.01):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 101
        return ap
    else:
        idx = np.argsort(rec)
        rec = rec[idx]
        prec = prec[idx]
        rec = np.concatenate(([0.], rec, [1.]))
        prec = np.concatenate(([0.], prec, [0.]))
        for i in range(prec.size - 1, 0, -1):
            prec[i - 1] = np.maximum(prec[i - 1], prec[i])
        unique_rec = np.where(rec[1:] != rec[:-1])[0]
        rec = rec[np.r_[unique_rec, rec.size-1]]
        prec = prec[np.r_[unique_rec, prec.size-1]]
            
        return auc(rec, prec)
 
def eval(true_boxes, pred_boxes, scores, iou_thresh=0.5):
    num_true = len(true_boxes)
    num_pred = len(pred_boxes)
    
    if num_true == 0 or num_pred == 0:
        return np.array([0.0]), np.array([0.0])

    idx = np.argsort(scores)[::-1]
    pred_boxes = pred_boxes[idx]
    scores = scores[idx]
    
    true_matched = np.zeros(num_true, dtype=bool)
    true_pos = np.zeros(num_pred)
    false_pos = np.zeros(num_pred)
    
    for i in range(num_pred):
        best_iou = 0
        best_match = -1
        
        for j in range(num_true):
            if true_matched[j]:
                continue
            curr_iou = iou(pred_boxes[i], true_boxes[j])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_match = j
        if best_iou >= iou_thresh:
            true_pos[i] = 1
            true_matched[best_match] = True
        else:
            false_pos[i] = 1
    
    fp_sum = np.cumsum(false_pos)
    tp_sum = np.cumsum(true_pos)
    recalls = tp_sum / (num_true + 1e-10)
    precisions = tp_sum / (tp_sum + fp_sum + 1e-10)
    
    if len(precisions) < 2:
        precisions = np.array([1.0, 0.0])
        recalls = np.array([0.0, 1.0])
    
    return precisions, recalls

def plot(image_path, label_path, predictions=None):
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])  
        image = np.moveaxis(image, 0, -1)
    img_height, img_width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    
    gt = plt.Rectangle((0,0), 1, 1, 
                           edgecolor='#00FF00',  
                           facecolor='none', 
                           linewidth=2,
                           label='Ground Truth Boxes')
    pred= plt.Rectangle((0,0), 1, 1, 
                             edgecolor='#FF1493',
                             facecolor='none', 
                             linewidth=2,
                             label='Predicted Boxes')
    ax.add_patch(gt)
    ax.add_patch(pred)
    
    if Path(label_path).exists():
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            data = line.strip().split()
            cls, x_center, y_center, width, height = map(float, data)
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            
            rect = plt.Rectangle((x_min, y_min), width, height, 
                               edgecolor='#00FF00', 
                               facecolor='none', 
                               linewidth=2)
            ax.add_patch(rect)
    
    if predictions is not None:
        pred_boxes = predictions.boxes.xywh.cpu().numpy()
        pred_scores = predictions.boxes.conf.cpu().numpy()
        
        for box, score in zip(pred_boxes, pred_scores):
            x_center, y_center, width, height = box
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            
            rect = plt.Rectangle((x_min, y_min), width, height, 
                               edgecolor='#FF1493',  
                               facecolor='none', 
                               linewidth=2)
            ax.add_patch(rect)
            ax.text(x_min, y_min-5, f'{score:.2f}', 
                   color='#FF1493', 
                   fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.legend(loc='upper right')
    ax.set_title('Solar Panel Detection')
    
    return fig, ax