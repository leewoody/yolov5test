# Three Image-Label Examples from Different Classification Classes

This document shows three examples of images and their corresponding labels from the Regurgitation-YOLODataset, each representing a different classification class using one-hot encoding.

## Dataset Structure
The dataset uses YOLO format with one-hot encoded classification labels representing different echocardiographic views:
- **Class 0**: `0 0 1` - Parasternal Short-Axis View (PSAX)
- **Class 1**: `0 1 0` - Parasternal Long-Axis View (PLAX) 
- **Class 2**: `1 0 0` - Apical Four-Chamber View (A4C)

## Detection Classes
The YOLO detection classes represent different types of regurgitation:
- **AR**: Aortic Regurgitation
- **MR**: Mitral Regurgitation  
- **PR**: Pulmonary Regurgitation
- **TR**: Tricuspid Regurgitation

## Example 1: Class 0 - Parasternal Short-Axis View (PSAX) (0 0 1)

**Label File**: `Regurgitation-YOLODataset-1new/train/labels/a2ZnwqdsaMKZ-unnamed_1_1.mp4-0.txt`

**Image File**: `Regurgitation-YOLODataset-1new/train/images/a2ZnwqdsaMKZ-unnamed_1_1.mp4-0.png`

**Label Content**:
```
0 0.42693274670018855 0.4429898648648649 0.6257071024512885 0.4429898648648649 0.6257071024512885 0.6488597972972973 0.42693274670018855 0.6488597972972973

0 0 1
```

**Description**: 
- YOLO bounding box: class 0 (AR - Aortic Regurgitation) with coordinates for a rectangular region
- Classification: Parasternal Short-Axis View (PSAX) - one-hot encoded as `0 0 1`

## Example 2: Class 1 - Parasternal Long-Axis View (PLAX) (0 1 0)

**Label File**: `Regurgitation-YOLODataset-1new/train/labels/a2lrwqduZsKc-unnamed_1_2.mp4-2.txt`

**Image File**: `Regurgitation-YOLODataset-1new/train/images/a2lrwqduZsKc-unnamed_1_2.mp4-2.png`

**Label Content**:
```
3 0.3860779384035198 0.35219594594594594 0.5306411062225016 0.35219594594594594 0.5306411062225016 0.6404138513513514 0.3860779384035198 0.6404138513513514

0 1 0
```

**Description**: 
- YOLO bounding box: class 3 (TR - Tricuspid Regurgitation) with coordinates for a rectangular region
- Classification: Parasternal Long-Axis View (PLAX) - one-hot encoded as `0 1 0`

## Example 3: Class 2 - Apical Four-Chamber View (A4C) (1 0 0)

**Label File**: `Regurgitation-YOLODataset-1new/train/labels/a2lrwqduZsKc-unnamed_1_2.mp4-16.txt`

**Image File**: `Regurgitation-YOLODataset-1new/train/images/a2lrwqduZsKc-unnamed_1_2.mp4-16.png`

**Label Content**:
```
3 0.3467944688874921 0.49260979729729726 0.5141420490257699 0.49260979729729726 0.5141420490257699 0.7206503378378379 0.3467944688874921 0.7206503378378379

1 0 0
```

**Description**: 
- YOLO bounding box: class 3 (TR - Tricuspid Regurgitation) with coordinates for a rectangular region
- Classification: Apical Four-Chamber View (A4C) - one-hot encoded as `1 0 0`

## Label Format Explanation

Each label file contains:
1. **First line**: YOLO format bounding box annotation
   - Format: `class_id center_x center_y width height x1 y1 x2 y2 x3 y3 x4 y4`
   - Where `class_id` is the detection class (0=AR, 1=MR, 2=PR, 3=TR)
   - Coordinates are normalized between 0 and 1
   - The last 8 values represent polygon coordinates (x1,y1,x2,y2,x3,y3,x4,y4)

2. **Second line**: One-hot encoded classification label
   - Format: `PSAX PLAX A4C`
   - Only one value is 1, others are 0
   - Indicates which echocardiographic view the image represents

## Summary

These three examples demonstrate the complete dataset structure with:
- **Object detection**: YOLO bounding boxes for specific regurgitation types (AR, MR, PR, TR)
- **Classification**: One-hot encoded labels indicating the echocardiographic view (PSAX, PLAX, A4C)
- **Multi-task learning**: Each image has both detection and classification annotations 