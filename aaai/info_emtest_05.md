# EM_Iteration - emtest_05 Info

### testing confusion output - Thu at 12:05:20AM

---

## Parameters:

### Iteration:
 - EM_target (int): `2`
 - lr_const (int): `2`
 - rebuild_model (bool): `True`
 - random_candidate_chance (float): `0.05`
 - weight_limit (int): `13`

### Environment:
 - tf_seed (int): `2001`
 - np_seed (int): `2001`
 - py_seed (int): `2001`

### Tensor Data:
 - WinShape (tuple): `(224, 224)`
 - labelBuffer (int): `2`
 - train_h_flip (bool): `True`
 - train_v_flip (bool): `True`
 - train_rotate (bool): `True`
 - val_h_flip (bool): `True`
 - val_v_flip (bool): `True`
 - val_rotate (bool): `True`

### Annotation Configuration:
 - maxRepairDist (int): `20`
 - weightBuffer (int): `2`
 - precisionBuffer (float): `8e-05`

### UNet Config:
 - learningRate (float): `0.1`
 - metrics (list): `[<function dice_coef at 0x7f97b2ebe0e0>, 'accuracy', <function f1_score at 0x7f989a8d9050>]`
 - batchSize (int): `32`
 - useShuffle (bool): `True`
 - epochs (int): `50`
 - dropval (float): `0.2`

### Baseline Directories:
 - CandidateDirectory (str): `./NVME_EM/segments`
 - preTrainedPath (str): `/data/GeometricErrors/EM/preweights.h5`
 - train_offsets_fp (str): `/data/GeometricErrors/EM/train_offsets.csv`
 - val_offsets_fp (str): `/data/GeometricErrors/EM/val_offsets.csv`

---

## Inputs:

### Input Arrays:
 - X_train (ndarray):
   `/data/GeometricErrors/EM/X_train.npy`
 - Y_train (ndarray):
   `/data/GeometricErrors/EM/Y_train.npy`
 - X_val (ndarray):
   `/data/GeometricErrors/EM/X_val.npy`
 - Y_val (ndarray):
   `/data/GeometricErrors/EM/Y_val.npy`
 - X_test (ndarray):
   `/data/GeometricErrors/EM/X_test.npy`
 - Y_test (ndarray):
   `/data/GeometricErrors/EM/Y_test.npy`

### Source_Data:
 - GroundTruth (GeoDataFrame):
   `/data/GeometricErrors/CompleteScene/GroundTruth.shp`
 - imperfectLines (GeoDataFrame):
   `/data/GeometricErrors/CompleteScene/imperfectLines.shp`
 - testRaster (DatasetReader):
   `/data/GeometricErrors/CompleteScene/testRaster.tif`
 - trainRaster (DatasetReader):
   `/data/GeometricErrors/CompleteScene/trainRaster.tif`
 - initial_pmap (DatasetReader):
   `/data/GeometricErrors/EM/initial_pmap.tif`

---

## Results:

### Baseline Data:
 - Precision (str): `27.39%`
 - UNet: Training Results (list): `['Dice Coef: 41.589%', 'F1 Score: 41.590%']`
 - UNet: Validation Results (list): `['Dice Coef: 34.709%', 'F1 Score: 34.704%']`
 - UNet: Testing Results (list): `['Dice Coef: 46.384%', 'F1 Score: 46.390%']`
 - UNet: Report (str): `              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99   9897977
         1.0       0.39      0.57      0.46    137223

    accuracy                           0.98  10035200
   macro avg       0.69      0.78      0.73  10035200
weighted avg       0.99      0.98      0.98  10035200
`
 - UNet: Confusion Matrix (ndarray): `[[9774361  123616]
 [  58450   78773]]`

### EM Data 00:
 - Time Elapsed: Re-training (float): `333.301146030426`
 - Time Elapsed: New Annotation (float): `73.1458592414856`
 - Precision (str): `22.45%`
 - SourceDelta (str): `-4.94%`
 - StepDelta (str): `-4.94%`
 - UNet: LR (str): `0.1`
 - UNet: Epochs (str): `27`
 - UNet: Training Results (list): `['Dice Coef: 22.669%', 'F1 Score: 22.670%']`
 - UNet: Validation Results (list): `['Dice Coef: 17.278%', 'F1 Score: 17.275%']`
 - UNet: Testing Results (list): `['Dice Coef: 33.541%', 'F1 Score: 33.542%']`
 - UNet: Confusion Matrix (ndarray): `[[9750497  147480]
 [  79854   57369]]`
 - UNet: Report (str): `              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99   9897977
         1.0       0.28      0.42      0.34    137223

    accuracy                           0.98  10035200
   macro avg       0.64      0.70      0.66  10035200
weighted avg       0.98      0.98      0.98  10035200
`

### EM Data 01:
 - Time Elapsed: Re-training (float): `378.9962160587311`
 - Time Elapsed: New Annotation (float): `71.54188823699951`
 - Precision (str): `41.87%`
 - SourceDelta (str): `14.48%`
 - StepDelta (str): `19.42%`
 - UNet: LR (str): `0.05`
 - UNet: Epochs (str): `31`
 - UNet: Training Results (list): `['Dice Coef: 43.966%', 'F1 Score: 43.966%']`
 - UNet: Validation Results (list): `['Dice Coef: 38.752%', 'F1 Score: 38.753%']`
 - UNet: Testing Results (list): `['Dice Coef: 45.570%', 'F1 Score: 45.572%']`
 - UNet: Confusion Matrix (ndarray): `[[9801632   96345]
 [  68297   68926]]`
 - UNet: Report (str): `              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99   9897977
         1.0       0.42      0.50      0.46    137223

    accuracy                           0.98  10035200
   macro avg       0.71      0.75      0.72  10035200
weighted avg       0.99      0.98      0.98  10035200
`

### EM Data 02:
 - Time Elapsed: Re-training (float): `599.3277382850647`
 - Time Elapsed: New Annotation (float): `67.14074397087097`
 - Precision (str): `46.53%`
 - SourceDelta (str): `19.14%`
 - StepDelta (str): `4.65%`
 - UNet: LR (str): `0.1`
 - UNet: Epochs (str): `50`
 - UNet: Training Results (list): `['Dice Coef: 60.272%', 'F1 Score: 60.274%']`
 - UNet: Validation Results (list): `['Dice Coef: 51.515%', 'F1 Score: 51.516%']`
 - UNet: Testing Results (list): `['Dice Coef: 58.210%', 'F1 Score: 58.212%']`
 - UNet: Confusion Matrix (ndarray): `[[9838755   59222]
 [  56571   80652]]`
 - UNet: Report (str): `              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99   9897977
         1.0       0.58      0.59      0.58    137223

    accuracy                           0.99  10035200
   macro avg       0.79      0.79      0.79  10035200
weighted avg       0.99      0.99      0.99  10035200
`

### EM Data 03:
 - Time Elapsed: Re-training (float): `478.6629765033722`
 - Time Elapsed: New Annotation (float): `65.41721940040588`
 - Precision (str): `43.20%`
 - SourceDelta (str): `15.81%`
 - StepDelta (str): `-3.33%`
 - UNet: LR (str): `0.05`
 - UNet: Epochs (str): `40`
 - UNet: Training Results (list): `['Dice Coef: 65.314%', 'F1 Score: 65.319%']`
 - UNet: Validation Results (list): `['Dice Coef: 53.285%', 'F1 Score: 53.289%']`
 - UNet: Testing Results (list): `['Dice Coef: 60.524%', 'F1 Score: 60.531%']`
 - UNet: Confusion Matrix (ndarray): `[[9848137   49840]
 [  56035   81188]]`
 - UNet: Report (str): `              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99   9897977
         1.0       0.62      0.59      0.61    137223

    accuracy                           0.99  10035200
   macro avg       0.81      0.79      0.80  10035200
weighted avg       0.99      0.99      0.99  10035200
`

### EM Data 04:
 - Time Elapsed: Re-training (float): `429.16247844696045`
 - Time Elapsed: New Annotation (float): `66.71263813972473`
 - Precision (str): `47.64%`
 - SourceDelta (str): `20.25%`
 - StepDelta (str): `4.44%`
 - UNet: LR (str): `0.1`
 - UNet: Epochs (str): `36`
 - UNet: Training Results (list): `['Dice Coef: 65.613%', 'F1 Score: 65.615%']`
 - UNet: Validation Results (list): `['Dice Coef: 57.193%', 'F1 Score: 57.201%']`
 - UNet: Testing Results (list): `['Dice Coef: 64.642%', 'F1 Score: 64.643%']`
 - UNet: Confusion Matrix (ndarray): `[[9845249   52728]
 [  46507   90716]]`
 - UNet: Report (str): `              precision    recall  f1-score   support

         0.0       1.00      0.99      0.99   9897977
         1.0       0.63      0.66      0.65    137223

    accuracy                           0.99  10035200
   macro avg       0.81      0.83      0.82  10035200
weighted avg       0.99      0.99      0.99  10035200
`

### EM Data 05:
 - Time Elapsed: Re-training (float): `583.8132796287537`
 - Time Elapsed: New Annotation (float): `65.54798197746277`
 - Precision (str): `47.35%`
 - SourceDelta (str): `19.96%`
 - StepDelta (str): `-0.29%`
 - UNet: LR (str): `0.05`
 - UNet: Epochs (str): `49`
 - UNet: Training Results (list): `['Dice Coef: 67.804%', 'F1 Score: 67.803%']`
 - UNet: Validation Results (list): `['Dice Coef: 59.464%', 'F1 Score: 59.473%']`
 - UNet: Testing Results (list): `['Dice Coef: 65.781%', 'F1 Score: 65.786%']`
 - UNet: Confusion Matrix (ndarray): `[[9852020   45957]
 [  47436   89787]]`
 - UNet: Report (str): `              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00   9897977
         1.0       0.66      0.65      0.66    137223

    accuracy                           0.99  10035200
   macro avg       0.83      0.82      0.83  10035200
weighted avg       0.99      0.99      0.99  10035200
`

---

## Other Test Data:
 - Ending: `'_05'`
 - Dir: `./Modules/EM_Iteration/emtest_05`
 - **Section 00**: `Prepare Base Data`
    - Time: `156.003 sec`
 - **Section 01**: `Prepare Model`
    - Time: `73.778 sec`
 - **Section 02**: `EM_Step 00`
    - Time: `479.124 sec`
 - **Section 03**: `EM_Step 01`
    - Time: `525.259 sec`
 - **Section 04**: `EM_Step 02`
    - Time: `740.497 sec`
 - **Section 05**: `EM_Step 03`
    - Time: `617.755 sec`
 - **Section 06**: `EM_Step 04`
    - Time: `571.332 sec`
 - **Section 07**: `EM_Step 05`
    - Time: `721.949 sec`