# Object detection evaluation metrics
## Current features
- [x] Confusion matrix
- [x] Precision
- [x] Recall
- [x] F1 Score
- [ ] mAP (COCO, Pascal voc etc.)

## How to use
**Prepare ground truth and prediction files**
- Each "image" should have separate text files.
- Use same names for both ground truth and predictions.
- Use separate folder for both ground truth and predictions.
- Each file should be in this format:
```xmin ymin xmax ymax label_id```

- Example (label id starts with 0):
```
1 2 3 4 0
1 2 3 4 1
1 2 3 4 1
```
Example code
```
>>> from pathlib import Path
>>> from odem import ObjectDetectionEval

>>> true_dir = Path(Path.cwd(), "examples", "true")
>>> pred_dir = Path(Path.cwd(), "examples", "pred")

>>> odem = ObjectDetectionEval(true_dir, pred_dir, labels=["cat", "dog"])
>>> odem.confusion_matrix()
>>> odem.classification_report()
        predictions
true     cat       dog       None      Total
cat       1         0         0         1
dog       0         1         3         4
None      2         0         0         2

Total     3         1         3
      precision     recall    f1-score
cat       1.00      0.33      0.50
dog       0.25      1.00      0.40
```

## Author
- Louis Philippe Facun
