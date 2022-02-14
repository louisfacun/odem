from pathlib import Path
from odem import ObjectDetectionEval

true_dir = Path(Path.cwd(), "examples", "true")
pred_dir = Path(Path.cwd(), "examples", "pred")

odem = ObjectDetectionEval(true_dir, pred_dir, labels=["cat", "dog"])
odem.confusion_matrix()
odem.classification_report()