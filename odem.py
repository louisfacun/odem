import numpy as np
import glob
from pathlib import Path

class ObjectDetectionEval:
    def __init__(self, true_dir, pred_dir, labels, iou_thresh=0.5):
        self._label_count = len(labels)
        self._labels = labels
        self._confusion_matrix = np.zeros((self._label_count+1, self._label_count+1), dtype=int)
        self._iou_thresh = iou_thresh

        # A list of images' bboxes
        self._multi_bboxes_true = []
        self._multi_bboxes_pred = []

        # browse true txt files
        for file_path in glob.iglob(f"{true_dir}/*.txt"):
            bboxes_true = []
            bboxes_pred = []
            
            # parse true txt files
            with open(file_path) as file:
                for line in file:
                    xmin, ymin, xmax, ymax, label_id = line.split() 
                    bboxes_true.append([float(xmin), float(ymin), float(xmax), float(ymax), int(label_id)])
            file_name = Path(file_path).name

            # TODO: check pred file first if it exist
            # load pred txt file based on the file name
            with open(Path(pred_dir, file_name)) as file:
                for line in file:
                    xmin, ymin, xmax, ymax, label_id = line.split() 
                    bboxes_pred.append([float(xmin), float(ymin), float(xmax), float(ymax), int(label_id)])
            
            self._multi_bboxes_true.append(bboxes_true)       
            self._multi_bboxes_pred.append(bboxes_pred)


    # For single image
    def confusion_matrix_single(self, bboxes_true, bboxes_pred):
        cf = np.zeros((self._label_count+1, self._label_count+1), dtype=int)
        
        bboxes_true = np.array(bboxes_true)
        bboxes_pred = np.array(bboxes_pred)

        # Keep track of already matched pred and true
        pred_del = []
        true_del = []

        # Find True Positives (TP) and remove those on bbox list
        for tid, bbox_true in enumerate(bboxes_true):
            for pid, bbox_pred in enumerate(bboxes_pred):
                # only checks not yet matched
                if tid not in true_del and pid not in pred_del:
                    if ObjectDetectionEval.get_iou(bbox_true[:4], bbox_pred[:4]) >= self._iou_thresh:
                        true_id = int(bbox_true[4])
                        pred_id = int(bbox_pred[4])
                        if true_id == pred_id:  # tp
                            cf[true_id, pred_id] += 1
                            true_del.append(tid)
                            pred_del.append(pid)

        bboxes_true = np.delete(bboxes_true, true_del, axis=0)
        bboxes_pred = np.delete(bboxes_pred, pred_del, axis=0)
        pred_del = []
        true_del = []

        # Find False Positives (FP) and remove those on bbox list
        for tid, bbox_true in enumerate(bboxes_true):
            for pid, bbox_pred in enumerate(bboxes_pred):
                # only checks not yet matched
                if tid not in true_del and pid not in pred_del:
                    if ObjectDetectionEval.get_iou(bbox_true[:4], bbox_pred[:4]) >= self._iou_thresh:
                        true_id = int(bbox_true[4])
                        pred_id = int(bbox_pred[4])
                        if true_id != pred_id: # fp  
                            cf[true_id, pred_id] += 1
                            true_del.append(tid)
                            pred_del.append(pid)     

        bboxes_true = np.delete(bboxes_true, true_del, axis=0)
        bboxes_pred = np.delete(bboxes_pred, pred_del, axis=0)
        pred_del = []
        true_del = []

        # What will be left is predictions that does not match the IoU thresh
        # So we assign these predictions in "None" (row)    
        for pid, pred_bbox in enumerate(bboxes_pred): 
            pred_id = int(bbox_pred[4])
            cf[self._label_count, pred_id] += 1
        
        # And these are actual bboxes that is not predicted: whether it's true positive or false positive(with class)
        # Also assign in "None" (column)
        for aid, actual_bbox in enumerate(bboxes_true):
            true_id = int(bbox_true[4])
            cf[true_id, self._label_count] += 1

        return cf


    def confusion_matrix(self):
        for bboxes_true, bboxes_pred in zip(self._multi_bboxes_true, self._multi_bboxes_pred):
            self._confusion_matrix += self.confusion_matrix_single(bboxes_true, bboxes_pred)

        # Header
        labels = [l for l in self._labels]
        labels.append("None")
        labels.append("Total")
        #labels = label_top
        #label_top.insert(0, "TRUE")
        print("        predictions")
        print("true     ", end="")    
        for l in labels:
            print(f"{l:<10}", end="")
        print()

        # Left labels + Confusion matrix
        for i, row in enumerate(self._confusion_matrix):
            print(f"{labels[i]:<10}", end="")
            total_row = np.sum(row)
            for col in row:
                print(f"{col:<10}", end="")
            print(f"{total_row:<10}")

        # Total col
        print("\nTotal     ", end="")
        total_col = np.sum(self._confusion_matrix, axis=0)
        for col in total_col:
                print(f"{col:<10}", end="")
        print()


    def classification_report(self):
        tp = np.zeros((self._label_count,), dtype=int)
        fp = np.zeros((self._label_count,), dtype=int)
        fn = np.zeros((self._label_count,), dtype=int)

        precisions = np.zeros((self._label_count,), dtype=float)
        recalls = np.zeros((self._label_count,), dtype=float)
        f1_scores = np.zeros((self._label_count,), dtype=float)

        # get tp
        for i in range(self._label_count):
            tp[i] = self._confusion_matrix[i, i]

        # calc fp
        for i in range(self._label_count):
            for j in range(self._label_count+1):
                fp[i] += self._confusion_matrix[j, i]
            fp[i] -= tp[i] 

        # calc fn
        for i in range(self._label_count):
            for j in range(self._label_count+1):
                fn[i] += self._confusion_matrix[i, j]
            fn[i] -= tp[i] 

        # calc precision and recall, f1 score
        for i, (tpval, fpval, fnval) in enumerate(zip(tp, fp, fn)):
            if i != self._label_count: # we dont calculate precision and recall for `none` row and column
                precisions[i] = tpval / (tpval + fnval)
                recalls[i] = tpval / (tpval + fpval)
                f1_scores[i] = 2 * tpval / ((2 * tpval) + fpval + fnval)

        print(f"      precision     recall    f1-score")
        for i in range(self._label_count):
            print(f"{self._labels[i]:<10}{precisions[i]:<10.2f}{recalls[i]:<10.2f}{f1_scores[i]:<10.2f}")

    # helper
    @staticmethod
    def get_iou(bbox_true, bbox_pred):
        ixmin = max(bbox_pred[0], bbox_true[0])
        iymin = max(bbox_pred[1], bbox_true[1])
        ixmax = min(bbox_pred[2], bbox_true[2])	
        iymax = min(bbox_pred[3], bbox_true[3])
        iw = max(ixmax - ixmin, 0.) 
        ih = max(iymax - iymin, 0.)
        if iw == 0 or ih == 0:
            return 0 
        ai = iw * ih
        bbox_pred_area = (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1])
        bbox_true_area = (bbox_true[2] - bbox_true[0]) * (bbox_true[3] - bbox_true[1])
        au = (bbox_pred_area + bbox_true_area - ai)
        return ai / au