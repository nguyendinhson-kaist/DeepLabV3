import numpy as np

class SemanticMetric():
    """Evaluation metric for segmentation task

    Arguments:
    - num_classes: the number of classes
    """
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.overall_hist = np.zeros(shape=(self.num_classes, self.num_classes))

    def _calculate_hist(self, gt, pred):
        """Calculate histogram of gt and pred
        Returned array has:
        - Diagonal contains the number of TRUE POSITIVE
        - Row contains the number FALSE NEGATIVE (exclude diag elements)
        - Colum contains the number FALSE POSITIVE (exclude diag elements)

        Return:
        - numpy array with size (num_classes, num_classes)
        """
        void_mask = (gt >= 0) & (gt < self.num_classes)
        hist = np.bincount(
            self.num_classes*gt[void_mask].astype(int) + pred[void_mask].astype(int),
            minlength=self.num_classes**2
        ).reshape((self.num_classes, self.num_classes))

        return hist
    
    def update(self, gt, pred):
        """Update metric by given ground truth and predicted labels

        Inputs:
        - gt: numpy array of ground truths
        - pred: numpy array of predictions
        """
        self.overall_hist += self._calculate_hist(gt, pred)

    def get_results(self):
        """Get results
        
        Return:
        - (mIoU, iou)
        """
        tp = np.diag(self.overall_hist)
        fn = self.overall_hist.sum(axis=1) - tp
        fp = self.overall_hist.sum(axis=0) - tp

        iou = tp / (tp+fp+fn)
        mIoU = np.nanmean(iou)

        return mIoU, iou
    
    def reset(self):
        """Reset evaluation metric
        """
        self.overall_hist = np.zeros(shape=(self.num_classes, self.num_classes))