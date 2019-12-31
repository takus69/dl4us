import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# https://github.com/XifengGuo/DEC-keras
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

nmi = normalized_mutual_info_score

ari = adjusted_rand_score

def show_score(true, pred, verbose=0):
    acc_score = acc(true, pred)
    nmi_score = nmi(true, pred)
    ari_score = ari(true, pred)
    if verbose > 0:
        print('acc: {:.3f}, nmi: {:.3f}, ari: {:.3f}'.format(acc_score, nmi_score, ari_score))
    scores = {
        'acc': acc_score,
        'nmi': nmi_score,
        'ari': ari_score,
    }
    return scores
