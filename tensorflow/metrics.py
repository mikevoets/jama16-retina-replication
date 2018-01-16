import tensorflow as tf

num_labels = 1

# Global constant for zeroed labels.



def true_positives(labels, predictions):
    zeroed_labels = tf.zeros(shape=(num_labels), dtype=tf.int64)
    tp = tf.Variable(
        zeroed_labels, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    tp_op = tf.assign(
        tp, tf.add(tp, tf.count_nonzero(labels * predictions, axis=0)))
    return tp, tp_op


def true_negatives(labels, predictions):
    zeroed_labels = tf.zeros(shape=(num_labels), dtype=tf.int64)
    tn = tf.Variable(
        zeroed_labels, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    tn_op = tf.assign(
        tn, tf.add(tn, tf.count_nonzero((labels-1) * (predictions-1), axis=0)))
    return tn, tn_op


def false_positives(labels, predictions):
    zeroed_labels = tf.zeros(shape=(num_labels), dtype=tf.int64)
    fp = tf.Variable(
        zeroed_labels, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    fp_op = tf.assign(
        fp, tf.add(fp, tf.count_nonzero(labels * (predictions-1), axis=0)))
    return fp, fp_op


def false_negatives(labels, predictions):
    zeroed_labels = tf.zeros(shape=(num_labels), dtype=tf.int64)
    fn = tf.Variable(
        zeroed_labels, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    fn_op = tf.assign(
        fn, tf.add(fn, tf.count_nonzero((labels-1) * predictions, axis=0)))
    return fn, fn_op


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op


def confusion_matrix(labels, predictions):
    def compute_confm(tp, fp, fn, tn):
        return tf.reshape(tf.stack([tp, fp, fn, tn], 1), [num_labels, 2, 2])

    metrics = [
        m(labels, predictions)
        for m in [true_positives, false_positives,
                  false_negatives, true_negatives]]

    confm = compute_confm(*[x[0] for x in metrics])
    confm_ops = compute_confm(*[x[1] for x in metrics])

    return confm, confm_ops
