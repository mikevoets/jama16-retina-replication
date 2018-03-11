import tensorflow as tf


def zeroed_labels(num_labels=1):
    return tf.zeros(shape=(num_labels), dtype=tf.int64)


def true_positives(labels, predictions, num_labels=1):
    tp = tf.Variable(
        zeroed_labels(num_labels), collections=[tf.GraphKeys.LOCAL_VARIABLES],
        name='tp')
    tp_op = tf.assign(
        tp, tf.add(tp, tf.count_nonzero(labels * predictions, axis=0)),
        name='op')
    return tp, tp_op


def true_negatives(labels, predictions, num_labels=1):
    tn = tf.Variable(
        zeroed_labels(num_labels), collections=[tf.GraphKeys.LOCAL_VARIABLES],
        name='tn')
    tn_op = tf.assign(
        tn, tf.add(tn, tf.count_nonzero((labels-1) * (predictions-1), axis=0)),
        name='op')
    return tn, tn_op


def false_positives(labels, predictions, num_labels=1):
    fp = tf.Variable(
        zeroed_labels(num_labels), collections=[tf.GraphKeys.LOCAL_VARIABLES],
        name='fp')
    fp_op = tf.assign(
        fp, tf.add(fp, tf.count_nonzero(labels * (predictions-1), axis=0)),
        name='op')
    return fp, fp_op


def false_negatives(labels, predictions, num_labels=1):
    fn = tf.Variable(
        zeroed_labels(num_labels), collections=[tf.GraphKeys.LOCAL_VARIABLES],
        name='fn')
    fn_op = tf.assign(
        fn, tf.add(fn, tf.count_nonzero((labels-1) * predictions, axis=0)),
        name='op')
    return fn, fn_op


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    with tf.variable_scope(scope) as s:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(
            s, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars, name='reset')
    return metric_op, update_op, reset_op


def confusion_matrix(tp, fp, fn, tn, num_labels=1, scope='confusion_matrix'):
    with tf.variable_scope(scope) as s:
        return tf.cast(tf.reshape(tf.stack([tp, fp, fn, tn], 0),
                                  [num_labels, 2, 2],
                                  name='confusion_matrix'), dtype=tf.int32)
