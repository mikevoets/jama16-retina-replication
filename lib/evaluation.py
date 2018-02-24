import tensorflow as tf


def perform_test(sess, init_op, summary_writer=None, epoch=None):
    tf.keras.backend.set_learning_phase(False)
    sess.run(init_op)

    # Retrieve all tensors.
    reset_tp = tf.get_collection('reset_tp')[0]
    reset_fp = tf.get_collection('reset_fp')[0]
    reset_fn = tf.get_collection('reset_fn')[0]
    reset_tn = tf.get_collection('reset_tn')[0]
    reset_brier = tf.get_collection('reset_brier')[0]
    reset_auc = tf.get_collection('reset_auc')[0]
    update_tp = tf.get_collection('update_tp')[0]
    update_fp = tf.get_collection('update_fp')[0]
    update_fn = tf.get_collection('update_fn')[0]
    update_tn = tf.get_collection('update_tn')[0]
    update_brier = tf.get_collection('update_brier')[0]
    update_auc = tf.get_collection('update_auc')[0]
    confusion_matrix = tf.get_collection('confusion_matrix')[0]
    brier = tf.get_collection('brier')[0]
    auc = tf.get_collection('auc')[0]
    summaries_op = tf.get_collection('summaries_op')[0]

    # Reset all streaming variables.
    sess.run([reset_tp, reset_fp, reset_fn, reset_tn, reset_brier, reset_auc])

    try:
        while True:
            # Retrieve the validation set confusion metrics.
            sess.run([update_tp, update_fp, update_fn,
                      update_tn, update_brier, update_auc])

    except tf.errors.OutOfRangeError:
        pass

    # Retrieve confusion matrix and estimated roc auc score.
    test_conf_matrix, test_brier, test_auc, summaries = sess.run(
        [confusion_matrix, brier, auc, summaries_op])

    # Write summary.
    if summary_writer is not None:
        summary_writer.add_summary(summaries, epoch)

    # Print total roc auc score for validation.
    print(f"Brier score: {test_brier:6.4}, AUC: {test_auc:10.8}")

    # Print confusion matrix.
    print(f"Confusion matrix:")
    print(test_conf_matrix[0])
