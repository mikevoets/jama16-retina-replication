import tensorflow as tf
import numpy as np

def _get_operations_by_names(graph, names):
    return [graph.get_operation_by_name(name) for name in names]


def _get_tensors_by_names(graph, names):
    return [graph.get_tensor_by_name(name) for name in names]


def perform_test(sess, init_op, summary_writer=None, epoch=None,
                 feed_dict_fn=None, feed_dict_args={}, custom_tensors=[]):
    tf.keras.backend.set_learning_phase(False)
    sess.run(init_op)

    if len(custom_tensors) == 0:
        # Retrieve all default tensors and operations.
        graph = tf.get_default_graph()
        reset_tp, reset_fp, reset_fn, reset_tn, reset_brier, reset_auc = \
            _get_operations_by_names(
                graph, ['tp/reset', 'fp/reset', 'fn/reset', 'tn/reset',
                        'brier/reset', 'auc/reset'])

        update_tp, update_fp, update_fn, update_tn, update_brier, update_auc, \
        brier, auc, confusion_matrix, summaries_op = \
            _get_tensors_by_names(
                graph, ['tp/true_positives/AssignAdd:0',
                        'fp/false_positives/AssignAdd:0',
                        'fn/false_negatives/AssignAdd:0',
                        'tn/true_negatives/AssignAdd:0',
                        'brier/mean_squared_error/update_op:0',
                        'auc/auc/update_op:0',
                        'brier/mean_squared_error/value:0',
                        'auc/auc/value:0',
                        'confusion_matrix/Cast:0',
                        'Merge/MergeSummary:0'])

        # Reset all streaming variables.
        sess.run([reset_tp, reset_fp, reset_fn, reset_tn, reset_brier, reset_auc])

        # Create an array with tensors to run for each batch.
        tensors = [update_tp, update_fp, update_fn,
                         update_tn, update_brier, update_auc]
    else:
        tensors = custom_tensors

    try:
        batch_results = []
        while True:
            if feed_dict_fn is not None:
                feed_dict = feed_dict_fn(**feed_dict_args)
            else:
                feed_dict = None

            # Retrieve the validation set confusion metrics.
            batch_results.append(sess.run(tensors, feed_dict))

    except tf.errors.OutOfRangeError:
        pass

    # Yield the result if custom tensors were defined.
    if len(custom_tensors) > 0:
        return [np.vstack(x) for x in zip(*batch_results)]

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

    return test_auc
