import os
from omegaconf import OmegaConf
from typing import Dict, List, Set

global conf_dict


def get_strategies():
    strategies: List[str] = sorted(
        [strat for strat in os.listdir("kp_test") if strat[0].isupper()],
        key=str.lower,
    )
    conf_dict["strategies"] = strategies
    return conf_dict


def get_datasets():
    datasets: List[str] = sorted(
        [dset for dset in os.listdir("kp_test/" + conf_dict["strategies"][0])],
        key=str.lower,
    )
    conf_dict["datasets"] = datasets


def get_metrices():
    metrices: List[str] = [
        "auc_full_avg_dist_labeled_time_lag.csv.xz",
        "biggest_drop_per_accuracy_time_lag.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_macro_f1-score.csv.xz",
        "auc_full_IMPROVES_ACCURACY_BY_time_lag.csv.xz",
        "class_distributions_chebyshev_batch.csv.xz",
        "auc_full_weighted_precision.csv.xz",
        "auc_full_avg_dist_batch.csv.xz",
        "auc_full_biggest_drop_per_weighted_recall.csv.xz",
        "auc_full_macro_recall_time_lag.csv.xz",
        "nr_decreasing_al_cycles_per_weighted_precision_time_lag.csv.xz",
        "biggest_drop_per_weighted_recall.csv.xz",
        "biggest_drop_per_macro_f1-score_time_lag.csv.xz",
        "nr_decreasing_al_cycles_per_weighted_recall.csv.xz",
        "accuracy.csv.xz",
        "auc_full_biggest_drop_per_weighted_precision_time_lag.csv.xz",
        "auc_full_class_distributions_chebyshev_batch_time_lag.csv.xz",
        "biggest_drop_per_weighted_recall_time_lag.csv.xz",
        "auc_full_biggest_drop_per_macro_f1-score.csv.xz",
        "class_distributions_manhattan_added_up.csv.xz",
        "auc_full_CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS_kNN_time_lag.csv.xz",
        "nr_decreasing_al_cycles_per_macro_recall_time_lag.csv.xz",
        "class_distributions_chebyshev_added_up_time_lag.csv.xz",
        "auc_full_class_distributions_manhattan_added_up.csv.xz",
        "auc_full_biggest_drop_per_weighted_f1-score.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_macro_precision_time_lag.csv.xz",
        "auc_full_avg_dist_unlabeled.csv.xz",
        "auc_full_CLOSENESS_TO_DECISION_BOUNDARY.csv.xz",
        "auc_full_macro_precision.csv.xz",
        "nr_decreasing_al_cycles_per_accuracy_time_lag.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_weighted_f1-score.csv.xz",
        "weighted_f1-score.csv.xz",
        "biggest_drop_per_macro_f1-score.csv.xz",
        "auc_full_CLOSENESS_TO_SAMPLES_OF_SAME_CLASS_kNN.csv.xz",
        "macro_recall_time_lag.csv.xz",
        "auc_full_biggest_drop_per_macro_precision_time_lag.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_weighted_recall.csv.xz",
        "auc_full_REGION_DENSITY_time_lag.csv.xz",
        "class_distributions_chebyshev_batch_time_lag.csv.xz",
        "auc_full_OUTLIERNESS.csv.xz",
        "biggest_drop_per_weighted_precision_time_lag.csv.xz",
        "auc_full_class_distributions_manhattan_batch.csv.xz",
        "nr_decreasing_al_cycles_per_macro_precision_time_lag.csv.xz",
        "class_distributions_chebyshev_added_up.csv.xz",
        "auc_full_macro_f1-score_time_lag.csv.xz",
        "weighted_f1-score_time_lag.csv.xz",
        "macro_precision_time_lag.csv.xz",
        "query_selection_time.csv.xz",
        "macro_f1-score_time_lag.csv.xz",
        "auc_full_SWITCHES_CLASS_OFTEN.csv.xz",
        "macro_precision.csv.xz",
        "auc_full_COUNT_WRONG_CLASSIFICATIONS.csv.xz",
        "auc_full_CLOSENESS_TO_SAMPLES_OF_SAME_CLASS_kNN_time_lag.csv.xz",
        "auc_full_biggest_drop_per_weighted_precision.csv.xz",
        "auc_full_CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS.csv.xz",
        "auc_full_mismatch_train_test.csv.xz",
        "auc_full_weighted_recall_time_lag.csv.xz",
        "avg_dist_batch_time_lag.csv.xz",
        "class_distributions_manhattan_added_up_time_lag.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_weighted_f1-score_time_lag.csv.xz",
        "macro_recall.csv.xz",
        "auc_full_biggest_drop_per_accuracy.csv.xz",
        "auc_full_class_distributions_chebyshev_added_up.csv.xz",
        "auc_full_query_selection_time.csv.xz",
        "macro_f1-score.csv.xz",
        "auc_full_weighted_f1-score.csv.xz",
        "auc_full_AVERAGE_UNCERTAINTY.csv.xz",
        "auc_full_avg_dist_batch_time_lag.csv.xz",
        "nr_decreasing_al_cycles_per_weighted_f1-score_time_lag.csv.xz",
        "auc_full_biggest_drop_per_macro_f1-score_time_lag.csv.xz",
        "nr_decreasing_al_cycles_per_weighted_precision.csv.xz",
        "nr_decreasing_al_cycles_per_macro_f1-score.csv.xz",
        "auc_full_biggest_drop_per_macro_precision.csv.xz",
        "auc_full_CLOSENESS_TO_CLUSTER_CENTER_time_lag.csv.xz",
        "learner_training_time_time_lag.csv.xz",
        "auc_full_weighted_precision_time_lag.csv.xz",
        "auc_full_query_selection_time_time_lag.csv.xz",
        "auc_full_biggest_drop_per_macro_recall_time_lag.csv.xz",
        "auc_full_CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS_time_lag.csv.xz",
        "biggest_drop_per_macro_precision_time_lag.csv.xz",
        "auc_full_macro_recall.csv.xz",
        "biggest_drop_per_macro_recall.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_macro_recall.csv.xz",
        "auc_full_accuracy_time_lag.csv.xz",
        "auc_full_AVERAGE_UNCERTAINTY_time_lag.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_weighted_precision.csv.xz",
        "auc_full_class_distributions_chebyshev_batch.csv.xz",
        "auc_full_class_distributions_chebyshev_added_up_time_lag.csv.xz",
        "nr_decreasing_al_cycles_per_macro_precision.csv.xz",
        "mismatch_train_test_time_lag.csv.xz",
        "auc_full_OUTLIERNESS_time_lag.csv.xz",
        "biggest_drop_per_macro_precision.csv.xz",
        "auc_full_learner_training_time.csv.xz",
        "auc_full_class_distributions_manhattan_batch_time_lag.csv.xz",
        "auc_full_avg_dist_unlabeled_time_lag.csv.xz",
        "auc_full_weighted_recall.csv.xz",
        "auc_full_CLOSENESS_TO_SAMPLES_OF_SAME_CLASS_time_lag.csv.xz",
        "mismatch_train_test.csv.xz",
        "auc_full_biggest_drop_per_weighted_recall_time_lag.csv.xz",
        "biggest_drop_per_accuracy.csv.xz",
        "auc_full_biggest_drop_per_macro_recall.csv.xz",
        "learner_training_time.csv.xz",
        "query_selection_time_time_lag.csv.xz",
        "auc_full_MELTING_POT_REGION.csv.xz",
        "auc_full_mismatch_train_test_time_lag.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_macro_precision.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_macro_f1-score_time_lag.csv.xz",
        "auc_full_SWITCHES_CLASS_OFTEN_time_lag.csv.xz",
        "avg_dist_batch.csv.xz",
        "auc_full_avg_dist_labeled.csv.xz",
        "avg_dist_labeled.csv.xz",
        "biggest_drop_per_macro_recall_time_lag.csv.xz",
        "auc_full_IMPROVES_ACCURACY_BY.csv.xz",
        "auc_full_accuracy.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_weighted_recall_time_lag.csv.xz",
        "class_distributions_manhattan_batch_time_lag.csv.xz",
        "weighted_precision_time_lag.csv.xz",
        "auc_full_weighted_f1-score_time_lag.csv.xz",
        "weighted_recall_time_lag.csv.xz",
        "auc_full_CLOSENESS_TO_DECISION_BOUNDARY_time_lag.csv.xz",
        "auc_full_biggest_drop_per_weighted_f1-score_time_lag.csv.xz",
        "auc_full_biggest_drop_per_accuracy_time_lag.csv.xz",
        "auc_full_CLOSENESS_TO_SAMPLES_OF_SAME_CLASS.csv.xz",
        "weighted_precision.csv.xz",
        "auc_full_macro_f1-score.csv.xz",
        "accuracy_time_lag.csv.xz",
        "auc_full_class_distributions_manhattan_added_up_time_lag.csv.xz",
        "weighted_recall.csv.xz",
        "nr_decreasing_al_cycles_per_weighted_f1-score.csv.xz",
        "auc_full_MELTING_POT_REGION_time_lag.csv.xz",
        "class_distributions_manhattan_batch.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_accuracy_time_lag.csv.xz",
        "auc_full_learner_training_time_time_lag.csv.xz",
        "avg_dist_labeled_time_lag.csv.xz",
        "nr_decreasing_al_cycles_per_accuracy.csv.xz",
        "nr_decreasing_al_cycles_per_weighted_recall_time_lag.csv.xz",
        "avg_dist_unlabeled.csv.xz",
        "auc_full_CLOSENESS_TO_CLUSTER_CENTER.csv.xz",
        "avg_dist_unlabeled_time_lag.csv.xz",
        "auc_full_CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS_kNN.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_accuracy.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_macro_recall_time_lag.csv.xz",
        "auc_full_COUNT_WRONG_CLASSIFICATIONS_time_lag.csv.xz",
        "biggest_drop_per_weighted_f1-score.csv.xz",
        "biggest_drop_per_weighted_f1-score_time_lag.csv.xz",
        "biggest_drop_per_weighted_precision.csv.xz",
        "auc_full_macro_precision_time_lag.csv.xz",
        "nr_decreasing_al_cycles_per_macro_recall.csv.xz",
        "auc_full_REGION_DENSITY.csv.xz",
        "auc_full_nr_decreasing_al_cycles_per_weighted_precision_time_lag.csv.xz",
        "nr_decreasing_al_cycles_per_macro_f1-score_time_lag.csv.xz",
    ]
    metrices = sorted(metrices, key=str.lower)
    # validate the metrices
    blacklist: Set = set()
    for metric in metrices:
        for strat in conf_dict["strategies"]:
            for dset in conf_dict["datasets"]:
                if not os.path.exists("kp_test/" + strat + "/" + dset + "/" + metric):
                    blacklist.add(metric)
    metrices = list(set(metrices) - blacklist)
    conf_dict["metrices"] = metrices


def get_data() -> Dict[str, List[str]]:
    get_strategies()
    get_datasets()
    get_metrices()


# this need to be adjustet for new environments -> for example after merge
FILE_DIR: str = os.getcwd() + "/new/config/"
FILE_NAME: str = "data.yaml"

if os.path.exists(FILE_DIR + FILE_NAME):
    os.remove(FILE_DIR + FILE_NAME)

conf_dict: Dict[str, List[str]] = {}
get_data()
conf = OmegaConf.create(conf_dict)
OmegaConf.save(conf, FILE_DIR + FILE_NAME)
