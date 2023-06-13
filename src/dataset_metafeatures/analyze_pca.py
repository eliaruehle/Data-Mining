import evaluate_metrics


def analyze_pca():
    eval_metrics = evaluate_metrics.Evaluate_Metrics("kp_test/datasets")

    eval_metrics.calculate_all_metrics()
    eval_metrics.generate_evaluations(
        normalisation=True, dimension_reductions=[["pca", {"n_components": 8}]]
    )

    reduced_cos_sim = eval_metrics.calculate_all_cosine_similarities(
        metafeatures_dict=eval_metrics.reduced_metafeatures_dict["pca"]
    )

    print(reduced_cos_sim)


if __name__ == "__main__":
    analyze_pca()
