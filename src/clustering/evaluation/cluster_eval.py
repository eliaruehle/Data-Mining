import math
import os

from src.clustering import SimilarityMatrix


def get_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def do_eval(directory: str):
    csv_files = get_csv_files(directory)

    # Load and normalize matrices
    matrices = [SimilarityMatrix.from_csv(filepath=path).normalize().as_2d_list() for path in csv_files]

    avg_sim_euc = average_similarity_euclidean(matrices)
    print(f"Average Euclidean similarity: {avg_sim_euc}")

    avg_sim_cos = average_similarity_cosine(matrices)
    print(f"Average Cosine similarity: {avg_sim_cos}")


def euclidean_distance(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions.")

    squared_diff_sum = 0.0
    for i in range(len(matrix1)):
        for j in range(i + 1, len(matrix1[0])):  # Start from i+1 to consider only the upper half
            squared_diff_sum += (matrix1[i][j] - matrix2[i][j]) ** 2

    euclidean_dist = math.sqrt(squared_diff_sum)
    return euclidean_dist


def cosine_similarity(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions.")

    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for i in range(len(matrix1)):
        for j in range(i + 1, len(matrix1[0])):  # Start from i+1 to consider only the upper half
            dot_product += matrix1[i][j] * matrix2[i][j]
            norm1 += matrix1[i][j] ** 2
            norm2 += matrix2[i][j] ** 2

    cosine_sim = dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
    return cosine_sim


def average_similarity_euclidean(matrices):
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            similarity = 1.0 / (1.0 + euclidean_distance(matrices[i], matrices[j]))
            total_similarity += similarity
            num_comparisons += 1

    average_similarity = (total_similarity / num_comparisons) * 100
    return average_similarity


def average_similarity_cosine(matrices):
    total_similarity = 0.0
    num_comparisons = 0

    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            similarity = (cosine_similarity(matrices[i],
                                            matrices[j]) + 1) / 2  # Adjust cosine similarity to a range of [0, 1]
            total_similarity += similarity
            num_comparisons += 1

    average_similarity = (total_similarity / num_comparisons) * 100
    return average_similarity


do_eval(directory="/home/ature/University/6th-Semester/Data-Mining/src/cl_res")
