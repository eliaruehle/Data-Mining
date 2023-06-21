import itertools
import math
import os

from scipy.stats import kendalltau, spearmanr
from itertools import combinations

from src.clustering import SimilarityMatrix


def get_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def calculate_similarity(directory: str):
    csv_files = get_csv_files(directory)

    # Load and normalize matrices
    matrices = [SimilarityMatrix.from_csv(filepath=path).normalize() for path in csv_files]

    avg_sim_euc = average_similarity_euclidean([matrix.as_2d_list() for matrix in matrices])
    print(f"Average Euclidean similarity: {avg_sim_euc}")

    avg_sim_cos = average_similarity_cosine([matrix.as_2d_list() for matrix in matrices])
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


# Deprecated
def compute_alignment(a, b):
    # Calculate the alignment between two lists a and b
    m = len(a)
    n = len(b)
    # Initialize the matrix
    D = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        D[i][0] = i
    for j in range(1, n + 1):
        D[0][j] = j
    # Calculate the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + cost)
    # Backtrace to find the alignment
    alignment_a = []
    alignment_b = []
    i = m
    j = n
    while i > 0 and j > 0:
        cost = 0 if a[i - 1] == b[j - 1] else 1
        if D[i][j] == D[i - 1][j - 1] + cost:
            alignment_a.insert(0, a[i - 1])
            alignment_b.insert(0, b[j - 1])
            i -= 1
            j -= 1
        elif D[i][j] == D[i - 1][j] + 1:
            alignment_a.insert(0, a[i - 1])
            alignment_b.insert(0, None)
            i -= 1
        else:
            alignment_a.insert(0, None)
            alignment_b.insert(0, b[j - 1])
            j -= 1
    while i > 0:
        alignment_a.insert(0, a[i - 1])
        alignment_b.insert(0, None)
        i -= 1
    while j > 0:
        alignment_a.insert(0, None)
        alignment_b.insert(0, b[j - 1])
        j -= 1
    return alignment_a, alignment_b



# Deprecated
def compute_alignment_cost(alignment):
    # Calculate the cost of an alignment
    cost = 0
    for a, b in zip(*alignment):
        if a is not None and b is not None:
            cost += 1
    return cost


# Deprecated
def find_best_alignment(lists):
    # Calculate all possible alignments between the lists and return the alignment with the minimum cost
    alignments = list(itertools.permutations(lists))
    best_alignment = None
    min_cost = float("inf")
    for alignment in alignments:
        aligned_lists = [alignment[0]]
        cost = 0
        for i in range(1, len(alignment)):
            a, b = compute_alignment(alignment[i - 1], alignment[i])
            aligned_lists.append(b)
            cost += compute_alignment_cost((a, b))
        if cost < min_cost:
            min_cost = cost
            best_alignment = aligned_lists
    return best_alignment, min_cost


# Deprecated
def calculate_similarity_scores(directory: str):
    csv_files = get_csv_files(directory)

    # Load and normalize matrices
    dictionaries = [SimilarityMatrix.from_csv(filepath=path).normalize().get_ordered_similarities() for path in
                    csv_files]

    keys = dictionaries[0].keys()
    scores = {}

    for key in keys:
        lists = [dictionary[key] for dictionary in dictionaries]
        best_alignment, min_cost = find_best_alignment(lists)
        scores[key] = min_cost

    return scores


# Average spearmanr: 0.5533234126984128
# Average kendalltau: 0.49801587301587313
def calc_average_ranking_agreement(directory: str):
    csv_files = get_csv_files(directory)

    # Load and normalize matrices
    dictionaries = [SimilarityMatrix.from_csv(filepath=path).normalize().get_ordered_similarities() for path in
                    csv_files]

    spear = {}
    tau = {}

    keys = dictionaries[0].keys()
    for key in keys:
        rankings = [dictionary[key] for dictionary in dictionaries]

        total_spear_correlation = 0
        total_tau_correlation = 0
        num_pairs = 0

        for rank1, rank2 in combinations(rankings, 2):
            spear_correlation, _ = spearmanr(rank1, rank2)
            total_spear_correlation = total_spear_correlation + spear_correlation

            tau_correlation, _ = kendalltau(rank1, rank2)
            total_tau_correlation = total_tau_correlation + tau_correlation

            num_pairs = num_pairs + 1

        avg_spear_correlation = total_spear_correlation / num_pairs
        spear[key] = avg_spear_correlation

        avg_tau_correlation = total_tau_correlation / num_pairs
        tau[key] = avg_tau_correlation

    spear = [spear[key] for key in spear]
    tau = [tau[key] for key in tau]

    avg_spear = sum(spear) / len(spear)
    avg_tau = sum(tau) / len(tau)

    return avg_spear, avg_tau


# Directory where similarity matrices are stored
source_directory = "/home/ature/University/6th-Semester/Data-Mining/src/clustering/generated/cl_res_small_dataset"

# Calculate both average Euclidean and average cosine similarity
# calculate_similarity(directory=source_directory)

spear, tau = calc_average_ranking_agreement(directory=source_directory)

print(f"Average spearmanr: {spear}")
print(f"Average kendalltau: {tau}")
