import copy
import platform
import random
import statistics
import sys
import time
import matplotlib.pyplot as plt

# Fixed seed so every generated dataset can be reproduced exactly
RANDOM_SEED = 2026
MIN_TIME_MS = 0
MAX_TIME_MS = 9_999_999
MAX_DIGITS = 7  # 0 through 9,999,999 has at most 7 decimal digits
INPUT_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]
TRIALS_PER_SIZE = 3


class MergeSortMetrics:
    """Tracks basic operations performed by Merge Sort."""
    def __init__(self):
        self.key_comparisons = 0
        self.element_copies = 0

    @property
    def total_operations(self):
        return self.key_comparisons + self.element_copies


class RadixSortMetrics:
    """Tracks basic operations performed by LSD Radix Sort."""
    def __init__(self):
        self.digit_inspections = 0
        self.element_placements = 0

    @property
    def total_operations(self):
        return self.digit_inspections + self.element_placements


# =====================================================================
# 1. MERGE SORT IMPLEMENTATION (Following Lecture 3, Slides 2-3)
# =====================================================================

def merge_lists(arr, start1, end1, start2, end2, temp_result, metrics):
    """
    Merges two contiguous sorted sublists arr[start1..end1] and arr[start2..end2]
    into temp_result and copies the merged range back into arr.
    """
    final_start = start1
    final_end = end2
    index_c = final_start

    while start1 <= end1 and start2 <= end2:
        metrics.key_comparisons += 1
        # Using <= preserves stability for equal finish times
        if arr[start1] <= arr[start2]:
            temp_result[index_c] = arr[start1]
            start1 += 1
        else:
            temp_result[index_c] = arr[start2]
            start2 += 1
        metrics.element_copies += 1
        index_c += 1

    # Move the part of the sublist that is left over
    if start1 <= end1:
        for i in range(start1, end1 + 1):
            temp_result[index_c] = arr[i]
            metrics.element_copies += 1
            index_c += 1
    else:
        for i in range(start2, end2 + 1):
            temp_result[index_c] = arr[i]
            metrics.element_copies += 1
            index_c += 1

    # Copy merged elements back into original list segment
    for i in range(final_start, final_end + 1):
        arr[i] = temp_result[i]
        metrics.element_copies += 1


def _merge_sort_recursive(arr, first, last, temp_result, metrics):
    """Recursive divide-and-conquer helper matching MergeSort(list, first, last)."""
    if first < last:
        middle = (first + last) // 2
        _merge_sort_recursive(arr, first, middle, temp_result, metrics)
        _merge_sort_recursive(arr, middle + 1, last, temp_result, metrics)
        merge_lists(arr, first, middle, middle + 1, last, temp_result, metrics)


def merge_sort(data):
    """
    Entry point for Merge Sort.
    Preserves original input by sorting a working copy.
    Returns (sorted_list, MergeSortMetrics).
    """
    arr = list(data)
    metrics = MergeSortMetrics()
    n = len(arr)
    if n > 1:
        temp_result = [0] * n
        _merge_sort_recursive(arr, 0, n - 1, temp_result, metrics)
    return arr, metrics


# =====================================================================
# 2. LSD RADIX SORT IMPLEMENTATION (Following Lecture 2, Slide 44)
# =====================================================================

def radix_sort_lsd(data, key_size=MAX_DIGITS):
    """
    Stable Least-Significant-Digit (LSD) Radix Sort in base 10.
    Follows the bucket distribution and CombineBuckets approach from Lecture 2.
    Handles leading zeroes implicitly via integer division and modulo 10.
    Returns (sorted_list, RadixSortMetrics).
    """
    arr = list(data)
    metrics = RadixSortMetrics()
    n = len(arr)
    if n <= 1:
        return arr, metrics

    shift = 1
    for _ in range(key_size):
        buckets = [[] for _ in range(10)]

        # Distribute entries into base-10 buckets by current digit
        for entry in range(n):
            val = arr[entry]
            bucket_number = (val // shift) % 10
            metrics.digit_inspections += 1

            buckets[bucket_number].append(val)
            metrics.element_placements += 1

        # CombineBuckets(): recombine buckets 0..9 in order to preserve stability
        idx = 0
        for b in range(10):
            for val in buckets[b]:
                arr[idx] = val
                metrics.element_placements += 1
                idx += 1

        shift *= 10

    return arr, metrics


# =====================================================================
# 3. CORRECTNESS VERIFICATION
# =====================================================================

def is_ascending(arr):
    """Checks that every element is <= the next element."""
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


def verify_sort_output(original, sorted_merge, sorted_radix):
    """
    Verifies that both outputs are in ascending order, match each other,
    and contain the exact same multiset of values as the original input.
    """
    expected = sorted(original)
    merge_ok = is_ascending(sorted_merge) and (sorted_merge == expected)
    radix_ok = is_ascending(sorted_radix) and (sorted_radix == expected)
    same_output = (sorted_merge == sorted_radix)
    return merge_ok and radix_ok and same_output


def run_correctness_suite():
    """Runs hand-checkable test cases required before benchmarking."""
    test_cases = [
        ("Empty list", []),
        ("One-value list", [42050]),
        ("Short hand-checked list", [125000, 45, 9999999, 0, 305, 45, 7]),
        ("Repeated values", [500, 120, 500, 120, 0, 0, 500, 120]),
        ("Already sorted values", [0, 15, 240, 1024, 50000, 1200000, 9999999]),
        ("Reverse-ordered values", [9999999, 850000, 40000, 3200, 150, 9, 0]),
    ]

    print("=" * 75)
    print("STEP 1: CORRECTNESS VERIFICATION ON HAND-CHECKABLE TEST CASES")
    print("=" * 75)

    for name, dataset in test_cases:
        input_snapshot = copy.deepcopy(dataset)
        out_merge, _ = merge_sort(dataset)
        out_radix, _ = radix_sort_lsd(dataset)

        # Ensure original input was not mutated
        input_preserved = (dataset == input_snapshot)
        passed = verify_sort_output(dataset, out_merge, out_radix) and input_preserved

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        print(f"       Input       : {dataset}")
        print(f"       Merge Output: {out_merge}")
        print(f"       Radix Output: {out_radix}")
        if not passed:
            raise AssertionError(f"Correctness check failed on {name}")
    print("-" * 75)
    print("All hand-checked correctness tests passed successfully.\n")


# =====================================================================
# 4. BENCHMARKING & CHART GENERATION
# =====================================================================

def run_benchmarks():
    print("=" * 75)
    print("ENVIRONMENT & SYSTEM DETAILS")
    print("=" * 75)
    print(f"OS / Platform       : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Processor / Machine : {platform.processor()} / {platform.machine()}")
    print(f"Python Version      : {sys.version.split()[0]}")
    print(f"Timer Function      : time.perf_counter() (high-resolution monotonic clock)")
    print(f"Random Seed         : {RANDOM_SEED}")
    print("=" * 75 + "\n")

    rng = random.Random(RANDOM_SEED)
    results = []

    print("=" * 75)
    print("STEP 2 & 3: BENCHMARKING ON RACE DATASETS (MEDIAN OF 3 RUNS)")
    print("=" * 75)

    for n in INPUT_SIZES:
        # Generate one random dataset for this size
        original_data = [rng.randint(MIN_TIME_MS, MAX_TIME_MS) for _ in range(n)]

        merge_times_ms = []
        radix_times_ms = []
        merge_metrics_final = None
        radix_metrics_final = None

        # 3 trials for Merge Sort
        for trial in range(TRIALS_PER_SIZE):
            work_copy = list(original_data)
            t0 = time.perf_counter()
            sorted_m, m_metrics = merge_sort(work_copy)
            t1 = time.perf_counter()
            merge_times_ms.append((t1 - t0) * 1000.0)
            if trial == 0:
                merge_metrics_final = m_metrics
                saved_merge_out = sorted_m

        # 3 trials for LSD Radix Sort
        for trial in range(TRIALS_PER_SIZE):
            work_copy = list(original_data)
            t0 = time.perf_counter()
            sorted_r, r_metrics = radix_sort_lsd(work_copy)
            t1 = time.perf_counter()
            radix_times_ms.append((t1 - t0) * 1000.0)
            if trial == 0:
                radix_metrics_final = r_metrics
                saved_radix_out = sorted_r

        # Verify correctness on every generated race dataset
        assert verify_sort_output(original_data, saved_merge_out, saved_radix_out), (
            f"Verification failed for N={n}"
        )

        med_merge_ms = statistics.median(merge_times_ms)
        med_radix_ms = statistics.median(radix_times_ms)

        row = {
            "n": n,
            "merge_time_ms": med_merge_ms,
            "merge_trials_ms": merge_times_ms,
            "merge_comparisons": merge_metrics_final.key_comparisons,
            "merge_copies": merge_metrics_final.element_copies,
            "merge_total_ops": merge_metrics_final.total_operations,
            "radix_time_ms": med_radix_ms,
            "radix_trials_ms": radix_times_ms,
            "radix_inspections": radix_metrics_final.digit_inspections,
            "radix_placements": radix_metrics_final.element_placements,
            "radix_total_ops": radix_metrics_final.total_operations,
        }
        results.append(row)

        print(
            f"N = {n:7,d} | VERIFIED OK | "
            f"Merge Median: {med_merge_ms:8.3f} ms "
            f"(Comp: {row['merge_comparisons']:,}, Copies: {row['merge_copies']:,}) | "
            f"Radix Median: {med_radix_ms:8.3f} ms "
            f"(Digit Insp: {row['radix_inspections']:,}, Placements: {row['radix_placements']:,})"
        )

    return results


def generate_charts(results):
    sizes = [r["n"] for r in results]
    merge_times = [r["merge_time_ms"] for r in results]
    radix_times = [r["radix_time_ms"] for r in results]

    merge_comps = [r["merge_comparisons"] for r in results]
    merge_copies = [r["merge_copies"] for r in results]
    radix_insp = [r["radix_inspections"] for r in results]
    radix_place = [r["radix_placements"] for r in results]

    # Chart 1: Elapsed Time vs Input Size
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(sizes, merge_times, marker="o", linewidth=2, label="Merge Sort (Median Time)")
    plt.plot(sizes, radix_times, marker="s", linewidth=2, label="LSD Radix Sort (Median Time)")
    plt.title("Elapsed Sorting Time vs. Input Size (N)")
    plt.xlabel("Input Size N (number of finish-time records)")
    plt.ylabel("Median Elapsed Time (milliseconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("elapsed_time_chart.png")
    plt.close()

    # Chart 2: Operation Counts vs Input Size (Side-by-Side Subplots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    ax1.plot(sizes, merge_comps, marker="o", linewidth=2, label="Key Comparisons")
    ax1.plot(sizes, merge_copies, marker="^", linewidth=2, label="Element Copies")
    ax1.set_title("Merge Sort Operation Counts vs. Input Size (N)")
    ax1.set_xlabel("Input Size N (records)")
    ax1.set_ylabel("Operation Count")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(sizes, radix_insp, marker="s", linewidth=2, color="tab:green", label="Digit Inspections")
    ax2.plot(sizes, radix_place, marker="d", linewidth=2, color="tab:red", label="Element Placements")
    ax2.set_title("LSD Radix Sort Operation Counts vs. Input Size (N)")
    ax2.set_xlabel("Input Size N (records)")
    ax2.set_ylabel("Operation Count")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("operation_counts_chart.png")
    plt.close()
    print("\nCharts saved successfully: 'elapsed_time_chart.png' and 'operation_counts_chart.png'.")


if __name__ == "__main__":
    run_correctness_suite()
    benchmark_results = run_benchmarks()
    generate_charts(benchmark_results)