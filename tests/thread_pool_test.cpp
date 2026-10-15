#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "thread_pool.h"

namespace {

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

// Every task of every run must execute exactly once, for all task counts
// around the thread count and for many consecutive runs.
void test_each_task_runs_once(std::size_t threads) {
    brisk::ThreadPool pool(threads);
    for (std::size_t tasks : {std::size_t{1}, std::size_t{2}, threads, threads + 1, std::size_t{37}, std::size_t{1000}}) {
        std::vector<std::atomic<int>> hits(tasks);
        for (auto& h : hits) h.store(0);
        for (int round = 0; round < 200; ++round) {
            pool.run(tasks, [&](std::size_t i) { hits[i].fetch_add(1, std::memory_order_relaxed); });
        }
        for (std::size_t i = 0; i < tasks; ++i) CHECK(hits[i].load() == 200);
    }
}

// Back-to-back runs with different task counts and functions, so a worker
// that notices a run late sees the next one already in place. This is the
// shape of the race that once let a worker pair a stale function with a new
// task count.
void test_rapid_runs_keep_functions_and_counts_together(std::size_t threads) {
    brisk::ThreadPool pool(threads);
    std::atomic<long> sum{0};
    long expected = 0;
    for (int round = 0; round < 3000; ++round) {
        const std::size_t tasks = static_cast<std::size_t>(1 + (round * 7) % 23);
        const long weight = round + 1;
        pool.run(tasks, [&, weight](std::size_t i) { sum.fetch_add(weight * static_cast<long>(i + 1), std::memory_order_relaxed); });
        expected += weight * static_cast<long>(tasks * (tasks + 1) / 2);
    }
    CHECK(sum.load() == expected);
}

// run() must not return before every task has finished its work.
void test_run_waits_for_completion(std::size_t threads) {
    brisk::ThreadPool pool(threads);
    for (int round = 0; round < 50; ++round) {
        std::atomic<int> finished{0};
        pool.run(16, [&](std::size_t) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            finished.fetch_add(1);
        });
        CHECK(finished.load() == 16);
    }
}

}  // namespace

int main() {
    for (const std::size_t threads : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{10}}) {
        test_each_task_runs_once(threads);
        test_rapid_runs_keep_functions_and_counts_together(threads);
        test_run_waits_for_completion(threads);
    }
    std::printf("thread_pool_test: ok\n");
    return 0;
}
