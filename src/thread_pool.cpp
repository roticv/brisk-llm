#include "thread_pool.h"

#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace brisk {

namespace {

// Asks the scheduler to treat the calling thread as latency-sensitive. On
// Apple silicon this keeps compute threads on the performance cores when
// those are free and off the efficiency cores' lowest frequencies.
void raise_thread_priority() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

// Workers spin this many times looking for new work before sleeping on the
// condition variable. Generation steps come about a millisecond apart, so
// spinning keeps the wake-up latency out of the per-token time.
constexpr int kSpinIterations = 100000;

constexpr std::uint64_t kIndexMask = 0xFFFFFFFFull;

}  // namespace

ThreadPool::ThreadPool(std::size_t threads) {
    raise_thread_priority();  // the caller works too
    for (std::size_t i = 1; i < threads; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
        generation_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    for (std::thread& worker : workers_) worker.join();
}

void ThreadPool::run(std::size_t tasks, const std::function<void(std::size_t)>& fn) {
    if (tasks == 0) return;
    if (workers_.empty() || tasks == 1) {
        for (std::size_t i = 0; i < tasks; ++i) fn(i);
        return;
    }

    // Publish this run, then bump the generation. Task claims carry the
    // generation in their high bits, so a worker that is late to notice a
    // new run can never take one of its tasks for the previous function.
    fn_ = &fn;
    tasks_ = tasks;
    completed_.store(0, std::memory_order_relaxed);
    const std::uint64_t generation = generation_.load(std::memory_order_relaxed) + 1;
    claims_.store(generation << 32, std::memory_order_release);  // publishes fn_ and tasks_ to late claimers
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        generation_.store(generation, std::memory_order_release);
    }
    wake_.notify_all();

    work(generation);
    // Wait for the tasks, not for every worker: a slow core that never got
    // a task does not hold up the run.
    while (completed_.load(std::memory_order_acquire) < tasks) {}
    fn_ = nullptr;
}

void ThreadPool::work(std::uint64_t generation) {
    // Each claim takes one task index with a single atomic increment. If the
    // claim turns out to carry a newer generation, a new run has started
    // since this thread looked: its function and task count were published
    // before that generation's claims, so the index is simply executed for
    // the new run.
    const std::function<void(std::size_t)>* fn = fn_;
    std::size_t tasks = tasks_;
    std::size_t done = 0;
    while (true) {
        const std::uint64_t claim = claims_.fetch_add(1, std::memory_order_acq_rel);
        const std::uint64_t claimed_generation = claim >> 32;
        if (claimed_generation != generation) {
            if (done > 0) completed_.fetch_add(done, std::memory_order_release);
            done = 0;
            generation = claimed_generation;
            fn = fn_;
            tasks = tasks_;
        }
        const std::size_t index = static_cast<std::size_t>(claim & kIndexMask);
        if (index >= tasks) break;
        (*fn)(index);
        ++done;
    }
    if (done > 0) completed_.fetch_add(done, std::memory_order_release);
}

void ThreadPool::worker_loop() {
    raise_thread_priority();
    std::uint64_t seen = 0;
    while (true) {
        // Wait for the next generation: spin first, then sleep.
        std::uint64_t generation = seen;
        for (int i = 0; i < kSpinIterations && generation == seen; ++i) {
            generation = generation_.load(std::memory_order_acquire);
        }
        if (generation == seen) {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return generation_.load(std::memory_order_acquire) != seen; });
            generation = generation_.load(std::memory_order_acquire);
        }
        seen = generation;
        if (stop_.load()) return;
        work(generation);
    }
}

}  // namespace brisk
