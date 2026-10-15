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

constexpr int kIndexBits = 20, kTaskBits = 20;
constexpr std::uint64_t kIndexMask = (1ull << kIndexBits) - 1;
constexpr std::uint64_t kTaskMask = (1ull << kTaskBits) - 1;
constexpr std::size_t kMaxTasks = kTaskMask - 1024;  // claims may run a little past the count

inline std::uint64_t claim_generation(std::uint64_t claim) { return claim >> (kIndexBits + kTaskBits); }
inline std::size_t claim_tasks(std::uint64_t claim) { return static_cast<std::size_t>((claim >> kIndexBits) & kTaskMask); }
inline std::size_t claim_index(std::uint64_t claim) { return static_cast<std::size_t>(claim & kIndexMask); }

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
    // generation and task count, so a worker that is late to notice a new run
    // can never pair a task index with the wrong function.
    if (tasks > kMaxTasks) {
        for (std::size_t i = 0; i < tasks; ++i) fn(i);
        return;
    }
    fn_ = &fn;
    completed_.store(0, std::memory_order_relaxed);
    const std::uint64_t generation = (generation_.load(std::memory_order_relaxed) + 1) & 0xFFFFFF;
    claims_.store((generation << (kIndexBits + kTaskBits)) | (static_cast<std::uint64_t>(tasks) << kIndexBits),
                  std::memory_order_release);  // publishes fn_ to late claimers
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
    // Each claim takes one task index with a single atomic increment and
    // carries its run's generation and task count. A claim with an index
    // below its count belongs to a run that is still in progress, so fn_ is
    // that run's function. A claim for a newer generation than expected
    // means a new run started since this thread looked; it is executed for
    // that run.
    std::size_t done = 0;
    while (true) {
        const std::uint64_t claim = claims_.fetch_add(1, std::memory_order_acq_rel);
        const std::uint64_t claimed_generation = claim_generation(claim);
        if (claimed_generation != generation) {
            if (done > 0) completed_.fetch_add(done, std::memory_order_release);
            done = 0;
            generation = claimed_generation;
        }
        if (claim_index(claim) >= claim_tasks(claim)) break;
        (*fn_)(claim_index(claim));
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
