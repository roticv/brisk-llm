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

    fn_ = &fn;
    tasks_ = tasks;
    next_task_.store(0, std::memory_order_relaxed);
    finished_workers_.store(0, std::memory_order_relaxed);
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        generation_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();

    work();
    while (finished_workers_.load(std::memory_order_acquire) < workers_.size()) {}
    fn_ = nullptr;
}

void ThreadPool::work() {
    for (std::size_t i = next_task_.fetch_add(1, std::memory_order_relaxed); i < tasks_;
         i = next_task_.fetch_add(1, std::memory_order_relaxed)) {
        (*fn_)(i);
    }
}

void ThreadPool::worker_loop() {
    raise_thread_priority();
    std::uint64_t seen = 0;
    while (true) {
        // Wait for the next generation: spin first, then sleep.
        bool ready = false;
        for (int i = 0; i < kSpinIterations && !ready; ++i) {
            ready = generation_.load(std::memory_order_acquire) != seen;
        }
        if (!ready) {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return generation_.load(std::memory_order_acquire) != seen; });
        }
        seen = generation_.load(std::memory_order_acquire);
        if (stop_.load()) return;

        work();
        finished_workers_.fetch_add(1, std::memory_order_release);
    }
}

}  // namespace brisk
