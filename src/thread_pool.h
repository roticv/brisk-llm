#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace brisk {

// A fixed set of worker threads that split numbered tasks between them. The
// calling thread works too, so a pool of size 1 runs everything inline.
// Tasks are handed out one at a time, so faster cores take more of them,
// which is what lets a mix of performance and efficiency cores all help, and
// a run ends when its tasks are done, not when every worker has reported in.
class ThreadPool {
public:
    explicit ThreadPool(std::size_t threads);  // total threads, caller included
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t size() const { return workers_.size() + 1; }

    // Calls fn(i) for each i in [0, tasks) across all threads and returns when
    // every call has finished. Not reentrant.
    void run(std::size_t tasks, const std::function<void(std::size_t)>& fn);

private:
    void worker_loop();
    void work(std::uint64_t generation);

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::atomic<std::uint64_t> generation_{0};  // bumped once per run()
    std::atomic<bool> stop_{false};

    // State of the current run(). `claims_` packs the run's generation (24
    // bits), its task count (20) and the next unclaimed index (20), so one
    // atomic increment yields a claim that is valid for exactly one run.
    // fn_ is only read under a valid claim, i.e. while that run is in
    // progress and fn_ is its function.
    const std::function<void(std::size_t)>* fn_ = nullptr;
    std::atomic<std::uint64_t> claims_{0};
    std::atomic<std::size_t> completed_{0};
};

}  // namespace brisk
