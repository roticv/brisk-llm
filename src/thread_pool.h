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
// which is what lets a mix of performance and efficiency cores all help.
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
    void work();

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::atomic<std::uint64_t> generation_{0};  // bumped once per run()
    std::atomic<bool> stop_{false};

    // State of the current run().
    const std::function<void(std::size_t)>* fn_ = nullptr;
    std::size_t tasks_ = 0;
    std::atomic<std::size_t> next_task_{0};
    std::atomic<std::size_t> finished_workers_{0};
};

}  // namespace brisk
