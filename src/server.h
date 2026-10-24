#pragma once
// An OpenAI-compatible HTTP server over one model: /v1/chat/completions
// (streaming or not), /v1/completions, /v1/models and /health, so chat
// front-ends and the OpenAI SDKs can use brisk as they would the API.
#include <cstddef>
#include <string>

namespace brisk {

struct ServeOptions {
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::size_t threads = 0;     // CPU threads (0: all cores)
    bool gpu = false;
    std::size_t max_context = 0; // tokens the session holds (0: the model's limit, capped at 8192)
    std::string name;            // the model name reported by /v1/models (default: the file's basename)
    bool verbose = false;        // log each request's tokens and timings to stderr
    bool prefetch = true;        // page the weights in at start (the first request would otherwise pay for it)
    bool lock = false;           // also pin them in memory (mlock)
};

// Serves until the process is killed. Throws std::runtime_error on failure to load or bind.
int serve(const ServeOptions& options);

}  // namespace brisk
