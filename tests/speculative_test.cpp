#include <cstdio>
#include <cstdlib>
#include <vector>

#include "speculative.h"

namespace {

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

using Tokens = std::vector<brisk::Token>;

void test_lookup_drafter() {
    brisk::LookupDrafter drafter;
    CHECK(drafter.draft(5).empty());  // nothing seen yet

    // "1 2 3 4 5 1 2 3" -> the suffix "1 2 3" occurred earlier, followed by "4 5 1 2 3".
    drafter.observe(Tokens{1, 2, 3, 4, 5, 1, 2, 3});
    CHECK(drafter.draft(5) == (Tokens{4, 5, 1, 2, 3}));
    CHECK(drafter.draft(1) == (Tokens{4}));

    // The most recent earlier occurrence wins: "... 7 8 9 | 7 8 10 | 7 8" -> expects 10.
    brisk::LookupDrafter recent;
    recent.observe(Tokens{7, 8, 9, 7, 8, 10, 7, 8});
    CHECK(recent.draft(3) == (Tokens{10, 7, 8}));

    // A match of the last token alone is used when longer suffixes do not repeat.
    brisk::LookupDrafter single;
    single.observe(Tokens{5, 6, 1, 5});
    CHECK(single.draft(4) == (Tokens{6, 1, 5}));

    // No repeat: nothing drafted.
    brisk::LookupDrafter fresh;
    fresh.observe(Tokens{1, 2, 3});
    CHECK(fresh.draft(4).empty());
}

}  // namespace

int main() {
    test_lookup_drafter();
    std::printf("speculative_test: ok\n");
    return 0;
}
