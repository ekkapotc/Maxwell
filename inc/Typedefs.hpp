#ifndef STRUCT_INCLUDE
#define STRUCT_INCLUDE

#include <cstddef>
#include <cstdint>

namespace maxwell
{

/*
 * SVEGP-08 : "unsigned long" is 64-bit on Linux/macOS but only 32-bit on
 * Windows/MSVC (LLP64).  largeint is used both as a vertex index and as the
 * MEMORY BUDGET type, so on Windows the budget silently capped at 4 GB and the
 * vertex counter wrapped after ~4e9 recorded operations.
 */
typedef unsigned char flag_t;
typedef std::uint64_t idx_t;
typedef idx_t         largeint;

enum elim_t
{
  FORWARD_ELIM,
  REVERSE_ELIM
};

}

#endif
