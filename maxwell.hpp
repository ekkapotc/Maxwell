#ifndef MAXWELL_INCLUDE
#define MAXWELL_INCLUDE

/*
 * The public surface: the active scalar, the free-function API, the Tape and
 * Jacobian value types, the typedefs those need, and the exception the
 * checkpoint loop throws.
 *
 * There are two ways to drive the library and both are here.  Tape (SVEGP-28)
 * is the one to reach for: a constructor and a destructor in place of
 * initialize()/finalize(), a run() that owns the checkpoint loop's try/catch,
 * and a Jacobian that is a value rather than a double** you must remember to
 * free.  The free functions underneath it are what every example still uses and
 * what Tape itself calls; they remain for the cases Tape cannot express, such
 * as stepping two tapes against each other.
 *
 * Vertex, Edge, Process and TapeState are internals and are deliberately NOT
 * included here -- Tape holds one forward-declared pointer.  Nothing outside
 * src/ has ever used them, and keeping them out means their representation can
 * change without breaking a single caller.  It already has, twice: the edge
 * arena and then sorted adjacency took the graph from 120 bytes and three
 * allocations per edge to 72 and one, and not one caller noticed.
 *
 * This header no longer opens the namespace for you.  Say
 *
 *     using namespace maxwell;
 *
 * in your own translation unit, or qualify: maxwell::active, maxwell::sin.
 */

#include "inc/Typedefs.hpp"
#include "inc/Active.hpp"
#include "inc/API.hpp"
#include "inc/BreakException.hpp"
#include "inc/Tape.hpp"

#endif
