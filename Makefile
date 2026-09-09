#
# Maxwell -- automatic differentiation by sparse graph vertex elimination
#
#   inc/         headers.  The public surface is maxwell.hpp, which pulls in
#                Typedefs, Active, API and BreakException.  Vertex, Edge,
#                Process and Tape are internals and are deliberately not
#                reachable from it.
#   src/         the library.  Six translation units, built by src/Makefile
#                into build/lib/libmaxwell.a.
#   examples/    one directory per example: a main.cpp and a Makefile of one
#                to four lines that includes ../common.mk.  examples/Makefile
#                discovers them from the filesystem -- there is no list to
#                edit here or anywhere else when one is added or removed.
#   build/       everything generated.  Nothing is written into the source
#                tree, and `make clean` is `rm -rf build`.
#
#     build/lib/libmaxwell.a
#     build/obj/src/*.o            one per library translation unit
#     build/obj/examples/<name>.o  one per example
#     build/bin/<name>             one per example, named after its directory
#
# Targets
#   make            the library and every example
#   make test       build, then run the whole suite in order
#   make list       every example discovered, with the arguments it runs with
#   make sanitize   rebuild under AddressSanitizer + LeakSanitizer, in a
#                   SEPARATE tree, and run the assertion suites
#   make clean      remove build/
#
# BUILD_DIR is absolute and exported, so the sub-makes agree on one tree no
# matter where make was invoked from.  src/Makefile and examples/common.mk each
# carry an ?= fallback computed from their own location as well, so
# `cd src && make` and `cd examples/pinn && make` still work standalone -- and
# common.mk's rule for the library means a single example bootstraps the whole
# thing from a cold tree.
#
BUILD_DIR := $(CURDIR)/build
export BUILD_DIR

CXX       ?= g++

# src/Makefile compiles with -g -std=gnu++0x and no warning flags of its own.
# The library is clean under these, and a top-level build is the only place
# that can say so for the whole tree, so they are turned on here.  Overridable:
#   make WARN=
STD       ?= -std=gnu++0x
WARN      ?= -Wall -Wextra
LIB_FLAGS ?= -g $(STD) $(WARN)

.PHONY: all lib examples test list sanitize clean help

all: examples

lib:
	@$(MAKE) --no-print-directory -C src FLAGS="$(LIB_FLAGS)"

# Not just for ordering under -j: examples/common.mk will happily bootstrap the
# library itself, and if several example directories raced into that rule at
# once they would each try to write the same archive.
examples: lib
	@$(MAKE) --no-print-directory -C examples all

test: all
	@$(MAKE) --no-print-directory -C examples test

list:
	@$(MAKE) --no-print-directory -C examples list

# ---------------------------------------------------------------- sanitize --
#
# A separate BUILD_DIR rather than a flags stamp.  make compares timestamps
# only, so sanitizing into build/ leaves every binary newer than its source and
# a following plain `make` rebuilds nothing -- you keep running instrumented
# code without being told, which is how a fast example comes to look like a
# hang.  Two trees cannot do that to you.
#
# The library has to be built first and explicitly: common.mk's bootstrap rule
# recurses into src/ WITHOUT these flags, so an example reaching it would link
# an uninstrumented archive against instrumented objects.
#
SAN_FLAGS    ?= -fsanitize=address -fno-omit-frame-pointer
SAN_BUILD    ?= $(CURDIR)/build-asan

# The four the graph representation work was checked against: the pinned
# invariants, the misuse suite, the 2-D checkpoint overload, and the deepest
# tape in the tree.  Override to widen:  make sanitize SAN_EXAMPLES="pinn"
SAN_EXAMPLES ?= invariants regress bratu heat_assim

sanitize:
	@$(MAKE) --no-print-directory -C src \
	    BUILD_DIR="$(SAN_BUILD)" \
	    FLAGS="-g $(STD) $(WARN) -O1 $(SAN_FLAGS)"
	@for d in $(SAN_EXAMPLES); do \
	  $(MAKE) --no-print-directory -C examples/$$d all \
	      BUILD_DIR="$(SAN_BUILD)" \
	      BASE_FLAGS="-g $(STD) $(WARN) -O1 $(SAN_FLAGS)" || exit 1; \
	done
	@echo
	@for d in $(SAN_EXAMPLES); do \
	  echo "=================================================== $$d (asan+lsan)"; \
	  $(MAKE) --no-print-directory -C examples/$$d run \
	      BUILD_DIR="$(SAN_BUILD)" || exit 1; \
	done
	@echo "=================================================== sanitizers clean"

clean:
	@$(MAKE) --no-print-directory -C examples clean
	@rm -rf $(BUILD_DIR) $(SAN_BUILD)

help:
	@echo "make            build build/lib/libmaxwell.a and every example into build/bin"
	@echo "make test       build, then run the whole suite"
	@echo "make list       list the examples and the arguments they run with"
	@echo "make sanitize   rebuild into build-asan/ under ASan+LSan and run: $(SAN_EXAMPLES)"
	@echo "make clean      remove build/ and build-asan/"
