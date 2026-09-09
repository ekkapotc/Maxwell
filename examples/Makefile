#
# Every subdirectory that contains a Makefile is an example.  Drop a new
# directory in with a main.cpp and a Makefile that includes ../common.mk, and
# `make`, `make clean` and `make test` pick it up with no edit here.  Remove
# one and it disappears just as quietly.  (This replaces the hand-maintained
# WORKING list, which had drifted: it named seven STALE directories that were
# no longer on disk at all.)
#
# Each example names its own arguments in its RUNARGS; `make list` prints them.
# Binaries are built into $(BUILD_DIR)/bin, one per directory.
#
EXAMPLES_ALL := $(sort $(patsubst %/,%,$(dir $(wildcard */Makefile))))

# The three assertion suites are seconds where the rest are minutes, so they go
# first when present.  invariants leads: it is the one that says whether the
# library still computes the same thing, so a failure there explains the rest.
# Everything else keeps alphabetical order.
FAST         := invariants fdcheck regress nobreak budget
# $(filter) would return them in EXAMPLES_ALL's (alphabetical) order, not in
# FAST's, so walk FAST explicitly to keep the order written above.
EXAMPLES     := $(foreach f,$(FAST),$(filter $(f),$(EXAMPLES_ALL))) $(filter-out $(FAST),$(EXAMPLES_ALL))

all clean:
	@for d in $(EXAMPLES); do $(MAKE) --no-print-directory -C $$d $@ || exit 1; done

test: all
	@for d in $(EXAMPLES); do \
	  echo "=================================================== $$d"; \
	  $(MAKE) --no-print-directory -C $$d run || exit 1; \
	done
	@echo "=================================================== all examples passed"

list:
	@for d in $(EXAMPLES); do \
	  printf "  %-10s %s\n" "$$d" "`$(MAKE) --no-print-directory -C $$d echo-args`"; \
	done

.PHONY: all clean test list
