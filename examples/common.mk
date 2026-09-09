#
# Shared build rules for every example.  An example directory needs nothing but
# main.cpp and a Makefile of one to four lines:
#
#     RUNARGS     = 20 1          # what `make run` passes to the binary
#     EXTRA_FLAGS = -O2           # optional
#     OMP         = -fopenmp      # optional
#     include ../common.mk
#
# ../Makefile discovers the directory automatically -- there is no list to
# update when an example is added or removed.
#
# Nothing is written into the example directory.  The object goes to
# $(BUILD_DIR)/obj/examples/<name>.o and the binary to $(BUILD_DIR)/bin/<name>,
# named after the directory, so the six binaries can share one bin/ instead of
# all being called "main".
#
NAME       := $(notdir $(CURDIR))

ROOT_DIR   := $(abspath ../..)

# Exported by the top-level Makefile; the fallback keeps a bare
# `cd examples/pinn && make` working.
BUILD_DIR  ?= $(ROOT_DIR)/build
OBJ_DIR     = $(BUILD_DIR)/obj/examples
BIN_DIR     = $(BUILD_DIR)/bin
LIB_DIR     = $(BUILD_DIR)/lib
LIBRARY     = $(LIB_DIR)/libmaxwell.a

# The same hole src/Makefile had: an example's object depended on its main.cpp
# and on the archive, so editing maxwell.hpp or a header under inc/ rebuilt the
# example only if the library happened to be re-archived as well.  Editing a
# header no library source includes -- or editing one while the library was
# already up to date -- left every example object stale.
API_HEADERS = $(ROOT_DIR)/maxwell.hpp $(wildcard $(ROOT_DIR)/inc/*.hpp)

PROG        = $(BIN_DIR)/$(NAME)
OBJ         = $(OBJ_DIR)/$(NAME).o

# CXX, not CC: make predefines CC as "cc", so "CC ?= g++" silently does nothing
# and the examples link with the C driver -- no libstdc++, and a page of
# undefined references to std::string and __cxa_end_catch.  CXX is predefined
# as g++, and a command-line override still wins.
CXX        ?= g++
BASE_FLAGS ?= -g -std=gnu++0x -Wall -Wextra
EXTRA_FLAGS ?=
FLAGS       = $(BASE_FLAGS) $(EXTRA_FLAGS)
LIBS        = -L$(LIB_DIR) -lmaxwell
OMP        ?=
RUNARGS    ?=

all: $(PROG)

$(PROG): $(OBJ) $(LIBRARY) | $(BIN_DIR)
	@echo "  LD  $@"
	@$(CXX) $(FLAGS) -o $@ $(OBJ) $(LIBS) $(OMP)

$(OBJ): main.cpp $(API_HEADERS) $(LIBRARY) | $(OBJ_DIR)
	@echo "  CXX $(NAME)/main.cpp"
	@$(CXX) $(FLAGS) -c $< -o $@ $(OMP)

# So that a single example builds from a cold tree without going via the top.
$(LIBRARY):
	@$(MAKE) --no-print-directory -C $(ROOT_DIR)/src

$(OBJ_DIR) $(BIN_DIR):
	@mkdir -p $@

run: all
	$(PROG) $(RUNARGS)

echo-args:
	@echo "$(NAME) $(RUNARGS)"

# Only this example's own artefacts, plus anything it wrote into its own
# directory at run time.  `make clean` at the top removes all of build/.
clean:
	@rm -rf $(OBJ) $(PROG) test.*.er output*.txt job.sh core* *.btr *.o main

.PHONY: all clean run echo-args
