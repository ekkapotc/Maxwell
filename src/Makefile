#
# The library.  Objects and archive land under $(BUILD_DIR), which the
# top-level Makefile exports; the fallback here keeps `cd src && make` working
# on its own.
#
INC_DIR    = ../inc
CXX       ?= g++

# ?= so that the top-level Makefile's FLAGS= override still wins, and so that
# `make FLAGS=...` and an exported FLAGS do too.  The warnings are on by
# default now: the library is clean under them, and it used to be that only the
# examples were compiled with any (-Wall -Wextra, via examples/common.mk), so a
# standalone `cd src && make` could introduce a warning that nothing reported.
FLAGS     ?= -g -std=gnu++0x -Wall -Wextra
#FLAGS     = -g -fno-elide-constructors

BUILD_DIR ?= $(abspath ..)/build
OBJ_DIR    = $(BUILD_DIR)/obj/src
LIB_DIR    = $(BUILD_DIR)/lib
LIBRARY    = $(LIB_DIR)/libmaxwell.a

#
# Every header, not just Typedefs.hpp.
#
# This was `HEADERS = $(INC_DIR)/Typedefs.hpp`, and with the pattern rule below
# naming only the object's OWN header on top of that, most of the include graph
# was invisible to make:
#
#   - touching inc/Tape.hpp rebuilt NOTHING, though API.cpp includes it and
#     defines Tape's constructor and destructor.  A change to the layout of a
#     Tape would have produced a silently mismatched library;
#   - touching inc/Vertex.hpp rebuilt Vertex.o alone, though Process.cpp,
#     Edge.cpp and API.cpp all pull Vertex.hpp in -- and Process.cpp compiles
#     sizeof(Vertex) into VERTEX_BYTES, the constant the memory budget is spent
#     against.  Exactly the kind of stale object that reads as a mysterious
#     change in chunk counts.
#
# A wildcard over inc/ is coarse -- any header change rebuilds all six objects
# -- but six objects is under two seconds, and the alternative (-MMD -MP depend
# files) is a bigger change to a build that is otherwise doing its job.  Coarse
# and correct beats precise and wrong here.
#
HEADERS    = $(wildcard $(INC_DIR)/*.hpp)
NAMES      = API Active Vertex Edge Process BreakException
OBJECTS    = $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(NAMES)))

all: $(LIBRARY)

$(LIBRARY): $(OBJECTS) | $(LIB_DIR)
	@ar -crs $@ $(OBJECTS)
	@echo "  AR  $@"

# $(INC_DIR)/%.hpp is gone from the prerequisites: $(HEADERS) already contains
# it, and naming it separately implied a per-object precision the rule never had.
$(OBJ_DIR)/%.o: %.cpp $(HEADERS) | $(OBJ_DIR)
	@echo "  CXX $<"
	@$(CXX) $(FLAGS) -c $< -o $@ -I$(INC_DIR)

$(OBJ_DIR) $(LIB_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(OBJ_DIR) $(LIBRARY)

.PHONY: all clean
