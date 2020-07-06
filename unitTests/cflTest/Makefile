# Required paths
ifndef LEGION_DIR
  $(error LEGION_DIR is not set)
endif
ifndef HTR_DIR
  $(error HTR_DIR is not set)
endif

# OS-specific options
ifeq ($(shell uname),Darwin)
  DYNLINK_PATH := DYLD_LIBRARY_PATH
else
  DYNLINK_PATH := LD_LIBRARY_PATH
endif

# CUDA options
USE_CUDA ?= 1

# HDF options
export USE_HDF ?= 1
export HDF_HEADER ?= hdf5.h
HDF_LIBNAME ?= hdf5

# C compiler options
CFLAGS += -O2 -Wall -Werror -fno-strict-aliasing -I$(LEGION_DIR)/runtime -I$(LEGION_DIR)/bindings/regent
CXXFLAGS += -std=c++11 -O3 -Wall -Werror -fno-strict-aliasing -I$(LEGION_DIR)/runtime -I$(LEGION_DIR)/bindings/regent

# Regent options
export TERRA_PATH := ?.rg;$(HTR_DIR)/src/?.rg
export INCLUDE_PATH := .
ifdef HDF_ROOT
  export INCLUDE_PATH := $(INCLUDE_PATH);$(HDF_ROOT)/include
  export $(DYNLINK_PATH) := $($(DYNLINK_PATH)):$(HDF_ROOT)/lib
endif
REGENT := $(LEGION_DIR)/language/regent.py
REGENT_FLAGS := -fflow 0 -finner 1 -fbounds-checks 1
ifeq ($(USE_CUDA), 1)
  REGENT_FLAGS += -fcuda 1 -fcuda-offline 1
else
  REGENT_FLAGS += -fcuda 0
endif
ifeq ($(USE_OPENMP), 1)
  REGENT_FLAGS += -fopenmp 1
else
  REGENT_FLAGS += -fopenmp 0
endif

# Link flags
LINK_FLAGS += -L$(LEGION_DIR)/bindings/regent -lregent
ifdef HDF_ROOT
  LINK_FLAGS += -L$(HDF_ROOT)/lib
endif
ifeq ($(USE_HDF), 1)
  LINK_FLAGS += -l$(HDF_LIBNAME)
endif
LINK_FLAGS += -lm
 
ifeq ($(DEBUG), 1)
  REGENT_FLAGS += -g
  CFLAGS += -g
  CXXFLAGS += -g
  LINK_FLAGS += -g
endif

.PHONY: default all clean

MODULES= $(HTR_DIR)/src/ConstPropMix.rg \
			$(HTR_DIR)/src/prometeo_cfl.rg

default: cflTest.exec

clean:
	$(RM) *.exec *.o

cflTest.exec: cflTest.o $(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/json.o
	$(CXX) -o $@ $^ $(LINK_FLAGS)

cflTest.o: cflTest.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/util-desugared.rg $(MODULES)
	$(REGENT) cflTest.rg $(REGENT_FLAGS)

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	make -C $(HTR_DIR)/src config_schema.o

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	make -C $(HTR_DIR)/src json.o

$(HTR_DIR)/src/util-desugared.rg: $(HTR_DIR)/src/util.rg
	make -C $(HTR_DIR)/src util-desugared.rg

