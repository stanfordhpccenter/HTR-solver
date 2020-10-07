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

# OpenMP options
USE_OPENMP ?= 1

# HDF options
export USE_HDF ?= 1
export HDF_HEADER ?= hdf5.h
HDF_LIBNAME ?= hdf5

# C compiler options
CFLAGS += -O2 -Wall -Werror -fno-strict-aliasing -I$(LEGION_DIR)/runtime -I$(LEGION_DIR)/bindings/regent
CXXFLAGS += -std=c++11 -O3 -Wall -Werror -fno-strict-aliasing -I$(LEGION_DIR)/runtime -I$(LEGION_DIR)/bindings/regent

# Regent options
export INCLUDE_PATH := .
ifdef HDF_ROOT
  export INCLUDE_PATH := $(INCLUDE_PATH);$(HDF_ROOT)/include
  export $(DYNLINK_PATH) := $($(DYNLINK_PATH)):$(HDF_ROOT)/lib
endif
REGENT := $(LEGION_DIR)/language/regent.py
REGENT_FLAGS := -fflow 0 -finner 1
ifeq ($(DEBUG), 1)
  REGENT_FLAGS += -g -fcuda 0 -fbounds-checks 1
  CFLAGS += -g
  CXXFLAGS += -g -DBOUNDS_CHECKS -DPRIVILEGE_CHECKS
  LINK_FLAGS += -g
else
ifeq ($(USE_CUDA), 1)
  REGENT_FLAGS += -fcuda 1 -fcuda-offline 1
  NVCC ?= $(CUDA_HOME)/bin/nvcc
  NVCCFLAGS += -std=c++11 -O3 -I$(LEGION_DIR)/runtime -I$(LEGION_DIR)/bindings/regent
else
  REGENT_FLAGS += -fcuda 0
endif
endif
ifeq ($(USE_OPENMP), 1)
  CFLAGS       += -fopenmp
  CXXFLAGS     += -fopenmp
  REGENT_FLAGS += -fopenmp 1
else
  REGENT_FLAGS += -fopenmp 0
endif

# Link flags
ifdef CRAYPE_VERSION
  LINK_FLAGS += -Bdynamic
  LINK_FLAGS += $(CRAY_UGNI_POST_LINK_OPTS) -lugni
  LINK_FLAGS += $(CRAY_UDREG_POST_LINK_OPTS) -ludreg
endif
LINK_FLAGS += -L$(LEGION_DIR)/bindings/regent -lregent
ifdef HDF_ROOT
  LINK_FLAGS += -L$(HDF_ROOT)/lib
endif
ifeq ($(USE_HDF), 1)
  LINK_FLAGS += -l$(HDF_LIBNAME)
endif
LINK_FLAGS += -lm
 
# Get git commit version and date
SOLVER_VERSION := ""$(shell git --no-pager describe --tags --always --dirty)" committed on "$(firstword $(shell git --no-pager show --date=short --format="%ad" --name-only))""

LEGION_VERSION := ""$(shell cd $(LEGION_DIR); git --no-pager describe --tags --always --dirty)" committed on "$(firstword $(shell cd $(LEGION_DIR); git --no-pager show --date=short --format="%ad" --name-only))""

.PHONY: default all clean force
.PRECIOUS: solver_version~ legion_version~ version.rg prometeo_main_%.o \
			prometeo_metric_%_gpu.o prometeo_metric_%_cpu.o prometeo_rhs_%_gpu.o prometeo_rhs_%_cpu.o \
			prometeo_sensor_%_gpu.o prometeo_sensor_%_cpu.o prometeo_variables_%_gpu.o prometeo_variables_%_cpu.o
.SUFFIXES: .rg .cc .cu .h .o

default: prometeo_ConstPropMix.exec prometeo_IsentropicMix.exec prometeo_AirMix.exec \
			prometeo_CH41StMix.exec

Headers= prometeo_registrar.h prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h
Common= hdf_helper.rg util-desugared.rg math_utils.rg prometeo_const.rg prometeo_macro.rg \
		prometeo_grid.rg prometeo_metric.rg prometeo_IO.rg prometeo_cfl.rg prometeo_chem.rg \
		prometeo_init.rg prometeo_variables.rg prometeo_average.rg prometeo_rk.rg \
		prometeo_sensor.rg prometeo_rhs.rg prometeo_stat.rg prometeo_bc.rg \
		prometeo_profiles.rg prometeo_partitioner.rg prometeo_probe.rg

ConstPropMixH= ConstPropMix.h
ConstPropMixHPP= $(ConstPropMixH:.h=.hpp)
ConstPropMixRG= $(ConstPropMixH:.h=.rg)

IsentropicMixH= IsentropicMix.h
IsentropicMixHPP= $(IsentropicMixH:.h=.hpp)
IsentropicMixRG= $(IsentropicMixH:.h=.rg)

AirMixH= Species.h Reaction.h AirMix.h
AirMixHPP= Species.hpp MultiComponent.hpp AirMix.hpp
AirMixRG= $(AirMixH:.h=.rg) MultiComponent.rg

CH41StMixH= Species.h Reaction.h CH41StMix.h
CH41StMixHPP= Species.hpp MultiComponent.hpp CH41StMix.hpp
CH41StMixRG= $(CH41StMixH:.h=.rg) MultiComponent.rg

clean:
	$(RM) *.exec *.o *-desugared.rg config_schema.h version.rg solver_version~ legion_version~

%-desugared.rg: %.rg
	./desugar.py $< > $@

getRecipe = $(if $(DEPENDENCY_GRAPH),@echo Target $@ depends on prerequisites "$^",$(1))

prometeo_%.exec: prometeo_main_%.o prometeo_mapper.o prometeo_registrar.o config_schema.o json.o \
						prometeo_metric_%_cpu.o $(if $(filter $(strip $(USE_CUDA)), 1), prometeo_metric_%_gpu.o) \
						prometeo_sensor_%_cpu.o $(if $(filter $(strip $(USE_CUDA)), 1), prometeo_sensor_%_gpu.o) \
						prometeo_variables_%_cpu.o $(if $(filter $(strip $(USE_CUDA)), 1), prometeo_variables_%_gpu.o) \
						prometeo_rhs_%_cpu.o $(if $(filter $(strip $(USE_CUDA)), 1), prometeo_rhs_%_gpu.o)
	$(call getRecipe)
	$(CXX) -o $@ $^ $(LINK_FLAGS)

.SECONDEXPANSION:
prometeo_main_%.o: version.rg prometeo_types.h prometeo-desugared.rg prometeo_mapper.h config_schema.o \
						config_schema.o $$($$*RG) $(Headers) $(Common)
	$(call getRecipe)
	EOS="$*" $(REGENT) prometeo-desugared.rg $(REGENT_FLAGS)

.SECONDEXPANSION:
prometeo_metric_%_cpu.o: prometeo_metric.cc prometeo_metric.h prometeo_metric.hpp prometeo_metric.inl task_helper.h \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h config_schema.o
	$(call getRecipe)
	$(CXX) $(CXXFLAGS) -DEOS="$*" -c -o  $@ $<

.SECONDEXPANSION:
prometeo_metric_%_gpu.o: prometeo_metric.cu prometeo_metric.h prometeo_metric.hpp prometeo_metric.inl task_helper.h cuda_utils.hpp \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h config_schema.o
	$(call getRecipe)
	$(NVCC) $(NVCCFLAGS) -DEOS="$*" -c -o  $@ $<

.SECONDEXPANSION:
prometeo_variables_%_cpu.o: prometeo_variables.cc prometeo_variables.h prometeo_variables.hpp prometeo_metric.inl task_helper.h \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h config_schema.o
	$(call getRecipe)
	$(CXX) $(CXXFLAGS) -DEOS="$*" -c -o  $@ $<

.SECONDEXPANSION:
prometeo_variables_%_gpu.o: prometeo_variables.cu prometeo_variables.h prometeo_variables.hpp prometeo_metric.inl task_helper.h cuda_utils.hpp \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h config_schema.o
	$(call getRecipe)
	$(NVCC) $(NVCCFLAGS) -DEOS="$*" -c -o  $@ $<

.SECONDEXPANSION:
prometeo_sensor_%_cpu.o: prometeo_sensor.cc prometeo_sensor.h prometeo_sensor.hpp prometeo_sensor.inl prometeo_metric.inl \
						task_helper.h prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h config_schema.o
	$(call getRecipe)
	$(CXX) $(CXXFLAGS) -DEOS="$*" -c -o  $@ $<

.SECONDEXPANSION:
prometeo_sensor_%_gpu.o: prometeo_sensor.cu prometeo_sensor.h prometeo_sensor.hpp prometeo_sensor.inl prometeo_metric.inl \
						task_helper.h cuda_utils.hpp prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h config_schema.o
	$(call getRecipe)
	$(NVCC) $(NVCCFLAGS) -DEOS="$*" -c -o  $@ $<

.SECONDEXPANSION:
prometeo_rhs_%_cpu.o: prometeo_rhs.cc prometeo_rhs.h prometeo_rhs.hpp prometeo_rhs.inl task_helper.h math_utils.hpp \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h prometeo_metric.inl config_schema.o \
						$$($$*H) $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXXFLAGS) -DEOS="$*" -c -o  $@ $<

.SECONDEXPANSION:
prometeo_rhs_%_gpu.o: prometeo_rhs.cu prometeo_rhs.h prometeo_rhs.hpp prometeo_rhs.inl task_helper.h math_utils.hpp cuda_utils.hpp \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h prometeo_metric.inl config_schema.o \
						$$($$*H) $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCCFLAGS) -DEOS="$*" -c -o  $@ $<

prometeo_registrar.o: prometeo_registrar.cc prometeo_registrar.h prometeo_mapper.h prometeo_metric.h prometeo_rhs.h
	$(call getRecipe)
	$(CXX) $(CXXFLAGS) -c -o  $@ $<

prometeo_mapper.o: prometeo_mapper.cc prometeo_mapper.h config_schema.o
	$(call getRecipe)
	$(CXX) $(CXXFLAGS) -c -o  $@ $<

config_schema.o: process_schema.rg config_schema.lua json.h util-desugared.rg
	$(call getRecipe)
	$(REGENT) process_schema.rg config_schema.lua $(REGENT_FLAGS)

json.o: json.c json.h
	$(call getRecipe)
	$(CC) $(CFLAGS) -c -o $@ $<

# recompile version.h dependants when GIT_VERSION changes, uses temporary file version~
solver_version~: force
	@echo "$(SOLVER_VERSION)" | cmp -s - $@ || echo "$(SOLVER_VERSION)" > $@

legion_version~: force
	@echo "$(LEGION_VERSION)" | cmp -s - $@ || echo "$(LEGION_VERSION)" > $@

version.rg: solver_version~ legion_version~
	@touch $@
	@echo Solver version $(SOLVER_VERSION)
	@echo Legion version $(LEGION_VERSION)
	@echo "-- This file is automatically generated, please do not touch it"                > $@
	@echo ""                                                                              >> $@
	@echo "local Exports = {}"                                                            >> $@
	@echo ""                                                                              >> $@
	@echo "local SolverLength = "$(shell echo $(SOLVER_VERSION) | awk "{print length}")"" >> $@
	@echo "Exports.SolverVersion = \""$(SOLVER_VERSION)"\""                               >> $@
	@echo ""                                                                              >> $@
	@echo "local LegionLength = "$(shell echo $(LEGION_VERSION) | awk "{print length}")"" >> $@
	@echo "Exports.LegionVersion = \""$(LEGION_VERSION)"\""                               >> $@
	@echo ""                                                                              >> $@
	@echo "Exports.Length = math.max(SolverLength, LegionLength)"                         >> $@
	@echo ""                                                                              >> $@
	@echo "return Exports"                                                                >> $@
	@echo ""                                                                              >> $@

