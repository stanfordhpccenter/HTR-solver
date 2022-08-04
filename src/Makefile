#######################################################
# Include standard variables
#######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Get git commit version and date
######################################################
SOLVER_VERSION := ""$(shell git --no-pager describe --tags --always --dirty)" committed on "$(firstword $(shell git --no-pager show --date=short --format="%ad" --name-only))""

LEGION_VERSION := ""$(shell cd $(LEGION_DIR); git --no-pager describe --tags --always --dirty)" committed on "$(firstword $(shell cd $(LEGION_DIR); git --no-pager show --date=short --format="%ad" --name-only))""

######################################################
# Mixtures
######################################################
ConstPropMixH= config_schema.o Utils/constants.h Mixtures/ConstPropMix.hpp
ConstPropMixHPP= Utils/my_array.hpp $(ConstPropMixH:.hpp=.inl) $(ConstPropMixH)

IsentropicMixH= config_schema.o Utils/constants.h Mixtures/IsentropicMix.hpp
IsentropicMixHPP= Utils/my_array.hpp $(IsentropicMixH:.hpp=.inl) $(IsentropicMixH)

AirMixH= Utils/constants.h Mixtures/Species.hpp Mixtures/Reaction.hpp Mixtures/MultiComponent.hpp config_schema.o Mixtures/AirMix.hpp
AirMixHPP= Utils/my_array.hpp Mixtures/Species.inl Mixtures/Reaction.inl Mixtures/MultiComponent.inl $(AirMixH)

CH41StMixH= Utils/constants.h Mixtures/Species.hpp Mixtures/Reaction.hpp Mixtures/MultiComponent.hpp config_schema.o Mixtures/CH41StMix.hpp
CH41StMixHPP= Utils/my_array.hpp Mixtures/Species.inl Mixtures/Reaction.inl Mixtures/MultiComponent.inl $(CH41StMixH)

CH4_30SpMixH= Utils/constants.h Mixtures/Species.hpp Mixtures/Reaction.hpp Mixtures/MultiComponent.hpp config_schema.o Mixtures/CH4_30SpMix.hpp
CH4_30SpMixHPP= Utils/my_array.hpp Mixtures/Species.inl Mixtures/Reaction.inl Mixtures/MultiComponent.inl $(CH4_30SpMixH)

CH4_43SpIonsMixH= Utils/constants.h Mixtures/Species.hpp Mixtures/Reaction.hpp Mixtures/MultiComponent.hpp config_schema.o Mixtures/CH4_43SpIonsMix.hpp
CH4_43SpIonsMixHPP= Utils/my_array.hpp Mixtures/Species.inl Mixtures/Reaction.inl Mixtures/MultiComponent.inl $(CH4_43SpIonsMixH)

CH4_26SpIonsMixH= Utils/constants.h Mixtures/Species.hpp Mixtures/Reaction.hpp Mixtures/MultiComponent.hpp config_schema.o Mixtures/CH4_26SpIonsMix.hpp
CH4_26SpIonsMixHPP= Utils/my_array.hpp Mixtures/Species.inl Mixtures/Reaction.inl Mixtures/MultiComponent.inl $(CH4_26SpIonsMixH)

FFCM1MixH= Utils/constants.h Mixtures/Species.hpp Mixtures/Reaction.hpp Mixtures/MultiComponent.hpp config_schema.o Mixtures/FFCM1Mix.hpp
FFCM1MixHPP= Utils/my_array.hpp Mixtures/Species.inl Mixtures/Reaction.inl Mixtures/MultiComponent.inl $(FFCM1MixH)

BoivinMixH= Utils/constants.h Mixtures/Species.hpp Mixtures/Reaction.hpp Mixtures/MultiComponent.hpp config_schema.o Mixtures/BoivinMix.hpp
BoivinMixHPP= Utils/my_array.hpp Mixtures/Species.inl Mixtures/Reaction.inl Mixtures/MultiComponent.inl $(BoivinMixH)

H2_UCSDMixH= Utils/constants.h Mixtures/Species.hpp Mixtures/Reaction.hpp Mixtures/MultiComponent.hpp config_schema.o Mixtures/H2_UCSDMix.hpp
H2_UCSDMixHPP= Utils/my_array.hpp Mixtures/Species.inl Mixtures/Reaction.inl Mixtures/MultiComponent.inl $(H2_UCSDMixH)

######################################################
# Rules
######################################################
.PHONY: default all clean force

.PRECIOUS: solver_version~ legion_version~ version.rg prometeo_main_%.o \
			prometeo_%_gpu.a prometeo_%_gpu.o prometeo_%_cpu.a \
			prometeo_mixture_%_gpu.o prometeo_mixture_%_cpu.o \
			prometeo_metric_%_gpu.o prometeo_metric_%_cpu.o \
			prometeo_rhs_%_gpu.o prometeo_rhs_%_cpu.o \
			prometeo_sensor_%_gpu.o prometeo_sensor_%_cpu.o \
			prometeo_variables_%_gpu.o prometeo_variables_%_cpu.o \
			prometeo_chem_%_gpu.o prometeo_chem_%_cpu.o\
			prometeo_bc_%_gpu.o prometeo_bc_%_cpu.o \
			prometeo_average_%_gpu.o prometeo_average_%_cpu.o \
			prometeo_cfl_%_gpu.o prometeo_cfl_%_cpu.o \
			Poisson/Poisson_%_cpu.o Poisson/Poisson_%_gpu.o \
			prometeo_electricField_%_cpu.o prometeo_electricField_%_gpu.o \
			prometeo_laser_%_gpu.o prometeo_laser_%_cpu.o \

.SUFFIXES: .rg .cc .cu .h .o .a

default: prometeo_ConstPropMix.exec prometeo_IsentropicMix.exec prometeo_AirMix.exec \
			prometeo_CH41StMix.exec prometeo_CH4_30SpMix.exec prometeo_FFCM1Mix.exec \
			$(if $(filter $(strip $(ELECTRIC_FIELD)), 1), prometeo_CH4_43SpIonsMix.exec, prometeo_CH4_26SpIonsMix.exec)

######################################################
# Objects
######################################################
Headers= prometeo_registrar.h prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h prometeo_bc_types.h \
			 prometeo_average_types.h prometeo_mixture_wrappers.hpp Utils/math_utils.h

Utils = Utils/task_helper.hpp Utils/my_array.hpp Utils/PointDomain_helper.hpp

MathUtils = Utils/math_utils.h Utils/math_utils.hpp

Common= Utils/hdf_helper.rg Utils/util.rg Utils/math_utils.rg prometeo_const.rg prometeo_macro.rg \
		prometeo_grid.rg prometeo_metric.rg prometeo_IO.rg prometeo_cfl.rg prometeo_chem.rg \
		prometeo_init.rg prometeo_variables.rg prometeo_average.rg prometeo_rk.rg \
		prometeo_sensor.rg prometeo_rhs.rg prometeo_stat.rg prometeo_bc.rg prometeo_mixture.rg \
		prometeo_profiles.rg prometeo_partitioner.rg prometeo_probe.rg prometeo_laser.rg

ifeq ($(ELECTRIC_FIELD), 1)
Headers+= Poisson/Poisson.h prometeo_electricField.h
Common+= Poisson/Poisson.rg prometeo_electricField.rg
endif

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o *.a config_schema.h version.rg solver_version~ legion_version~
	$(RM) Mixtures/*.o Poisson/*.o

getRecipe = $(if $(DEPENDENCY_GRAPH),@echo Target $@ depends on prerequisites "$^",$(1))

prometeo_%.exec: prometeo_main_%.o prometeo_%_cpu.a $(if $(filter $(strip $(USE_CUDA)), 1), prometeo_%_gpu.a)
	$(call getRecipe)
	$(CXX) -o $@ $^ $(LINK_FLAGS)

prometeo_%_cpu.a: prometeo_mapper.o prometeo_registrar.o config_schema.o json.o \
						prometeo_metric_coeffs_cpu.o prometeo_mixture_%_cpu.o prometeo_metric_%_cpu.o prometeo_sensor_%_cpu.o \
						prometeo_variables_%_cpu.o prometeo_rhs_%_cpu.o prometeo_chem_%_cpu.o prometeo_bc_%_cpu.o \
						prometeo_average_%_cpu.o prometeo_cfl_%_cpu.o \
						$(if $(filter $(strip $(ELECTRIC_FIELD)), 1), prometeo_electricField_%_cpu.o Poisson/Poisson_%_cpu.o)
	$(call getRecipe)
	$(AR) rcs $@ $^

prometeo_%_gpu.a: cuda_utils.o \
						prometeo_metric_coeffs_gpu.o prometeo_mixture_%_gpu.o prometeo_metric_%_gpu.o prometeo_sensor_%_gpu.o \
						prometeo_variables_%_gpu.o prometeo_rhs_%_gpu.o prometeo_chem_%_gpu.o prometeo_bc_%_gpu.o \
						prometeo_average_%_gpu.o prometeo_cfl_%_gpu.o \
						$(if $(filter $(strip $(ELECTRIC_FIELD)), 1), prometeo_electricField_%_gpu.o Poisson/Poisson_%_gpu.o)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -dlink $^ -o $(@:.a=.o)
	$(NVCC) -lib -o $@ $^ $(@:.a=.o)

.SECONDEXPANSION:
prometeo_main_%.o: version.rg prometeo_types.h prometeo.rg prometeo_mapper.h $$($$*H) $(Headers) $(Common)
	$(call getRecipe)
	EOS="$*" $(REGENT) $(REGENT_OPTS) prometeo.rg $(REGENT_FLAGS)

.SECONDEXPANSION:
prometeo_mixture_%_cpu.o: prometeo_mixture.cc prometeo_mixture.h prometeo_mixture.hpp prometeo_mixture_wrappers.hpp $(Utils) \
						prometeo_const.h prometeo_types.h $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -DEOS="$*" -c -o $@ $<

.SECONDEXPANSION:
prometeo_mixture_%_gpu.o: prometeo_mixture.cu prometeo_mixture.h prometeo_mixture.hpp prometeo_mixture_wrappers.hpp $(Utils) Utils/cuda_utils.hpp \
						prometeo_const.h prometeo_types.h $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -DEOS="$*" -dc -o $@ $<

.SECONDEXPANSION:
prometeo_metric_%_cpu.o: prometeo_metric.cc prometeo_metric.h prometeo_metric.hpp prometeo_metric.inl $(Utils) \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -DEOS="$*" -c -o $@ $<

.SECONDEXPANSION:
prometeo_metric_%_gpu.o: prometeo_metric.cu prometeo_metric.h prometeo_metric.hpp prometeo_metric.inl $(Utils) Utils/cuda_utils.hpp \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -DEOS="$*" -dc -o $@ $<

.SECONDEXPANSION:
prometeo_variables_%_cpu.o: prometeo_variables.cc prometeo_variables.h prometeo_variables.hpp prometeo_metric.inl $(Utils) \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -DEOS="$*" -c -o $@ $<

.SECONDEXPANSION:
prometeo_variables_%_gpu.o: prometeo_variables.cu prometeo_variables.h prometeo_variables.hpp prometeo_metric.inl $(Utils) Utils/cuda_utils.hpp \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -DEOS="$*" -dc -o $@ $<

.SECONDEXPANSION:
prometeo_sensor_%_cpu.o: prometeo_sensor.cc prometeo_sensor.h prometeo_sensor.hpp prometeo_sensor.inl prometeo_metric.inl \
						$(Utils) prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -DEOS="$*" -c -o $@ $<

.SECONDEXPANSION:
prometeo_sensor_%_gpu.o: prometeo_sensor.cu prometeo_sensor.h prometeo_sensor.hpp prometeo_sensor.inl prometeo_metric.inl \
						$(Utils) Utils/cuda_utils.hpp prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -DEOS="$*" -dc -o $@ $<

.SECONDEXPANSION:
prometeo_rhs_%_cpu.o: prometeo_rhs.cc prometeo_rhs.h prometeo_rhs.hpp prometeo_rhs.inl $(Utils) $(MathUtils) \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h prometeo_metric.inl $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -DEOS="$*" -c -o $@ $<

.SECONDEXPANSION:
prometeo_rhs_%_gpu.o: prometeo_rhs.cu prometeo_rhs.h prometeo_rhs.hpp prometeo_rhs.inl $(Utils) $(MathUtils) Utils/cuda_utils.hpp \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h prometeo_metric.inl $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -DEOS="$*" -dc -o $@ $<

.SECONDEXPANSION:
prometeo_chem_%_cpu.o: prometeo_chem.cc prometeo_chem.h prometeo_chem.hpp $(Utils) $(MathUtils) \
						prometeo_const.h prometeo_types.h $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -DEOS="$*" -c -o $@ $<

.SECONDEXPANSION:
prometeo_chem_%_gpu.o: prometeo_chem.cu prometeo_chem.h prometeo_chem.hpp $(Utils) $(MathUtils) Utils/cuda_utils.hpp \
						prometeo_const.h prometeo_types.h $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -DEOS="$*" -dc -o $@ $<

.SECONDEXPANSION:
prometeo_bc_%_cpu.o: prometeo_bc.cc prometeo_bc.h prometeo_bc.hpp prometeo_bc_types.h prometeo_variables.hpp prometeo_redop.inl \
						$(Utils) $(MathUtils) prometeo_const.h prometeo_types.h $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -DEOS="$*" -c -o $@ $<

.SECONDEXPANSION:
prometeo_bc_%_gpu.o: prometeo_bc.cu prometeo_bc.h prometeo_bc.hpp prometeo_bc_types.h prometeo_variables.hpp prometeo_redop.inl \
						$(Utils) $(MathUtils) Utils/cuda_utils.hpp prometeo_const.h prometeo_types.h $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -DEOS="$*" -dc -o $@ $<

.SECONDEXPANSION:
prometeo_average_%_cpu.o: prometeo_average.cc prometeo_average.h prometeo_average.hpp prometeo_average.inl prometeo_average_types.h \
						prometeo_redop.inl $(Utils) prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h prometeo_metric.inl $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -DEOS="$*" -c -o $@ $<

.SECONDEXPANSION:
prometeo_average_%_gpu.o: prometeo_average.cu prometeo_average.h prometeo_average.hpp prometeo_average.inl prometeo_average_types.h \
						prometeo_redop.inl $(Utils) Utils/cuda_utils.hpp prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h prometeo_metric.inl $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -DEOS="$*" -dc -o $@ $<

.SECONDEXPANSION:
prometeo_cfl_%_cpu.o: prometeo_cfl.cc prometeo_cfl.h prometeo_cfl.hpp $(Utils) prometeo_const.h prometeo_types.h $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -DEOS="$*" -c -o $@ $<

.SECONDEXPANSION:
prometeo_cfl_%_gpu.o: prometeo_cfl.cu prometeo_cfl.h prometeo_cfl.hpp $(Utils) Utils/cuda_utils.hpp \
						prometeo_const.h prometeo_types.h $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -DEOS="$*" -dc -o $@ $<

prometeo_metric_coeffs_cpu.o: prometeo_metric_coeffs.cc prometeo_metric_coeffs.h
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

prometeo_metric_coeffs_gpu.o: prometeo_metric_coeffs.cu prometeo_metric_coeffs.h
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $<

.SECONDEXPANSION:
Poisson/Poisson_%_cpu.o: Poisson/Poisson.cc Poisson/Poisson.h Poisson/Poisson.hpp $(Utils) \
						prometeo_const.h prometeo_types.h $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -DEOS="$*" -c -o  $@ $<

.SECONDEXPANSION:
Poisson/Poisson_%_gpu.o: Poisson/Poisson.cu Poisson/Poisson.h Poisson/Poisson.hpp  $(Utils) Utils/cuda_utils.hpp \
						prometeo_const.h prometeo_types.h $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -DEOS="$*" -dc -o  $@ $<

.SECONDEXPANSION:
prometeo_electricField_%_cpu.o: prometeo_electricField.cc prometeo_electricField.h prometeo_electricField.hpp prometeo_electricField.inl \
						Poisson/Poisson.h Poisson/Poisson.hpp $(Utils) \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h prometeo_metric.inl $$($$*HPP)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -DEOS="$*" -c -o  $@ $<

.SECONDEXPANSION:
prometeo_electricField_%_gpu.o: prometeo_electricField.cu prometeo_electricField.h prometeo_electricField.hpp prometeo_electricField.inl \
						Poisson/Poisson.h Poisson/Poisson.hpp $(Utils) Utils/cuda_utils.hpp \
						prometeo_const.h prometeo_types.h prometeo_metric_coeffs.h prometeo_metric.inl $$($$*HPP)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -DEOS="$*" -dc -o  $@ $<

prometeo_registrar.o: prometeo_registrar.cc prometeo_registrar.h \
							prometeo_mapper.h prometeo_variables.h prometeo_mixture.h \
							prometeo_metric.h prometeo_sensor.h prometeo_chem.h \
							prometeo_bc.h prometeo_rhs.h prometeo_cfl.h prometeo_average.h \
							$(if $(filter $(strip $(ELECTRIC_FIELD)), 1), prometeo_electricField.h Poisson/Poisson.h)
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

prometeo_mapper.o: prometeo_mapper.cc prometeo_mapper.h config_schema.o
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

config_schema.o: Utils/process_schema.rg config_schema.lua json.h Utils/util.rg
	$(call getRecipe)
	$(REGENT) $(REGENT_OPTS) $< config_schema.lua $(REGENT_FLAGS)

json.o: json.c json.h
	$(call getRecipe)
	$(CC) $(C_FLAGS) -c -o $@ $<

cuda_utils.o: Utils/cuda_utils.cu Utils/cuda_utils.hpp $(Utils)
	$(call getRecipe)
	$(NVCC) $(NVCC_FLAGS) -dc -o  $@ $<

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

