######################################################
# Include standard variables
######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean force

default: variablesTest.exec

######################################################
# Objects
######################################################
MODULES= $(HTR_DIR)/src/prometeo_const.rg \
			$(HTR_DIR)/src/prometeo_macro.rg \
			$(HTR_DIR)/src/prometeo_metric.rg \
			$(HTR_DIR)/src/prometeo_mixture.rg \
			$(HTR_DIR)/src/prometeo_partitioner.rg \
			$(HTR_DIR)/src/prometeo_variables.rg \
			$(HTR_DIR)/src/prometeo_variables.h

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o *.a

variablesTest.exec: variablesTest.o registrar.o libtest.a
	$(CXX) -o $@ $^ $(LINK_FLAGS)

variablesTest.o: variablesTest.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/Utils/util.rg $(MODULES)
	EOS=AirMix $(REGENT) $(REGENT_OPTS) variablesTest.rg $(REGENT_FLAGS)

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/Utils/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	make -C $(HTR_DIR)/src config_schema.o

libtest.a: $(HTR_DIR)/src/prometeo_AirMix_cpu.a \
			  $(if $(filter $(strip $(USE_CUDA)), 1),  $(HTR_DIR)/src/prometeo_AirMix_gpu.a)
	$(AR) rcT $@ $^

registrar.o: registrar.cc registrar.h $(HTR_DIR)/src/prometeo_variables.h $(HTR_DIR)/src/prometeo_mixture.h
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(HTR_DIR)/src/prometeo_AirMix_cpu.a: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_AirMix_cpu.a

$(HTR_DIR)/src/prometeo_AirMix_gpu.a: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_AirMix_gpu.a

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	make -C $(HTR_DIR)/src json.o

