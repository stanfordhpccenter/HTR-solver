######################################################
# Include standard variables
######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean force

default: averageTest.exec average1DTest.exec

######################################################
# Objects
######################################################
MODULES= $(HTR_DIR)/src/Utils/hdf_helper.rg \
			$(HTR_DIR)/src/prometeo_IO.rg \
			$(HTR_DIR)/src/prometeo_const.rg \
			$(HTR_DIR)/src/prometeo_macro.rg \
			$(HTR_DIR)/src/prometeo_metric.rg \
			$(HTR_DIR)/src/prometeo_partitioner.rg \
			$(HTR_DIR)/src/prometeo_mixture.rg \
			$(HTR_DIR)/src/prometeo_average.rg \
			$(HTR_DIR)/src/prometeo_average_types.h

######################################################
# Recipes
######################################################
clean:
	$(RM) -r *.exec *.o *.a *Averages

average1DTest.exec: average1DTest.o registrar.o libtest.a
	$(CXX) -o $@ $^ $(LINK_FLAGS)

average1DTest.o: average1DTest.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/Utils/util.rg $(MODULES)
	EOS="AirMix" $(REGENT) $(REGENT_OPTS) average1DTest.rg $(REGENT_FLAGS)

averageTest.exec: averageTest.o registrar.o libtest.a
	$(CXX) -o $@ $^ $(LINK_FLAGS)

averageTest.o: averageTest.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/Utils/util.rg $(MODULES)
	EOS="AirMix" $(REGENT) $(REGENT_OPTS) averageTest.rg $(REGENT_FLAGS)

libtest.a: $(HTR_DIR)/src/prometeo_AirMix_cpu.a \
			  $(if $(filter $(strip $(USE_CUDA)), 1),  $(HTR_DIR)/src/prometeo_AirMix_gpu.a)
	$(AR) rcT $@ $^

registrar.o: registrar.cc registrar.h $(HTR_DIR)/src/prometeo_average.h $(HTR_DIR)/src/prometeo_mixture.h
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(HTR_DIR)/src/prometeo_AirMix_cpu.a: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_AirMix_cpu.a

$(HTR_DIR)/src/prometeo_AirMix_gpu.a: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_AirMix_gpu.a

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/Utils/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	$(MAKE) -C $(HTR_DIR)/src config_schema.o

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	$(MAKE) -C $(HTR_DIR)/src json.o

