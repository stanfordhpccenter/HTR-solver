######################################################
# Include standard variables
######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean force

default: poissonTest.exec

######################################################
# Objects
######################################################
MODULES= $(HTR_DIR)/src/version.rg \
			$(HTR_DIR)/src/prometeo_const.rg \
			$(HTR_DIR)/src/prometeo_macro.rg \
			$(HTR_DIR)/src/prometeo_partitioner.rg \
			$(HTR_DIR)/src/prometeo_grid.rg \
			$(HTR_DIR)/src/prometeo_metric.rg \
			$(HTR_DIR)/src/Poisson/Poisson.rg

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o *.a
	$(RM) -rf PoissonDir

poissonTest.exec: poissonTest.o registrar.o libtest.a
	$(CXX) -o $@ $^ $(LINK_FLAGS)

poissonTest.o: poissonTest.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/Utils/util.rg $(MODULES)
	EOS="IsentropicMix" $(REGENT) $(REGENT_OPTS) $< $(REGENT_FLAGS)

libtest.a: $(HTR_DIR)/src/prometeo_IsentropicMix_cpu.a \
			  $(if $(filter $(strip $(USE_CUDA)), 1),  $(HTR_DIR)/src/prometeo_IsentropicMix_gpu.a)
	$(AR) rcT $@ $^

registrar.o: registrar.cc registrar.h $(HTR_DIR)/src/prometeo_metric.h $(HTR_DIR)/src/prometeo_mixture.h \
				 $(HTR_DIR)/src/Poisson/Poisson.h
	$(call getRecipe)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(HTR_DIR)/src/prometeo_IsentropicMix_cpu.a: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_IsentropicMix_cpu.a

$(HTR_DIR)/src/prometeo_IsentropicMix_gpu.a: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_IsentropicMix_gpu.a

$(HTR_DIR)/src/version.rg:
	 $(MAKE) -C $(HTR_DIR)/src version.rg

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/Utils/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	$(MAKE) -C $(HTR_DIR)/src config_schema.o

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	$(MAKE) -C $(HTR_DIR)/src json.o

