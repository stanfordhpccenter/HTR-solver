#######################################################
# Include standard variables
#######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean force

default: geometryTest.exec

######################################################
# Objects
######################################################
MODULES= $(HTR_DIR)/src/version.rg \
			$(HTR_DIR)/src/prometeo_const.rg \
			$(HTR_DIR)/src/prometeo_macro.rg \
			$(HTR_DIR)/src/prometeo_metric.rg \
			$(HTR_DIR)/src/prometeo_partitioner.rg \
			$(HTR_DIR)/src/prometeo_grid.rg

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o *.a
	$(RM) -rf PeriodicDir* CollocatedDir* StaggeredDir*

geometryTest.exec: geometryTest.o $(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/json.o libtest.a
	$(CXX) -o $@ $^ $(LINK_FLAGS)

geometryTest.o: geometryTest.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/Utils/util.rg $(MODULES)
	EOS=ConstPropMix $(REGENT) $(REGENT_OPTS) geometryTest.rg $(REGENT_FLAGS)

libtest.a: $(HTR_DIR)/src/prometeo_metric_coeffs_cpu.o \
			  $(if $(filter $(strip $(USE_CUDA)), 1),  $(HTR_DIR)/src/prometeo_metric_coeffs_gpu.o)
	$(AR) rcT $@ $^

$(HTR_DIR)/src/prometeo_metric_coeffs_cpu.o: force
	 $(MAKE) -C $(HTR_DIR)/src prometeo_metric_coeffs_cpu.o

$(HTR_DIR)/src/prometeo_metric_coeffs_gpu.o: force
	 $(MAKE) -C $(HTR_DIR)/src prometeo_metric_coeffs_gpu.o

$(HTR_DIR)/src/version.rg:
	 $(MAKE) -C $(HTR_DIR)/src version.rg

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/Utils/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	make -C $(HTR_DIR)/src config_schema.o

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	make -C $(HTR_DIR)/src json.o

