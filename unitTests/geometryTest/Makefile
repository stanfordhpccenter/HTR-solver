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
MODULES= $(HTR_DIR)/src/prometeo_const.rg \
			$(HTR_DIR)/src/prometeo_macro.rg \
			$(HTR_DIR)/src/prometeo_metric.rg \
			$(HTR_DIR)/src/prometeo_partitioner.rg \
			$(HTR_DIR)/src/prometeo_grid.rg

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o

geometryTest.exec: geometryTest.o $(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/json.o $(HTR_DIR)/src/prometeo_metric_coeffs_cpu.o
	$(CXX) -o $@ $^ $(LINK_FLAGS)

geometryTest.o: geometryTest.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/util-desugared.rg $(MODULES)
	EOS=ConstPropMix $(REGENT) geometryTest.rg $(REGENT_FLAGS)

$(HTR_DIR)/src/prometeo_metric_coeffs_cpu.o: force
	 $(MAKE) -C $(HTR_DIR)/src prometeo_metric_coeffs_cpu.o

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	make -C $(HTR_DIR)/src config_schema.o

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	make -C $(HTR_DIR)/src json.o

$(HTR_DIR)/src/util-desugared.rg: $(HTR_DIR)/src/util.rg
	make -C $(HTR_DIR)/src util-desugared.rg

