######################################################
# Include standard variables
######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean force

default: operatorsTest_Periodic.exec operatorsTest_Staggered.exec operatorsTest_Collocated.exec \
         metricTest_Periodic.exec metricTest_Collocated.exec metricTest_Staggered.exec

######################################################
# Objects
######################################################
MODULES= $(HTR_DIR)/src/prometeo_const.rg \
			$(HTR_DIR)/src/prometeo_macro.rg \
			$(HTR_DIR)/src/prometeo_partitioner.rg \
			$(HTR_DIR)/src/prometeo_grid.rg \
			$(HTR_DIR)/src/prometeo_metric.rg

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o

operatorsTest_%.exec: operatorsTest_%.o $(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/json.o \
							$(HTR_DIR)/src/prometeo_metric_ConstPropMix_cpu.o $(HTR_DIR)/src/prometeo_metric_coeffs_cpu.o
	$(CXX) -o $@ $^ $(LINK_FLAGS)

operatorsTest_%.o: operatorsTest_%.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/util-desugared.rg \
							$(HTR_DIR)/src/prometeo_metric.h $(MODULES)
	EOS="ConstPropMix" $(REGENT) $^ $(REGENT_FLAGS)

metricTest_%.exec: metricTest_%.o $(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/json.o \
							$(HTR_DIR)/src/prometeo_metric_ConstPropMix_cpu.o $(HTR_DIR)/src/prometeo_metric_coeffs_cpu.o
	$(CXX) -o $@ $^ $(LINK_FLAGS)

metricTest_%.o: metricTest_%.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/util-desugared.rg \
							$(HTR_DIR)/src/prometeo_metric.h $(MODULES)
	EOS="ConstPropMix" $(REGENT) $^ $(REGENT_FLAGS)

$(HTR_DIR)/src/prometeo_metric_ConstPropMix_cpu.o: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_metric_ConstPropMix_cpu.o

$(HTR_DIR)/src/prometeo_metric_coeffs_cpu.o: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_metric_coeffs_cpu.o

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	$(MAKE) -C $(HTR_DIR)/src config_schema.o

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	$(MAKE) -C $(HTR_DIR)/src json.o

$(HTR_DIR)/src/util-desugared.rg: $(HTR_DIR)/src/util.rg
	$(MAKE) -C $(HTR_DIR)/src util-desugared.rg

