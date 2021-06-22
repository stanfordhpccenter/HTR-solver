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
			$(HTR_DIR)/src/prometeo_variables.rg \
			$(HTR_DIR)/src/prometeo_variables.h

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o

variablesTest.exec: variablesTest.o $(HTR_DIR)/src/prometeo_AirMix_cpu.a
	$(CXX) -o $@ $^ $(LINK_FLAGS)

variablesTest.o: variablesTest.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/util-desugared.rg $(MODULES)
	EOS=AirMix $(REGENT) variablesTest.rg $(REGENT_FLAGS)

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	make -C $(HTR_DIR)/src config_schema.o

$(HTR_DIR)/src/prometeo_AirMix_cpu.a: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_AirMix_cpu.a

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	make -C $(HTR_DIR)/src json.o

$(HTR_DIR)/src/util-desugared.rg: $(HTR_DIR)/src/util.rg
	make -C $(HTR_DIR)/src util-desugared.rg

