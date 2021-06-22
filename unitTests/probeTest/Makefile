######################################################
# Include standard variables
######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean force

default: probeTest.exec

######################################################
# Objects
######################################################
MODULES= $(HTR_DIR)/src/hdf_helper.rg \
			$(HTR_DIR)/src/prometeo_IO.rg \
			$(HTR_DIR)/src/prometeo_const.rg \
			$(HTR_DIR)/src/prometeo_macro.rg \
			$(HTR_DIR)/src/prometeo_mixture.rg \
			$(HTR_DIR)/src/prometeo_probe.rg

######################################################
# Recipes
######################################################
clean:
	$(RM) -r *.exec *.o *csv

probeTest.exec: probeTest.o $(HTR_DIR)/src/prometeo_AirMix_cpu.a
	$(CXX) -o $@ $^ $(LINK_FLAGS)

probeTest.o: probeTest.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/util-desugared.rg $(MODULES)
	EOS="AirMix" $(REGENT) probeTest.rg $(REGENT_FLAGS)

$(HTR_DIR)/src/prometeo_AirMix_cpu.a: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_AirMix_cpu.a

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	$(MAKE) -C $(HTR_DIR)/src config_schema.o

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	$(MAKE) -C $(HTR_DIR)/src json.o

$(HTR_DIR)/src/util-desugared.rg: $(HTR_DIR)/src/util.rg
	$(MAKE) -C $(HTR_DIR)/src util-desugared.rg

