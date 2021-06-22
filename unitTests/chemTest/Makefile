#######################################################
# Include standard variables
#######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean force

default: chemTest.exec

######################################################
# Objects
######################################################
MODULES= $(HTR_DIR)/src/prometeo_mixture.rg \
			$(HTR_DIR)/src/prometeo_chem.rg

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o

chemTest.exec: chemTest.o $(HTR_DIR)/src/prometeo_AirMix_cpu.a
	$(CXX) -o $@ $^ $(LINK_FLAGS)

chemTest.o: chemTest.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/util-desugared.rg $(MODULES)
	EOS="AirMix" $(REGENT) chemTest.rg $(REGENT_FLAGS)

$(HTR_DIR)/src/prometeo_AirMix_cpu.a: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_AirMix_cpu.a

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	$(MAKE) -C $(HTR_DIR)/src config_schema.o

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	$(MAKE) -C $(HTR_DIR)/src json.o

$(HTR_DIR)/src/util-desugared.rg: $(HTR_DIR)/src/util.rg
	$(MAKE) -C $(HTR_DIR)/src util-desugared.rg

