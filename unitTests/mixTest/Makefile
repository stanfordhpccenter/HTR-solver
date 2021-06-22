######################################################
# Include standard variables
######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean force

default: mixTest_ConstPropMix.exec mixTest_AirMix.exec mixTest_CH41StMix.exec \
         mixTest_CH4_30SpMix.exec mixTest_CH4_43SpIonsMix.exec mixTest_FFCM1Mix.exec \
         mixTest_BoivinMix.exec

######################################################
# Objects
######################################################
MIXS= $(HTR_DIR)/src/prometeo_mixture.rg

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o

mixTest_%.exec: mixTest_%.o $(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/json.o $(HTR_DIR)/src/prometeo_mixture_%_cpu.o
	$(CXX) -o $@ $^ $(LINK_FLAGS)

mixTest_%.o: mixTest.rg $(HTR_DIR)/src/config_schema.h $(MIXS)
	EOS="$*" $(REGENT) mixTest.rg $(REGENT_FLAGS)

$(HTR_DIR)/src/prometeo_mixture_%_cpu.o: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_mixture_$*_cpu.o

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	make -C $(HTR_DIR)/src config_schema.o

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	make -C $(HTR_DIR)/src json.o

