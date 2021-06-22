#######################################################
# Include standard variables
#######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean force

default: cflTest.exec

######################################################
# Objects
######################################################
MODULES= $(HTR_DIR)/src/prometeo_mixture.rg \
			$(HTR_DIR)/src/prometeo_cfl.rg

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o

cflTest.exec: cflTest.o $(HTR_DIR)/src/prometeo_ConstPropMix_cpu.a
	$(CXX) -o $@ $^ $(LINK_FLAGS)

cflTest.o: cflTest.rg $(HTR_DIR)/src/config_schema.h $(HTR_DIR)/src/util-desugared.rg $(MODULES)
	EOS="ConstPropMix" $(REGENT) cflTest.rg $(REGENT_FLAGS)

$(HTR_DIR)/src/prometeo_ConstPropMix_cpu.a: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_ConstPropMix_cpu.a

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	make -C $(HTR_DIR)/src config_schema.o

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	make -C $(HTR_DIR)/src json.o

$(HTR_DIR)/src/util-desugared.rg: $(HTR_DIR)/src/util.rg
	make -C $(HTR_DIR)/src util-desugared.rg

