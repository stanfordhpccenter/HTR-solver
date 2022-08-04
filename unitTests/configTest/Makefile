######################################################
# Include standard variables
######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean force

default: configTest.exec

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o

configTest.exec: configTest.o $(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/json.o
	$(CXX) -o $@ $^ $(LINK_FLAGS)

configTest.o: configTest.rg $(HTR_DIR)/src/config_schema.h
	$(REGENT) $(REGENT_OPTS) configTest.rg $(REGENT_FLAGS)

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/Utils/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	make -C $(HTR_DIR)/src config_schema.o

json.o json.h : $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	make -C $(HTR_DIR)/src json.o
