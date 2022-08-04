######################################################
# Include standard variables
######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean

default: hdfTest.exec

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o *.hdf

hdfTest.exec: hdfTest.o
	$(CXX) -o $@ $^ $(LINK_FLAGS)

hdfTest.o: hdfTest.rg $(HTR_DIR)/src/Utils/hdf_helper.rg $(HTR_DIR)/src/Utils/util.rg
	$(REGENT) $(REGENT_OPTS) hdfTest.rg $(REGENT_FLAGS)

