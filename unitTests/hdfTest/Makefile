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

hdfTest.o: hdfTest.rg $(HTR_DIR)/src/hdf_helper.rg $(HTR_DIR)/src/util-desugared.rg
	$(REGENT) hdfTest.rg $(REGENT_FLAGS)

$(HTR_DIR)/src/util-desugared.rg: $(HTR_DIR)/src/util.rg
	make -C $(HTR_DIR)/src util-desugared.rg
