#######################################################
# Include standard variables
#######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean

default: mathUtilsTest.exec CPPmathUtilsTest.exec

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o

mathUtilsTest.exec: mathUtilsTest.o
	$(CXX) -o $@ $^ $(LINK_FLAGS)

mathUtilsTest.o: mathUtilsTest.rg $(HTR_DIR)/src/Utils/math_utils.rg $(HTR_DIR)/src/Utils/math_utils.h
	$(REGENT) $(REGENT_OPTS) mathUtilsTest.rg $(REGENT_FLAGS)

CPPmathUtilsTest.exec: CPPmathUtilsTest.cc $(HTR_DIR)/src/Utils/math_utils.hpp $(HTR_DIR)/src/Utils/math_utils.h
	$(CXX) $(CXX_FLAGS) -o $@ $<
