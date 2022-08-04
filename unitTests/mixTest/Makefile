######################################################
# Include standard variables
######################################################
include $(HTR_DIR)/Makefile.in

######################################################
# Rules
######################################################
.PHONY: default all clean force

default: mixTest_ConstPropMix.exec mixTest_AirMix.exec mixTest_CH41StMix.exec \
         mixTest_CH4_30SpMix.exec mixTest_CH4_43SpIonsMix.exec mixTest_CH4_26SpIonsMix.exec \
         mixTest_FFCM1Mix.exec mixTest_BoivinMix.exec mixTest_H2_UCSDMix.exec

######################################################
# Objects
######################################################
MIXS= $(HTR_DIR)/src/prometeo_mixture.rg

######################################################
# Recipes
######################################################
clean:
	$(RM) *.exec *.o *.a

mixTest_%.exec: mixTest_%.o $(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/json.o libtest_%.a
	$(CXX) -o $@ $^ $(LINK_FLAGS)

mixTest_%.o: mixTest.rg $(HTR_DIR)/src/config_schema.h $(MIXS)
	EOS="$*" $(REGENT) $(REGENT_OPTS) mixTest.rg $(REGENT_FLAGS)

libtest_%.a: $(HTR_DIR)/src/prometeo_mixture_%_cpu.o \
			  $(if $(filter $(strip $(USE_CUDA)), 1),  libGPU_%.a)
	$(AR) rcT $@ $^

libGPU_%.a: $(HTR_DIR)/src/prometeo_mixture_%_gpu.o
	$(NVCC) $(NVCC_FLAGS) -dlink $^ -o $(@:.a=.o)
	$(NVCC) -lib -o $@ $^ $(@:.a=.o)

$(HTR_DIR)/src/prometeo_mixture_%_cpu.o: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_mixture_$*_cpu.o

$(HTR_DIR)/src/prometeo_mixture_%_gpu.o: force
	$(MAKE) -C $(HTR_DIR)/src prometeo_mixture_$*_gpu.o

$(HTR_DIR)/src/config_schema.o $(HTR_DIR)/src/config_schema.h: $(HTR_DIR)/src/Utils/process_schema.rg $(HTR_DIR)/src/config_schema.lua
	make -C $(HTR_DIR)/src config_schema.o

json.o json.h: $(HTR_DIR)/src/json.c $(HTR_DIR)/src/json.h
	make -C $(HTR_DIR)/src json.o

