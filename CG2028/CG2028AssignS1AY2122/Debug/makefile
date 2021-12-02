################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include src/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: CG2028AssignS1AY2122.axf

# Tool invocations
CG2028AssignS1AY2122.axf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: MCU Linker'
	arm-none-eabi-gcc -nostdlib -L"/Users/vera/Documents/WORKSPACE/Lib_CMSISv1p30_LPC17xx/Debug" -Xlinker --gc-sections -Xlinker -Map=CG2028AssignS1AY2122.map -mcpu=cortex-m3 -mthumb -T "CG2028AssignS1AY2122_Debug.ld" -o"CG2028AssignS1AY2122.axf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '
	$(MAKE) --no-print-directory post-build

# Other Targets
clean:
	-$(RM) $(EXECUTABLES)$(OBJS)$(C_DEPS) CG2028AssignS1AY2122.axf
	-@echo ' '

post-build:
	-@echo 'Performing post-build steps'
	-arm-none-eabi-size CG2028AssignS1AY2122.axf; # arm-none-eabi-objcopy -O ihex CG2028AssignS1AY2122.axf CG2028AssignS1AY2122.hex ;
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY: post-build

-include ../makefile.targets
