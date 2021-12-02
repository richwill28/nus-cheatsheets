################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/acc.c \
../src/eeprom.c \
../src/flash.c \
../src/font5x7.c \
../src/joystick.c \
../src/led7seg.c \
../src/light.c \
../src/oled.c \
../src/pca9532.c \
../src/rgb.c \
../src/rotary.c \
../src/temp.c \
../src/uart2.c 

OBJS += \
./src/acc.o \
./src/eeprom.o \
./src/flash.o \
./src/font5x7.o \
./src/joystick.o \
./src/led7seg.o \
./src/light.o \
./src/oled.o \
./src/pca9532.o \
./src/rgb.o \
./src/rotary.o \
./src/temp.o \
./src/uart2.o 

C_DEPS += \
./src/acc.d \
./src/eeprom.d \
./src/flash.d \
./src/font5x7.d \
./src/joystick.d \
./src/led7seg.d \
./src/light.d \
./src/oled.d \
./src/pca9532.d \
./src/rgb.d \
./src/rotary.d \
./src/temp.d \
./src/uart2.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: MCU C Compiler'
	arm-none-eabi-gcc -DDEBUG -D__CODE_RED -D__REDLIB__ -I"/Users/vera/Documents/WORKSPACE/Lib_CMSISv1p30_LPC17xx/inc" -I"/Users/vera/Documents/WORKSPACE/Lib_EaBaseBoard/inc" -I"/Users/vera/Documents/WORKSPACE/Lib_MCU/inc" -O0 -g3 -Wall -c -fmessage-length=0 -fno-builtin -ffunction-sections -mcpu=cortex-m3 -mthumb -D__REDLIB__ -specs=redlib.specs -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.o)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


