   1              		.syntax unified
   2              		.cpu cortex-m3
   3              		.fpu softvfp
   4              		.eabi_attribute 20, 1
   5              		.eabi_attribute 21, 1
   6              		.eabi_attribute 23, 3
   7              		.eabi_attribute 24, 1
   8              		.eabi_attribute 25, 1
   9              		.eabi_attribute 26, 1
  10              		.eabi_attribute 30, 6
  11              		.eabi_attribute 34, 1
  12              		.eabi_attribute 18, 4
  13              		.thumb
  14              		.syntax unified
  15              		.file	"cr_startup_lpc17.c"
  16              		.text
  17              	.Ltext0:
  18              		.cfi_sections	.debug_frame
  19              		.global	g_pfnVectors
  20              		.section	.isr_vector,"a",%progbits
  21              		.align	2
  24              	g_pfnVectors:
  25 0000 00000000 		.word	_vStackTop
  26 0004 00000000 		.word	ResetISR
  27 0008 00000000 		.word	NMI_Handler
  28 000c 00000000 		.word	HardFault_Handler
  29 0010 00000000 		.word	MemManage_Handler
  30 0014 00000000 		.word	BusFault_Handler
  31 0018 00000000 		.word	UsageFault_Handler
  32 001c 00000000 		.word	0
  33 0020 00000000 		.word	0
  34 0024 00000000 		.word	0
  35 0028 00000000 		.word	0
  36 002c 00000000 		.word	SVCall_Handler
  37 0030 00000000 		.word	DebugMon_Handler
  38 0034 00000000 		.word	0
  39 0038 00000000 		.word	PendSV_Handler
  40 003c 00000000 		.word	SysTick_Handler
  41 0040 00000000 		.word	WDT_IRQHandler
  42 0044 00000000 		.word	TIMER0_IRQHandler
  43 0048 00000000 		.word	TIMER1_IRQHandler
  44 004c 00000000 		.word	TIMER2_IRQHandler
  45 0050 00000000 		.word	TIMER3_IRQHandler
  46 0054 00000000 		.word	UART0_IRQHandler
  47 0058 00000000 		.word	UART1_IRQHandler
  48 005c 00000000 		.word	UART2_IRQHandler
  49 0060 00000000 		.word	UART3_IRQHandler
  50 0064 00000000 		.word	PWM1_IRQHandler
  51 0068 00000000 		.word	I2C0_IRQHandler
  52 006c 00000000 		.word	I2C1_IRQHandler
  53 0070 00000000 		.word	I2C2_IRQHandler
  54 0074 00000000 		.word	SPI_IRQHandler
  55 0078 00000000 		.word	SSP0_IRQHandler
  56 007c 00000000 		.word	SSP1_IRQHandler
  57 0080 00000000 		.word	PLL0_IRQHandler
  58 0084 00000000 		.word	RTC_IRQHandler
  59 0088 00000000 		.word	EINT0_IRQHandler
  60 008c 00000000 		.word	EINT1_IRQHandler
  61 0090 00000000 		.word	EINT2_IRQHandler
  62 0094 00000000 		.word	EINT3_IRQHandler
  63 0098 00000000 		.word	ADC_IRQHandler
  64 009c 00000000 		.word	BOD_IRQHandler
  65 00a0 00000000 		.word	USB_IRQHandler
  66 00a4 00000000 		.word	CAN_IRQHandler
  67 00a8 00000000 		.word	DMA_IRQHandler
  68 00ac 00000000 		.word	I2S_IRQHandler
  69 00b0 00000000 		.word	ENET_IRQHandler
  70 00b4 00000000 		.word	RIT_IRQHandler
  71 00b8 00000000 		.word	MCPWM_IRQHandler
  72 00bc 00000000 		.word	QEI_IRQHandler
  73 00c0 00000000 		.word	PLL1_IRQHandler
  74 00c4 00000000 		.word	USBActivity_IRQHandler
  75 00c8 00000000 		.word	CANActivity_IRQHandler
  76              		.section	.text.ResetISR,"ax",%progbits
  77              		.align	2
  78              		.global	ResetISR
  79              		.thumb
  80              		.thumb_func
  82              	ResetISR:
  83              	.LFB0:
  84              		.file 1 "../src/cr_startup_lpc17.c"
   1:../src/cr_startup_lpc17.c **** //*****************************************************************************
   2:../src/cr_startup_lpc17.c **** //   +--+       
   3:../src/cr_startup_lpc17.c **** //   | ++----+   
   4:../src/cr_startup_lpc17.c **** //   +-++    |  
   5:../src/cr_startup_lpc17.c **** //     |     |  
   6:../src/cr_startup_lpc17.c **** //   +-+--+  |   
   7:../src/cr_startup_lpc17.c **** //   | +--+--+  
   8:../src/cr_startup_lpc17.c **** //   +----+    Copyright (c) 2009-10 Code Red Technologies Ltd.
   9:../src/cr_startup_lpc17.c **** //
  10:../src/cr_startup_lpc17.c **** // Microcontroller Startup code for use with Red Suite
  11:../src/cr_startup_lpc17.c **** //
  12:../src/cr_startup_lpc17.c **** // Software License Agreement
  13:../src/cr_startup_lpc17.c **** // 
  14:../src/cr_startup_lpc17.c **** // The software is owned by Code Red Technologies and/or its suppliers, and is 
  15:../src/cr_startup_lpc17.c **** // protected under applicable copyright laws.  All rights are reserved.  Any 
  16:../src/cr_startup_lpc17.c **** // use in violation of the foregoing restrictions may subject the user to criminal 
  17:../src/cr_startup_lpc17.c **** // sanctions under applicable laws, as well as to civil liability for the breach 
  18:../src/cr_startup_lpc17.c **** // of the terms and conditions of this license.
  19:../src/cr_startup_lpc17.c **** // 
  20:../src/cr_startup_lpc17.c **** // THIS SOFTWARE IS PROVIDED "AS IS".  NO WARRANTIES, WHETHER EXPRESS, IMPLIED
  21:../src/cr_startup_lpc17.c **** // OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
  22:../src/cr_startup_lpc17.c **** // MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE.
  23:../src/cr_startup_lpc17.c **** // USE OF THIS SOFTWARE FOR COMMERCIAL DEVELOPMENT AND/OR EDUCATION IS SUBJECT
  24:../src/cr_startup_lpc17.c **** // TO A CURRENT END USER LICENSE AGREEMENT (COMMERCIAL OR EDUCATIONAL) WITH
  25:../src/cr_startup_lpc17.c **** // CODE RED TECHNOLOGIES LTD. 
  26:../src/cr_startup_lpc17.c **** //
  27:../src/cr_startup_lpc17.c **** //*****************************************************************************
  28:../src/cr_startup_lpc17.c **** #if defined (__cplusplus)
  29:../src/cr_startup_lpc17.c **** #ifdef __REDLIB__
  30:../src/cr_startup_lpc17.c **** #error Redlib does not support C++
  31:../src/cr_startup_lpc17.c **** #else
  32:../src/cr_startup_lpc17.c **** //*****************************************************************************
  33:../src/cr_startup_lpc17.c **** //
  34:../src/cr_startup_lpc17.c **** // The entry point for the C++ library startup
  35:../src/cr_startup_lpc17.c **** //
  36:../src/cr_startup_lpc17.c **** //*****************************************************************************
  37:../src/cr_startup_lpc17.c **** extern "C" {
  38:../src/cr_startup_lpc17.c **** 	extern void __libc_init_array(void);
  39:../src/cr_startup_lpc17.c **** }
  40:../src/cr_startup_lpc17.c **** #endif
  41:../src/cr_startup_lpc17.c **** #endif
  42:../src/cr_startup_lpc17.c **** 
  43:../src/cr_startup_lpc17.c **** #define WEAK __attribute__ ((weak))
  44:../src/cr_startup_lpc17.c **** #define ALIAS(f) __attribute__ ((weak, alias (#f)))
  45:../src/cr_startup_lpc17.c **** 
  46:../src/cr_startup_lpc17.c **** // Code Red - if CMSIS is being used, then SystemInit() routine
  47:../src/cr_startup_lpc17.c **** // will be called by startup code rather than in application's main()
  48:../src/cr_startup_lpc17.c **** #if defined (__USE_CMSIS)
  49:../src/cr_startup_lpc17.c **** #include "system_LPC17xx.h"
  50:../src/cr_startup_lpc17.c **** #endif
  51:../src/cr_startup_lpc17.c **** 
  52:../src/cr_startup_lpc17.c **** //*****************************************************************************
  53:../src/cr_startup_lpc17.c **** #if defined (__cplusplus)
  54:../src/cr_startup_lpc17.c **** extern "C" {
  55:../src/cr_startup_lpc17.c **** #endif
  56:../src/cr_startup_lpc17.c **** 
  57:../src/cr_startup_lpc17.c **** //*****************************************************************************
  58:../src/cr_startup_lpc17.c **** //
  59:../src/cr_startup_lpc17.c **** // Forward declaration of the default handlers. These are aliased.
  60:../src/cr_startup_lpc17.c **** // When the application defines a handler (with the same name), this will 
  61:../src/cr_startup_lpc17.c **** // automatically take precedence over these weak definitions
  62:../src/cr_startup_lpc17.c **** //
  63:../src/cr_startup_lpc17.c **** //*****************************************************************************
  64:../src/cr_startup_lpc17.c ****      void ResetISR(void);
  65:../src/cr_startup_lpc17.c **** WEAK void NMI_Handler(void);
  66:../src/cr_startup_lpc17.c **** WEAK void HardFault_Handler(void);
  67:../src/cr_startup_lpc17.c **** WEAK void MemManage_Handler(void);
  68:../src/cr_startup_lpc17.c **** WEAK void BusFault_Handler(void);
  69:../src/cr_startup_lpc17.c **** WEAK void UsageFault_Handler(void);
  70:../src/cr_startup_lpc17.c **** WEAK void SVCall_Handler(void);
  71:../src/cr_startup_lpc17.c **** WEAK void DebugMon_Handler(void);
  72:../src/cr_startup_lpc17.c **** WEAK void PendSV_Handler(void);
  73:../src/cr_startup_lpc17.c **** WEAK void SysTick_Handler(void);
  74:../src/cr_startup_lpc17.c **** WEAK void IntDefaultHandler(void);
  75:../src/cr_startup_lpc17.c **** 
  76:../src/cr_startup_lpc17.c **** //*****************************************************************************
  77:../src/cr_startup_lpc17.c **** //
  78:../src/cr_startup_lpc17.c **** // Forward declaration of the specific IRQ handlers. These are aliased
  79:../src/cr_startup_lpc17.c **** // to the IntDefaultHandler, which is a 'forever' loop. When the application
  80:../src/cr_startup_lpc17.c **** // defines a handler (with the same name), this will automatically take 
  81:../src/cr_startup_lpc17.c **** // precedence over these weak definitions
  82:../src/cr_startup_lpc17.c **** //
  83:../src/cr_startup_lpc17.c **** //*****************************************************************************
  84:../src/cr_startup_lpc17.c **** void WDT_IRQHandler(void) ALIAS(IntDefaultHandler);
  85:../src/cr_startup_lpc17.c **** void TIMER0_IRQHandler(void) ALIAS(IntDefaultHandler);
  86:../src/cr_startup_lpc17.c **** void TIMER1_IRQHandler(void) ALIAS(IntDefaultHandler);
  87:../src/cr_startup_lpc17.c **** void TIMER2_IRQHandler(void) ALIAS(IntDefaultHandler);
  88:../src/cr_startup_lpc17.c **** void TIMER3_IRQHandler(void) ALIAS(IntDefaultHandler);
  89:../src/cr_startup_lpc17.c **** void UART0_IRQHandler(void) ALIAS(IntDefaultHandler);
  90:../src/cr_startup_lpc17.c **** void UART1_IRQHandler(void) ALIAS(IntDefaultHandler);
  91:../src/cr_startup_lpc17.c **** void UART2_IRQHandler(void) ALIAS(IntDefaultHandler);
  92:../src/cr_startup_lpc17.c **** void UART3_IRQHandler(void) ALIAS(IntDefaultHandler);
  93:../src/cr_startup_lpc17.c **** void PWM1_IRQHandler(void) ALIAS(IntDefaultHandler);
  94:../src/cr_startup_lpc17.c **** void I2C0_IRQHandler(void) ALIAS(IntDefaultHandler);
  95:../src/cr_startup_lpc17.c **** void I2C1_IRQHandler(void) ALIAS(IntDefaultHandler);
  96:../src/cr_startup_lpc17.c **** void I2C2_IRQHandler(void) ALIAS(IntDefaultHandler);
  97:../src/cr_startup_lpc17.c **** void SPI_IRQHandler(void) ALIAS(IntDefaultHandler);
  98:../src/cr_startup_lpc17.c **** void SSP0_IRQHandler(void) ALIAS(IntDefaultHandler);
  99:../src/cr_startup_lpc17.c **** void SSP1_IRQHandler(void) ALIAS(IntDefaultHandler);
 100:../src/cr_startup_lpc17.c **** void PLL0_IRQHandler(void) ALIAS(IntDefaultHandler);
 101:../src/cr_startup_lpc17.c **** void RTC_IRQHandler(void) ALIAS(IntDefaultHandler);
 102:../src/cr_startup_lpc17.c **** void EINT0_IRQHandler(void) ALIAS(IntDefaultHandler);
 103:../src/cr_startup_lpc17.c **** void EINT1_IRQHandler(void) ALIAS(IntDefaultHandler);
 104:../src/cr_startup_lpc17.c **** void EINT2_IRQHandler(void) ALIAS(IntDefaultHandler);
 105:../src/cr_startup_lpc17.c **** void EINT3_IRQHandler(void) ALIAS(IntDefaultHandler);
 106:../src/cr_startup_lpc17.c **** void ADC_IRQHandler(void) ALIAS(IntDefaultHandler);
 107:../src/cr_startup_lpc17.c **** void BOD_IRQHandler(void) ALIAS(IntDefaultHandler);
 108:../src/cr_startup_lpc17.c **** void USB_IRQHandler(void) ALIAS(IntDefaultHandler);
 109:../src/cr_startup_lpc17.c **** void CAN_IRQHandler(void) ALIAS(IntDefaultHandler);
 110:../src/cr_startup_lpc17.c **** void DMA_IRQHandler(void) ALIAS(IntDefaultHandler);
 111:../src/cr_startup_lpc17.c **** void I2S_IRQHandler(void) ALIAS(IntDefaultHandler);
 112:../src/cr_startup_lpc17.c **** void ENET_IRQHandler(void) ALIAS(IntDefaultHandler);
 113:../src/cr_startup_lpc17.c **** void RIT_IRQHandler(void) ALIAS(IntDefaultHandler);
 114:../src/cr_startup_lpc17.c **** void MCPWM_IRQHandler(void) ALIAS(IntDefaultHandler);
 115:../src/cr_startup_lpc17.c **** void QEI_IRQHandler(void) ALIAS(IntDefaultHandler);
 116:../src/cr_startup_lpc17.c **** void PLL1_IRQHandler(void) ALIAS(IntDefaultHandler);
 117:../src/cr_startup_lpc17.c **** void USBActivity_IRQHandler(void) ALIAS(IntDefaultHandler);
 118:../src/cr_startup_lpc17.c **** void CANActivity_IRQHandler(void) ALIAS(IntDefaultHandler);
 119:../src/cr_startup_lpc17.c **** 
 120:../src/cr_startup_lpc17.c **** //*****************************************************************************
 121:../src/cr_startup_lpc17.c **** //
 122:../src/cr_startup_lpc17.c **** // The entry point for the application.
 123:../src/cr_startup_lpc17.c **** // __main() is the entry point for Redlib based applications
 124:../src/cr_startup_lpc17.c **** // main() is the entry point for Newlib based applications
 125:../src/cr_startup_lpc17.c **** //
 126:../src/cr_startup_lpc17.c **** //*****************************************************************************
 127:../src/cr_startup_lpc17.c **** #if defined (__REDLIB__)
 128:../src/cr_startup_lpc17.c **** extern void __main(void);
 129:../src/cr_startup_lpc17.c **** #endif
 130:../src/cr_startup_lpc17.c **** extern int main(void);
 131:../src/cr_startup_lpc17.c **** //*****************************************************************************
 132:../src/cr_startup_lpc17.c **** //
 133:../src/cr_startup_lpc17.c **** // External declaration for the pointer to the stack top from the Linker Script
 134:../src/cr_startup_lpc17.c **** //
 135:../src/cr_startup_lpc17.c **** //*****************************************************************************
 136:../src/cr_startup_lpc17.c **** extern void _vStackTop(void);
 137:../src/cr_startup_lpc17.c **** 
 138:../src/cr_startup_lpc17.c **** //*****************************************************************************
 139:../src/cr_startup_lpc17.c **** #if defined (__cplusplus)
 140:../src/cr_startup_lpc17.c **** } // extern "C"
 141:../src/cr_startup_lpc17.c **** #endif
 142:../src/cr_startup_lpc17.c **** //*****************************************************************************
 143:../src/cr_startup_lpc17.c **** //
 144:../src/cr_startup_lpc17.c **** // The vector table.
 145:../src/cr_startup_lpc17.c **** // This relies on the linker script to place at correct location in memory.
 146:../src/cr_startup_lpc17.c **** //
 147:../src/cr_startup_lpc17.c **** //*****************************************************************************
 148:../src/cr_startup_lpc17.c **** extern void (* const g_pfnVectors[])(void);
 149:../src/cr_startup_lpc17.c **** __attribute__ ((section(".isr_vector")))
 150:../src/cr_startup_lpc17.c **** void (* const g_pfnVectors[])(void) = {
 151:../src/cr_startup_lpc17.c **** 	// Core Level - CM3
 152:../src/cr_startup_lpc17.c **** 	&_vStackTop, // The initial stack pointer
 153:../src/cr_startup_lpc17.c **** 	ResetISR,								// The reset handler
 154:../src/cr_startup_lpc17.c **** 	NMI_Handler,							// The NMI handler
 155:../src/cr_startup_lpc17.c **** 	HardFault_Handler,						// The hard fault handler
 156:../src/cr_startup_lpc17.c **** 	MemManage_Handler,						// The MPU fault handler
 157:../src/cr_startup_lpc17.c **** 	BusFault_Handler,						// The bus fault handler
 158:../src/cr_startup_lpc17.c **** 	UsageFault_Handler,						// The usage fault handler
 159:../src/cr_startup_lpc17.c **** 	0,										// Reserved
 160:../src/cr_startup_lpc17.c **** 	0,										// Reserved
 161:../src/cr_startup_lpc17.c **** 	0,										// Reserved
 162:../src/cr_startup_lpc17.c **** 	0,										// Reserved
 163:../src/cr_startup_lpc17.c **** 	SVCall_Handler,							// SVCall handler
 164:../src/cr_startup_lpc17.c **** 	DebugMon_Handler,						// Debug monitor handler
 165:../src/cr_startup_lpc17.c **** 	0,										// Reserved
 166:../src/cr_startup_lpc17.c **** 	PendSV_Handler,							// The PendSV handler
 167:../src/cr_startup_lpc17.c **** 	SysTick_Handler,						// The SysTick handler
 168:../src/cr_startup_lpc17.c **** 
 169:../src/cr_startup_lpc17.c **** 	// Chip Level - LPC17
 170:../src/cr_startup_lpc17.c **** 	WDT_IRQHandler,							// 16, 0x40 - WDT
 171:../src/cr_startup_lpc17.c **** 	TIMER0_IRQHandler,						// 17, 0x44 - TIMER0
 172:../src/cr_startup_lpc17.c **** 	TIMER1_IRQHandler,						// 18, 0x48 - TIMER1
 173:../src/cr_startup_lpc17.c **** 	TIMER2_IRQHandler,						// 19, 0x4c - TIMER2
 174:../src/cr_startup_lpc17.c **** 	TIMER3_IRQHandler,						// 20, 0x50 - TIMER3
 175:../src/cr_startup_lpc17.c **** 	UART0_IRQHandler,						// 21, 0x54 - UART0
 176:../src/cr_startup_lpc17.c **** 	UART1_IRQHandler,						// 22, 0x58 - UART1
 177:../src/cr_startup_lpc17.c **** 	UART2_IRQHandler,						// 23, 0x5c - UART2
 178:../src/cr_startup_lpc17.c **** 	UART3_IRQHandler,						// 24, 0x60 - UART3
 179:../src/cr_startup_lpc17.c **** 	PWM1_IRQHandler,						// 25, 0x64 - PWM1
 180:../src/cr_startup_lpc17.c **** 	I2C0_IRQHandler,						// 26, 0x68 - I2C0
 181:../src/cr_startup_lpc17.c **** 	I2C1_IRQHandler,						// 27, 0x6c - I2C1
 182:../src/cr_startup_lpc17.c **** 	I2C2_IRQHandler,						// 28, 0x70 - I2C2
 183:../src/cr_startup_lpc17.c **** 	SPI_IRQHandler,							// 29, 0x74 - SPI
 184:../src/cr_startup_lpc17.c **** 	SSP0_IRQHandler,						// 30, 0x78 - SSP0
 185:../src/cr_startup_lpc17.c **** 	SSP1_IRQHandler,						// 31, 0x7c - SSP1
 186:../src/cr_startup_lpc17.c **** 	PLL0_IRQHandler,						// 32, 0x80 - PLL0 (Main PLL)
 187:../src/cr_startup_lpc17.c **** 	RTC_IRQHandler,							// 33, 0x84 - RTC
 188:../src/cr_startup_lpc17.c **** 	EINT0_IRQHandler,						// 34, 0x88 - EINT0
 189:../src/cr_startup_lpc17.c **** 	EINT1_IRQHandler,						// 35, 0x8c - EINT1
 190:../src/cr_startup_lpc17.c **** 	EINT2_IRQHandler,						// 36, 0x90 - EINT2
 191:../src/cr_startup_lpc17.c **** 	EINT3_IRQHandler,						// 37, 0x94 - EINT3
 192:../src/cr_startup_lpc17.c **** 	ADC_IRQHandler,							// 38, 0x98 - ADC
 193:../src/cr_startup_lpc17.c **** 	BOD_IRQHandler,							// 39, 0x9c - BOD
 194:../src/cr_startup_lpc17.c **** 	USB_IRQHandler,							// 40, 0xA0 - USB
 195:../src/cr_startup_lpc17.c **** 	CAN_IRQHandler,							// 41, 0xa4 - CAN
 196:../src/cr_startup_lpc17.c **** 	DMA_IRQHandler,							// 42, 0xa8 - GP DMA
 197:../src/cr_startup_lpc17.c **** 	I2S_IRQHandler,							// 43, 0xac - I2S
 198:../src/cr_startup_lpc17.c **** 	ENET_IRQHandler,						// 44, 0xb0 - Ethernet
 199:../src/cr_startup_lpc17.c **** 	RIT_IRQHandler,							// 45, 0xb4 - RITINT
 200:../src/cr_startup_lpc17.c **** 	MCPWM_IRQHandler,						// 46, 0xb8 - Motor Control PWM
 201:../src/cr_startup_lpc17.c **** 	QEI_IRQHandler,							// 47, 0xbc - Quadrature Encoder
 202:../src/cr_startup_lpc17.c **** 	PLL1_IRQHandler,						// 48, 0xc0 - PLL1 (USB PLL)
 203:../src/cr_startup_lpc17.c **** 	USBActivity_IRQHandler,					// 49, 0xc4 - USB Activity interrupt to wakeup
 204:../src/cr_startup_lpc17.c **** 	CANActivity_IRQHandler, 				// 50, 0xc8 - CAN Activity interrupt to wakeup
 205:../src/cr_startup_lpc17.c **** };
 206:../src/cr_startup_lpc17.c **** 
 207:../src/cr_startup_lpc17.c **** //*****************************************************************************
 208:../src/cr_startup_lpc17.c **** //
 209:../src/cr_startup_lpc17.c **** // The following are constructs created by the linker, indicating where the
 210:../src/cr_startup_lpc17.c **** // the "data" and "bss" segments reside in memory.  The initializers for the
 211:../src/cr_startup_lpc17.c **** // for the "data" segment resides immediately following the "text" segment.
 212:../src/cr_startup_lpc17.c **** //
 213:../src/cr_startup_lpc17.c **** //*****************************************************************************
 214:../src/cr_startup_lpc17.c **** extern unsigned long _etext;
 215:../src/cr_startup_lpc17.c **** extern unsigned long _data;
 216:../src/cr_startup_lpc17.c **** extern unsigned long _edata;
 217:../src/cr_startup_lpc17.c **** extern unsigned long _bss;
 218:../src/cr_startup_lpc17.c **** extern unsigned long _ebss;
 219:../src/cr_startup_lpc17.c **** 
 220:../src/cr_startup_lpc17.c **** //*****************************************************************************
 221:../src/cr_startup_lpc17.c **** // Reset entry point for your code.
 222:../src/cr_startup_lpc17.c **** // Sets up a simple runtime environment and initializes the C/C++
 223:../src/cr_startup_lpc17.c **** // library.
 224:../src/cr_startup_lpc17.c **** //
 225:../src/cr_startup_lpc17.c **** //*****************************************************************************
 226:../src/cr_startup_lpc17.c **** void
 227:../src/cr_startup_lpc17.c **** ResetISR(void) {
  85              		.loc 1 227 0
  86              		.cfi_startproc
  87              		@ args = 0, pretend = 0, frame = 8
  88              		@ frame_needed = 1, uses_anonymous_args = 0
  89 0000 80B5     		push	{r7, lr}
  90              		.cfi_def_cfa_offset 8
  91              		.cfi_offset 7, -8
  92              		.cfi_offset 14, -4
  93 0002 82B0     		sub	sp, sp, #8
  94              		.cfi_def_cfa_offset 16
  95 0004 00AF     		add	r7, sp, #0
  96              		.cfi_def_cfa_register 7
 228:../src/cr_startup_lpc17.c ****     unsigned long *pulSrc, *pulDest;
 229:../src/cr_startup_lpc17.c **** 
 230:../src/cr_startup_lpc17.c ****     //
 231:../src/cr_startup_lpc17.c ****     // Copy the data segment initializers from flash to SRAM.
 232:../src/cr_startup_lpc17.c ****     //
 233:../src/cr_startup_lpc17.c ****     pulSrc = &_etext;
  97              		.loc 1 233 0
  98 0006 0F4B     		ldr	r3, .L5
  99 0008 7B60     		str	r3, [r7, #4]
 234:../src/cr_startup_lpc17.c ****     for(pulDest = &_data; pulDest < &_edata; )
 100              		.loc 1 234 0
 101 000a 0F4B     		ldr	r3, .L5+4
 102 000c 3B60     		str	r3, [r7]
 103 000e 07E0     		b	.L2
 104              	.L3:
 235:../src/cr_startup_lpc17.c ****     {
 236:../src/cr_startup_lpc17.c ****         *pulDest++ = *pulSrc++;
 105              		.loc 1 236 0
 106 0010 3B68     		ldr	r3, [r7]
 107 0012 1A1D     		adds	r2, r3, #4
 108 0014 3A60     		str	r2, [r7]
 109 0016 7A68     		ldr	r2, [r7, #4]
 110 0018 111D     		adds	r1, r2, #4
 111 001a 7960     		str	r1, [r7, #4]
 112 001c 1268     		ldr	r2, [r2]
 113 001e 1A60     		str	r2, [r3]
 114              	.L2:
 234:../src/cr_startup_lpc17.c ****     for(pulDest = &_data; pulDest < &_edata; )
 115              		.loc 1 234 0 discriminator 1
 116 0020 3B68     		ldr	r3, [r7]
 117 0022 0A4A     		ldr	r2, .L5+8
 118 0024 9342     		cmp	r3, r2
 119 0026 F3D3     		bcc	.L3
 237:../src/cr_startup_lpc17.c ****     }
 238:../src/cr_startup_lpc17.c **** 
 239:../src/cr_startup_lpc17.c ****     //
 240:../src/cr_startup_lpc17.c ****     // Zero fill the bss segment.  This is done with inline assembly since this
 241:../src/cr_startup_lpc17.c ****     // will clear the value of pulDest if it is not kept in a register.
 242:../src/cr_startup_lpc17.c ****     //
 243:../src/cr_startup_lpc17.c ****     __asm("    ldr     r0, =_bss\n"
 120              		.loc 1 243 0
 121              		.syntax unified
 122              	@ 243 "../src/cr_startup_lpc17.c" 1
 123 0028 0948     		    ldr     r0, =_bss
 124 002a 0A49     	    ldr     r1, =_ebss
 125 002c 4FF00002 	    mov     r2, #0
 126              	    .thumb_func
 127              	zero_loop:
 128 0030 8842     	        cmp     r0, r1
 129 0032 B8BF     	        it      lt
 130 0034 40F8042B 	        strlt   r2, [r0], #4
 131 0038 FADB     	        blt     zero_loop
 132              	@ 0 "" 2
 244:../src/cr_startup_lpc17.c ****           "    ldr     r1, =_ebss\n"
 245:../src/cr_startup_lpc17.c ****           "    mov     r2, #0\n"
 246:../src/cr_startup_lpc17.c ****           "    .thumb_func\n"
 247:../src/cr_startup_lpc17.c ****           "zero_loop:\n"
 248:../src/cr_startup_lpc17.c ****           "        cmp     r0, r1\n"
 249:../src/cr_startup_lpc17.c ****           "        it      lt\n"
 250:../src/cr_startup_lpc17.c ****           "        strlt   r2, [r0], #4\n"
 251:../src/cr_startup_lpc17.c ****           "        blt     zero_loop");
 252:../src/cr_startup_lpc17.c **** 
 253:../src/cr_startup_lpc17.c **** #ifdef __USE_CMSIS
 254:../src/cr_startup_lpc17.c **** 	SystemInit();
 133              		.loc 1 254 0
 134              		.thumb
 135              		.syntax unified
 136 003a FFF7FEFF 		bl	SystemInit
 255:../src/cr_startup_lpc17.c **** #endif
 256:../src/cr_startup_lpc17.c **** 
 257:../src/cr_startup_lpc17.c **** #if defined (__cplusplus)
 258:../src/cr_startup_lpc17.c **** 	//
 259:../src/cr_startup_lpc17.c **** 	// Call C++ library initialisation
 260:../src/cr_startup_lpc17.c **** 	//
 261:../src/cr_startup_lpc17.c **** 	__libc_init_array();
 262:../src/cr_startup_lpc17.c **** #endif
 263:../src/cr_startup_lpc17.c **** 
 264:../src/cr_startup_lpc17.c **** #if defined (__REDLIB__)
 265:../src/cr_startup_lpc17.c **** 	// Call the Redlib library, which in turn calls main()
 266:../src/cr_startup_lpc17.c **** 	__main() ;
 137              		.loc 1 266 0
 138 003e FFF7FEFF 		bl	__main
 139              	.L4:
 267:../src/cr_startup_lpc17.c **** #else
 268:../src/cr_startup_lpc17.c **** 	main();
 269:../src/cr_startup_lpc17.c **** #endif
 270:../src/cr_startup_lpc17.c **** 
 271:../src/cr_startup_lpc17.c **** 	//
 272:../src/cr_startup_lpc17.c **** 	// main() shouldn't return, but if it does, we'll just enter an infinite loop 
 273:../src/cr_startup_lpc17.c **** 	//
 274:../src/cr_startup_lpc17.c **** 	while (1) {
 275:../src/cr_startup_lpc17.c **** 		;
 276:../src/cr_startup_lpc17.c **** 	}
 140              		.loc 1 276 0 discriminator 1
 141 0042 FEE7     		b	.L4
 142              	.L6:
 143              		.align	2
 144              	.L5:
 145 0044 00000000 		.word	_etext
 146 0048 00000000 		.word	_data
 147 004c 00000000 		.word	_edata
 148              		.cfi_endproc
 149              	.LFE0:
 151              		.section	.text.NMI_Handler,"ax",%progbits
 152              		.align	2
 153              		.weak	NMI_Handler
 154              		.thumb
 155              		.thumb_func
 157              	NMI_Handler:
 158              	.LFB1:
 277:../src/cr_startup_lpc17.c **** }
 278:../src/cr_startup_lpc17.c **** 
 279:../src/cr_startup_lpc17.c **** //*****************************************************************************
 280:../src/cr_startup_lpc17.c **** //
 281:../src/cr_startup_lpc17.c **** // This is the code that gets called when the processor receives a NMI.  This
 282:../src/cr_startup_lpc17.c **** // simply enters an infinite loop, preserving the system state for examination
 283:../src/cr_startup_lpc17.c **** // by a debugger.
 284:../src/cr_startup_lpc17.c **** //
 285:../src/cr_startup_lpc17.c **** //*****************************************************************************
 286:../src/cr_startup_lpc17.c **** void NMI_Handler(void)
 287:../src/cr_startup_lpc17.c **** {
 159              		.loc 1 287 0
 160              		.cfi_startproc
 161              		@ args = 0, pretend = 0, frame = 0
 162              		@ frame_needed = 1, uses_anonymous_args = 0
 163              		@ link register save eliminated.
 164 0000 80B4     		push	{r7}
 165              		.cfi_def_cfa_offset 4
 166              		.cfi_offset 7, -4
 167 0002 00AF     		add	r7, sp, #0
 168              		.cfi_def_cfa_register 7
 169              	.L8:
 288:../src/cr_startup_lpc17.c ****     while(1)
 289:../src/cr_startup_lpc17.c ****     {
 290:../src/cr_startup_lpc17.c ****     }
 170              		.loc 1 290 0 discriminator 1
 171 0004 FEE7     		b	.L8
 172              		.cfi_endproc
 173              	.LFE1:
 175 0006 00BF     		.section	.text.HardFault_Handler,"ax",%progbits
 176              		.align	2
 177              		.weak	HardFault_Handler
 178              		.thumb
 179              		.thumb_func
 181              	HardFault_Handler:
 182              	.LFB2:
 291:../src/cr_startup_lpc17.c **** }
 292:../src/cr_startup_lpc17.c **** 
 293:../src/cr_startup_lpc17.c **** void HardFault_Handler(void)
 294:../src/cr_startup_lpc17.c **** {
 183              		.loc 1 294 0
 184              		.cfi_startproc
 185              		@ args = 0, pretend = 0, frame = 0
 186              		@ frame_needed = 1, uses_anonymous_args = 0
 187              		@ link register save eliminated.
 188 0000 80B4     		push	{r7}
 189              		.cfi_def_cfa_offset 4
 190              		.cfi_offset 7, -4
 191 0002 00AF     		add	r7, sp, #0
 192              		.cfi_def_cfa_register 7
 193              	.L10:
 295:../src/cr_startup_lpc17.c ****     while(1)
 296:../src/cr_startup_lpc17.c ****     {
 297:../src/cr_startup_lpc17.c ****     }
 194              		.loc 1 297 0 discriminator 1
 195 0004 FEE7     		b	.L10
 196              		.cfi_endproc
 197              	.LFE2:
 199 0006 00BF     		.section	.text.MemManage_Handler,"ax",%progbits
 200              		.align	2
 201              		.weak	MemManage_Handler
 202              		.thumb
 203              		.thumb_func
 205              	MemManage_Handler:
 206              	.LFB3:
 298:../src/cr_startup_lpc17.c **** }
 299:../src/cr_startup_lpc17.c **** 
 300:../src/cr_startup_lpc17.c **** void MemManage_Handler(void)
 301:../src/cr_startup_lpc17.c **** {
 207              		.loc 1 301 0
 208              		.cfi_startproc
 209              		@ args = 0, pretend = 0, frame = 0
 210              		@ frame_needed = 1, uses_anonymous_args = 0
 211              		@ link register save eliminated.
 212 0000 80B4     		push	{r7}
 213              		.cfi_def_cfa_offset 4
 214              		.cfi_offset 7, -4
 215 0002 00AF     		add	r7, sp, #0
 216              		.cfi_def_cfa_register 7
 217              	.L12:
 302:../src/cr_startup_lpc17.c ****     while(1)
 303:../src/cr_startup_lpc17.c ****     {
 304:../src/cr_startup_lpc17.c ****     }
 218              		.loc 1 304 0 discriminator 1
 219 0004 FEE7     		b	.L12
 220              		.cfi_endproc
 221              	.LFE3:
 223 0006 00BF     		.section	.text.BusFault_Handler,"ax",%progbits
 224              		.align	2
 225              		.weak	BusFault_Handler
 226              		.thumb
 227              		.thumb_func
 229              	BusFault_Handler:
 230              	.LFB4:
 305:../src/cr_startup_lpc17.c **** }
 306:../src/cr_startup_lpc17.c **** 
 307:../src/cr_startup_lpc17.c **** void BusFault_Handler(void)
 308:../src/cr_startup_lpc17.c **** {
 231              		.loc 1 308 0
 232              		.cfi_startproc
 233              		@ args = 0, pretend = 0, frame = 0
 234              		@ frame_needed = 1, uses_anonymous_args = 0
 235              		@ link register save eliminated.
 236 0000 80B4     		push	{r7}
 237              		.cfi_def_cfa_offset 4
 238              		.cfi_offset 7, -4
 239 0002 00AF     		add	r7, sp, #0
 240              		.cfi_def_cfa_register 7
 241              	.L14:
 309:../src/cr_startup_lpc17.c ****     while(1)
 310:../src/cr_startup_lpc17.c ****     {
 311:../src/cr_startup_lpc17.c ****     }
 242              		.loc 1 311 0 discriminator 1
 243 0004 FEE7     		b	.L14
 244              		.cfi_endproc
 245              	.LFE4:
 247 0006 00BF     		.section	.text.UsageFault_Handler,"ax",%progbits
 248              		.align	2
 249              		.weak	UsageFault_Handler
 250              		.thumb
 251              		.thumb_func
 253              	UsageFault_Handler:
 254              	.LFB5:
 312:../src/cr_startup_lpc17.c **** }
 313:../src/cr_startup_lpc17.c **** 
 314:../src/cr_startup_lpc17.c **** void UsageFault_Handler(void)
 315:../src/cr_startup_lpc17.c **** {
 255              		.loc 1 315 0
 256              		.cfi_startproc
 257              		@ args = 0, pretend = 0, frame = 0
 258              		@ frame_needed = 1, uses_anonymous_args = 0
 259              		@ link register save eliminated.
 260 0000 80B4     		push	{r7}
 261              		.cfi_def_cfa_offset 4
 262              		.cfi_offset 7, -4
 263 0002 00AF     		add	r7, sp, #0
 264              		.cfi_def_cfa_register 7
 265              	.L16:
 316:../src/cr_startup_lpc17.c ****     while(1)
 317:../src/cr_startup_lpc17.c ****     {
 318:../src/cr_startup_lpc17.c ****     }
 266              		.loc 1 318 0 discriminator 1
 267 0004 FEE7     		b	.L16
 268              		.cfi_endproc
 269              	.LFE5:
 271 0006 00BF     		.section	.text.SVCall_Handler,"ax",%progbits
 272              		.align	2
 273              		.weak	SVCall_Handler
 274              		.thumb
 275              		.thumb_func
 277              	SVCall_Handler:
 278              	.LFB6:
 319:../src/cr_startup_lpc17.c **** }
 320:../src/cr_startup_lpc17.c **** 
 321:../src/cr_startup_lpc17.c **** void SVCall_Handler(void)
 322:../src/cr_startup_lpc17.c **** {
 279              		.loc 1 322 0
 280              		.cfi_startproc
 281              		@ args = 0, pretend = 0, frame = 0
 282              		@ frame_needed = 1, uses_anonymous_args = 0
 283              		@ link register save eliminated.
 284 0000 80B4     		push	{r7}
 285              		.cfi_def_cfa_offset 4
 286              		.cfi_offset 7, -4
 287 0002 00AF     		add	r7, sp, #0
 288              		.cfi_def_cfa_register 7
 289              	.L18:
 323:../src/cr_startup_lpc17.c ****     while(1)
 324:../src/cr_startup_lpc17.c ****     {
 325:../src/cr_startup_lpc17.c ****     }
 290              		.loc 1 325 0 discriminator 1
 291 0004 FEE7     		b	.L18
 292              		.cfi_endproc
 293              	.LFE6:
 295 0006 00BF     		.section	.text.DebugMon_Handler,"ax",%progbits
 296              		.align	2
 297              		.weak	DebugMon_Handler
 298              		.thumb
 299              		.thumb_func
 301              	DebugMon_Handler:
 302              	.LFB7:
 326:../src/cr_startup_lpc17.c **** }
 327:../src/cr_startup_lpc17.c **** 
 328:../src/cr_startup_lpc17.c **** void DebugMon_Handler(void)
 329:../src/cr_startup_lpc17.c **** {
 303              		.loc 1 329 0
 304              		.cfi_startproc
 305              		@ args = 0, pretend = 0, frame = 0
 306              		@ frame_needed = 1, uses_anonymous_args = 0
 307              		@ link register save eliminated.
 308 0000 80B4     		push	{r7}
 309              		.cfi_def_cfa_offset 4
 310              		.cfi_offset 7, -4
 311 0002 00AF     		add	r7, sp, #0
 312              		.cfi_def_cfa_register 7
 313              	.L20:
 330:../src/cr_startup_lpc17.c ****     while(1)
 331:../src/cr_startup_lpc17.c ****     {
 332:../src/cr_startup_lpc17.c ****     }
 314              		.loc 1 332 0 discriminator 1
 315 0004 FEE7     		b	.L20
 316              		.cfi_endproc
 317              	.LFE7:
 319 0006 00BF     		.section	.text.PendSV_Handler,"ax",%progbits
 320              		.align	2
 321              		.weak	PendSV_Handler
 322              		.thumb
 323              		.thumb_func
 325              	PendSV_Handler:
 326              	.LFB8:
 333:../src/cr_startup_lpc17.c **** }
 334:../src/cr_startup_lpc17.c **** 
 335:../src/cr_startup_lpc17.c **** void PendSV_Handler(void)
 336:../src/cr_startup_lpc17.c **** {
 327              		.loc 1 336 0
 328              		.cfi_startproc
 329              		@ args = 0, pretend = 0, frame = 0
 330              		@ frame_needed = 1, uses_anonymous_args = 0
 331              		@ link register save eliminated.
 332 0000 80B4     		push	{r7}
 333              		.cfi_def_cfa_offset 4
 334              		.cfi_offset 7, -4
 335 0002 00AF     		add	r7, sp, #0
 336              		.cfi_def_cfa_register 7
 337              	.L22:
 337:../src/cr_startup_lpc17.c ****     while(1)
 338:../src/cr_startup_lpc17.c ****     {
 339:../src/cr_startup_lpc17.c ****     }
 338              		.loc 1 339 0 discriminator 1
 339 0004 FEE7     		b	.L22
 340              		.cfi_endproc
 341              	.LFE8:
 343 0006 00BF     		.section	.text.SysTick_Handler,"ax",%progbits
 344              		.align	2
 345              		.weak	SysTick_Handler
 346              		.thumb
 347              		.thumb_func
 349              	SysTick_Handler:
 350              	.LFB9:
 340:../src/cr_startup_lpc17.c **** }
 341:../src/cr_startup_lpc17.c **** 
 342:../src/cr_startup_lpc17.c **** void SysTick_Handler(void) 
 343:../src/cr_startup_lpc17.c **** {
 351              		.loc 1 343 0
 352              		.cfi_startproc
 353              		@ args = 0, pretend = 0, frame = 0
 354              		@ frame_needed = 1, uses_anonymous_args = 0
 355              		@ link register save eliminated.
 356 0000 80B4     		push	{r7}
 357              		.cfi_def_cfa_offset 4
 358              		.cfi_offset 7, -4
 359 0002 00AF     		add	r7, sp, #0
 360              		.cfi_def_cfa_register 7
 361              	.L24:
 344:../src/cr_startup_lpc17.c ****     while(1)
 345:../src/cr_startup_lpc17.c ****     {
 346:../src/cr_startup_lpc17.c ****     }
 362              		.loc 1 346 0 discriminator 1
 363 0004 FEE7     		b	.L24
 364              		.cfi_endproc
 365              	.LFE9:
 367 0006 00BF     		.section	.text.IntDefaultHandler,"ax",%progbits
 368              		.align	2
 369              		.weak	IntDefaultHandler
 370              		.thumb
 371              		.thumb_func
 373              	IntDefaultHandler:
 374              	.LFB10:
 347:../src/cr_startup_lpc17.c **** }
 348:../src/cr_startup_lpc17.c **** 
 349:../src/cr_startup_lpc17.c **** 
 350:../src/cr_startup_lpc17.c **** //*****************************************************************************
 351:../src/cr_startup_lpc17.c **** //
 352:../src/cr_startup_lpc17.c **** // Processor ends up here if an unexpected interrupt occurs or a handler
 353:../src/cr_startup_lpc17.c **** // is not present in the application code.
 354:../src/cr_startup_lpc17.c **** //
 355:../src/cr_startup_lpc17.c **** //*****************************************************************************
 356:../src/cr_startup_lpc17.c **** void IntDefaultHandler(void)
 357:../src/cr_startup_lpc17.c **** {
 375              		.loc 1 357 0
 376              		.cfi_startproc
 377              		@ args = 0, pretend = 0, frame = 0
 378              		@ frame_needed = 1, uses_anonymous_args = 0
 379              		@ link register save eliminated.
 380 0000 80B4     		push	{r7}
 381              		.cfi_def_cfa_offset 4
 382              		.cfi_offset 7, -4
 383 0002 00AF     		add	r7, sp, #0
 384              		.cfi_def_cfa_register 7
 385              	.L26:
 358:../src/cr_startup_lpc17.c ****     //
 359:../src/cr_startup_lpc17.c ****     // Go into an infinite loop.
 360:../src/cr_startup_lpc17.c ****     //
 361:../src/cr_startup_lpc17.c ****     while(1)
 362:../src/cr_startup_lpc17.c ****     {
 363:../src/cr_startup_lpc17.c ****     }
 386              		.loc 1 363 0 discriminator 1
 387 0004 FEE7     		b	.L26
 388              		.cfi_endproc
 389              	.LFE10:
 391              		.weak	CANActivity_IRQHandler
 392              		.thumb_set CANActivity_IRQHandler,IntDefaultHandler
 393              		.weak	USBActivity_IRQHandler
 394              		.thumb_set USBActivity_IRQHandler,IntDefaultHandler
 395              		.weak	PLL1_IRQHandler
 396              		.thumb_set PLL1_IRQHandler,IntDefaultHandler
 397              		.weak	QEI_IRQHandler
 398              		.thumb_set QEI_IRQHandler,IntDefaultHandler
 399              		.weak	MCPWM_IRQHandler
 400              		.thumb_set MCPWM_IRQHandler,IntDefaultHandler
 401              		.weak	RIT_IRQHandler
 402              		.thumb_set RIT_IRQHandler,IntDefaultHandler
 403              		.weak	ENET_IRQHandler
 404              		.thumb_set ENET_IRQHandler,IntDefaultHandler
 405              		.weak	I2S_IRQHandler
 406              		.thumb_set I2S_IRQHandler,IntDefaultHandler
 407              		.weak	DMA_IRQHandler
 408              		.thumb_set DMA_IRQHandler,IntDefaultHandler
 409              		.weak	CAN_IRQHandler
 410              		.thumb_set CAN_IRQHandler,IntDefaultHandler
 411              		.weak	USB_IRQHandler
 412              		.thumb_set USB_IRQHandler,IntDefaultHandler
 413              		.weak	BOD_IRQHandler
 414              		.thumb_set BOD_IRQHandler,IntDefaultHandler
 415              		.weak	ADC_IRQHandler
 416              		.thumb_set ADC_IRQHandler,IntDefaultHandler
 417              		.weak	EINT3_IRQHandler
 418              		.thumb_set EINT3_IRQHandler,IntDefaultHandler
 419              		.weak	EINT2_IRQHandler
 420              		.thumb_set EINT2_IRQHandler,IntDefaultHandler
 421              		.weak	EINT1_IRQHandler
 422              		.thumb_set EINT1_IRQHandler,IntDefaultHandler
 423              		.weak	EINT0_IRQHandler
 424              		.thumb_set EINT0_IRQHandler,IntDefaultHandler
 425              		.weak	RTC_IRQHandler
 426              		.thumb_set RTC_IRQHandler,IntDefaultHandler
 427              		.weak	PLL0_IRQHandler
 428              		.thumb_set PLL0_IRQHandler,IntDefaultHandler
 429              		.weak	SSP1_IRQHandler
 430              		.thumb_set SSP1_IRQHandler,IntDefaultHandler
 431              		.weak	SSP0_IRQHandler
 432              		.thumb_set SSP0_IRQHandler,IntDefaultHandler
 433              		.weak	SPI_IRQHandler
 434              		.thumb_set SPI_IRQHandler,IntDefaultHandler
 435              		.weak	I2C2_IRQHandler
 436              		.thumb_set I2C2_IRQHandler,IntDefaultHandler
 437              		.weak	I2C1_IRQHandler
 438              		.thumb_set I2C1_IRQHandler,IntDefaultHandler
 439              		.weak	I2C0_IRQHandler
 440              		.thumb_set I2C0_IRQHandler,IntDefaultHandler
 441              		.weak	PWM1_IRQHandler
 442              		.thumb_set PWM1_IRQHandler,IntDefaultHandler
 443              		.weak	UART3_IRQHandler
 444              		.thumb_set UART3_IRQHandler,IntDefaultHandler
 445              		.weak	UART2_IRQHandler
 446              		.thumb_set UART2_IRQHandler,IntDefaultHandler
 447              		.weak	UART1_IRQHandler
 448              		.thumb_set UART1_IRQHandler,IntDefaultHandler
 449              		.weak	UART0_IRQHandler
 450              		.thumb_set UART0_IRQHandler,IntDefaultHandler
 451              		.weak	TIMER3_IRQHandler
 452              		.thumb_set TIMER3_IRQHandler,IntDefaultHandler
 453              		.weak	TIMER2_IRQHandler
 454              		.thumb_set TIMER2_IRQHandler,IntDefaultHandler
 455              		.weak	TIMER1_IRQHandler
 456              		.thumb_set TIMER1_IRQHandler,IntDefaultHandler
 457              		.weak	TIMER0_IRQHandler
 458              		.thumb_set TIMER0_IRQHandler,IntDefaultHandler
 459              		.weak	WDT_IRQHandler
 460              		.thumb_set WDT_IRQHandler,IntDefaultHandler
 461 0006 00BF     		.text
 462              	.Letext0:
DEFINED SYMBOLS
                            *ABS*:0000000000000000 cr_startup_lpc17.c
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:24     .isr_vector:0000000000000000 g_pfnVectors
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:21     .isr_vector:0000000000000000 $d
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:82     .text.ResetISR:0000000000000000 ResetISR
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:157    .text.NMI_Handler:0000000000000000 NMI_Handler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:181    .text.HardFault_Handler:0000000000000000 HardFault_Handler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:205    .text.MemManage_Handler:0000000000000000 MemManage_Handler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:229    .text.BusFault_Handler:0000000000000000 BusFault_Handler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:253    .text.UsageFault_Handler:0000000000000000 UsageFault_Handler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:277    .text.SVCall_Handler:0000000000000000 SVCall_Handler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:301    .text.DebugMon_Handler:0000000000000000 DebugMon_Handler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:325    .text.PendSV_Handler:0000000000000000 PendSV_Handler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:349    .text.SysTick_Handler:0000000000000000 SysTick_Handler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 WDT_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 TIMER0_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 TIMER1_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 TIMER2_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 TIMER3_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 UART0_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 UART1_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 UART2_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 UART3_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 PWM1_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 I2C0_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 I2C1_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 I2C2_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 SPI_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 SSP0_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 SSP1_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 PLL0_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 RTC_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 EINT0_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 EINT1_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 EINT2_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 EINT3_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 ADC_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 BOD_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 USB_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 CAN_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 DMA_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 I2S_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 ENET_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 RIT_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 MCPWM_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 QEI_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 PLL1_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 USBActivity_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 CANActivity_IRQHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:77     .text.ResetISR:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:127    .text.ResetISR:0000000000000030 zero_loop
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:145    .text.ResetISR:0000000000000044 $d
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:152    .text.NMI_Handler:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:176    .text.HardFault_Handler:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:200    .text.MemManage_Handler:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:224    .text.BusFault_Handler:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:248    .text.UsageFault_Handler:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:272    .text.SVCall_Handler:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:296    .text.DebugMon_Handler:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:320    .text.PendSV_Handler:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:344    .text.SysTick_Handler:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:368    .text.IntDefaultHandler:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:373    .text.IntDefaultHandler:0000000000000000 IntDefaultHandler
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccvGhxUz.s:4225   .text.ResetISR:0000000000000050 $d
                     .debug_frame:0000000000000010 $d
                           .group:0000000000000000 wm4.0.0c5e979f1ec464b8f03bc190bd321363
                           .group:0000000000000000 wm4.redlib_version.h.14.62abddb5b4efb2dd619a7dca5647eb78
                           .group:0000000000000000 wm4.libconfigarm.h.18.48d18a57a6aa6fedadbcea02294a713f
                           .group:0000000000000000 wm4.stdint.h.30.30f575a65a9bde434c705acf40986e1b

UNDEFINED SYMBOLS
_vStackTop
_bss
_ebss
SystemInit
__main
_etext
_data
_edata
