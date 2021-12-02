/*****************************************************************************
 *   pca9532.h:  Header file for PCA9532 16-bit I2C LED dimmer
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
******************************************************************************/
#ifndef __PCA9532C_H
#define __PCA9532C_H


#define PCA9532_I2C_ADDR    (0x60/* << 1*/)

#define PCA9532_INPUT0 0x00
#define PCA9532_INPUT1 0x01
#define PCA9532_PSC0   0x02
#define PCA9532_PWM0   0x03
#define PCA9532_PSC1   0x04
#define PCA9532_PWM1   0x05
#define PCA9532_LS0    0x06
#define PCA9532_LS1    0x07
#define PCA9532_LS2    0x08
#define PCA9532_LS3    0x09

#define PCA9532_AUTO_INC 0x10

/*
 * Below are the LED constants to use when enabling/disabling a LED.
 * The LED names are the names printed on the base board and not
 * the names from the PCA9532 device. base_LED4 -> LED0 on PCA9532,
 * base_LED5 -> LED1, and so on.
 */

#define LED4  0x0001
#define LED5  0x0002
#define LED6  0x0004
#define LED7  0x0008
#define LED8  0x0010
#define LED9  0x0020
#define LED10 0x0040
#define LED11 0x0080
#define LED12 0x0100
#define LED13 0x0200
#define LED14 0x0400
#define LED15 0x0800
#define LED16 0x1000
#define LED17 0x2000
#define LED18 0x4000
#define LED19 0x8000


void pca9532_init (void);
uint16_t pca9532_getLedState (uint32_t shadow);
void pca9532_setLeds (uint16_t ledOnMask, uint16_t ledOffMask);
void pca9532_setBlink0Period(uint8_t period);
void pca9532_setBlink0Duty(uint8_t duty);
void pca9532_setBlink0Leds(uint16_t ledMask);
void pca9532_setBlink1Period(uint8_t period);
void pca9532_setBlink1Duty(uint8_t duty);
void pca9532_setBlink1Leds(uint16_t ledMask);

#endif /* end __PCA9532C_H */
/****************************************************************************
**                            End Of File
*****************************************************************************/
