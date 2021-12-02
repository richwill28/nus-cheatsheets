/*****************************************************************************
 *   uart2.h:  Header file for I2C/SPI to UART Device
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
******************************************************************************/
#ifndef __UART2_H
#define __UART2_H

typedef enum {
    CHANNEL_A,
    CHANNEL_B
} uart2_channel_t;

#define MCR_RTS 0x02

#define MSR_CTS 0x10


void uart2_init (uint32_t baudRate, uart2_channel_t chan);
void uart2_setBaudRate(uint32_t baudRate);
void uart2_send(uint8_t *buffer, uint32_t length);
void uart2_sendString(uint8_t *string);
uint32_t uart2_receive(uint8_t *buffer, uint32_t length, uint32_t blocking);
uint8_t uart2_getModemStatus(void);
void uart2_setModemStatus(uint8_t msr);



#endif /* end __UART2_H */
/****************************************************************************
**                            End Of File
*****************************************************************************/
