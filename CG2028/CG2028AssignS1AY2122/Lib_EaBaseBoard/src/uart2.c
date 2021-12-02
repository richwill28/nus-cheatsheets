/*****************************************************************************
 *   uart2.c:  Driver for the I2C/SPi to UART (SC16IS752) device
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
 ******************************************************************************/

/*
 * NOTE: I2C must have been initialized before calling any functions in this
 * file.
 */

/******************************************************************************
 * Includes
 *****************************************************************************/

#include "lpc17xx_i2c.h"
#include "lpc17xx_uart.h"
#include "lpc17xx_gpio.h"
#include "uart2.h"

/******************************************************************************
 * Defines and typedefs
 *****************************************************************************/

#define I2CDEV LPC_I2C2

#define UART2_ADDR (0x48)

#define R_RHR 0x00
#define R_THR 0x00
#define R_IER 0x01
#define R_FCR 0x02
#define R_IIR 0x02
#define R_LCR 0x03
#define R_MCR 0x04
#define R_LSR 0x05
#define R_MSR 0x06

#define R_IOCTRL 0x0E
#define R_EFCR   0x0F

#define R_DLL 0x00
#define R_DLH 0x01

#define CH_A 0x00
#define CH_B 0x01
#define SUB_ADDR(ch, reg) ((ch&0x03) << 1 | ((reg&0x0F) << 3))

#define LSR_THRE	0x20
#define LSR_RDR		0x01

/******************************************************************************
 * External global variables
 *****************************************************************************/

/******************************************************************************
 * Local variables
 *****************************************************************************/

static uint8_t channel = 0;

/******************************************************************************
 * Local Functions
 *****************************************************************************/

static int I2CRead(uint8_t addr, uint8_t* buf, uint32_t len)
{
	I2C_M_SETUP_Type rxsetup;

	rxsetup.sl_addr7bit = addr;
	rxsetup.tx_data = NULL;	// Get address to read at writing address
	rxsetup.tx_length = 0;
	rxsetup.rx_data = buf;
	rxsetup.rx_length = len;
	rxsetup.retransmissions_max = 3;

	if (I2C_MasterTransferData(I2CDEV, &rxsetup, I2C_TRANSFER_POLLING) == SUCCESS){
		return (0);
	} else {
		return (-1);
	}
}

static int I2CWrite(uint8_t addr, uint8_t* buf, uint32_t len)
{
	I2C_M_SETUP_Type txsetup;

	txsetup.sl_addr7bit = addr;
	txsetup.tx_data = buf;
	txsetup.tx_length = len;
	txsetup.rx_data = NULL;
	txsetup.rx_length = 0;
	txsetup.retransmissions_max = 3;

	if (I2C_MasterTransferData(I2CDEV, &txsetup, I2C_TRANSFER_POLLING) == SUCCESS){
		return (0);
	} else {
		return (-1);
	}
}

static void writeReg(uint8_t reg, uint8_t data)
{
    uint8_t buf[2];

    buf[0] = SUB_ADDR(channel, reg);
    buf[1] = data;
    I2CWrite(UART2_ADDR, buf, 2);
}

static uint8_t readReg(uint8_t reg)
{
    uint8_t buf[1];

    buf[0] = SUB_ADDR(channel, reg);
    I2CWrite(UART2_ADDR, buf, 1);
    I2CRead(UART2_ADDR, buf, 1);

    return buf[0];
}


/******************************************************************************
 * Public Functions
 *****************************************************************************/

/******************************************************************************
 *
 * Description:
 *    Initialize the ISL29003 Device
 *
 * Params:
 *   [in] baudRate - the baud rate to use
 *   [in] chan - the channel to use. Channel A connected to RS232 port.
 *               Channel B connected to J53.
 *
 *****************************************************************************/
void uart2_init (uint32_t baudRate, uart2_channel_t chan)
{

    GPIO_SetDir(0, 1<<9, 1);
    GPIO_SetDir(2, 1<<8, 1);

    GPIO_SetDir(0, 1<<9, 1); // SI-A1
    GPIO_SetDir(2, 1<<8, 1); // CS#-A0

    channel = chan;
    uart2_setBaudRate(baudRate);
}

/******************************************************************************
 *
 * Description:
 *    Change the Baud rate
 *
 * Params:
 *   [in] baudRate - the new baud rate
 *
 *****************************************************************************/
void uart2_setBaudRate(uint32_t baudRate)
{
    uint32_t div = 0;

    if (baudRate < 100 || baudRate > 230400)
        return;

    /* set divisor latch enable */
    writeReg(R_LCR, (1 << 7));

    /*
     * divisor = (3.6864 MHz / prescaler) / (baudRate * 16)
     *
     * Prescaler is by default 1, but can be changed in MCR[7] to 4.
     * (not handled here though)
     */
    div = 3686400 / (baudRate * 16);

    /* set divisor */
    writeReg(R_DLL, (uint8_t)(div & 0xff));
    writeReg(R_DLH, (uint8_t)((div >> 8) & 0xff));

    /* line control  */
    writeReg(R_LCR, 0x03); // 8 bit data, 1 stop bit, no parity
}

/******************************************************************************
 *
 * Description:
 *    Send data to UART
 *
 * Params:
 *   [in] buffer - buffer with data
 *   [in] length - number of bytes of data
 *
 *****************************************************************************/
void uart2_send(uint8_t *buffer, uint32_t length)
{
    if (!buffer) {
        /* error */
        return;
    }

    while ( length != 0 )
    {
        /* THRE status, contain valid data */
        while ( !(readReg(R_LSR) & LSR_THRE) );

        writeReg(R_THR, *buffer);

        buffer++;
        length--;
    }
    return;
}

/******************************************************************************
 *
 * Description:
 *    Send a null-terminated string of data to UART
 *
 * Params:
 *   [in] string - null-terminated string
 *
 *****************************************************************************/
void uart2_sendString(uint8_t *string)
{
    if (!string) {
        /* error */
        return;
    }

    while ( *string != '\0' )
    {
        /* THRE status, contain valid data */
        while ( !(readReg(R_LSR) & LSR_THRE) );
        writeReg(R_THR, *string);

        string++;
    }

    return;
}

/******************************************************************************
 *
 * Description:
 *    Receive data from UART
 *
 * Params:
 *   [in] buffer - data will be written to this buffer
 *   [in] length - length of buffer in bytes
 *   [in] blocking - TRUE if blocking mode should be used; otherwise FALSE
 *
 *****************************************************************************/
uint32_t uart2_receive(uint8_t *buffer, uint32_t length, uint32_t blocking)
{
    uint32_t recvd = 0;
    uint32_t toRecv = length;

    if (blocking) {

        while (toRecv) {
            /* wait for data */
            while (!(readReg(R_LSR) & LSR_RDR));

            *buffer++ = readReg(R_RHR);

            recvd++;
            toRecv--;
        }

    }
    else {

        while (toRecv) {
            /* break if no data */
            if (!(readReg(R_LSR) & LSR_RDR)) {
                break;
            }

            *buffer++ = readReg(R_RHR);

            recvd++;
            toRecv--;
        }
    }

    return recvd;
}

uint8_t uart2_getModemStatus(void)
{
    return readReg(R_MSR);
}

void uart2_setModemStatus(uint8_t msr)
{
    writeReg(R_MCR, msr);
}
