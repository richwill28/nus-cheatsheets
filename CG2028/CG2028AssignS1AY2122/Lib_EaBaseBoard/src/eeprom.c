/*****************************************************************************
 *   eeprom.c:  Driver for the 24LC08 EEPROM
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
#include "string.h"
#include "stdio.h"
#include "eeprom.h"

/******************************************************************************
 * Defines and typedefs
 *****************************************************************************/

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#define I2CDEV LPC_I2C2

#define EEPROM_I2C_ADDR1    (0x50)
#define EEPROM_I2C_ADDR2    (0x51)
#define EEPROM_I2C_ADDR3    (0x52)
#define EEPROM_I2C_ADDR4    (0x53)

#define EEPROM_TOTAL_SIZE 1024
#define EEPROM_BLOCK_SIZE  256
#define EEPROM_PAGE_SIZE    16


/******************************************************************************
 * External global variables
 *****************************************************************************/

/******************************************************************************
 * Local variables
 *****************************************************************************/


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


static void eepromDelay(void)
{
    volatile int i = 0;

    for (i = 0; i <0x20000; i++);
}

/******************************************************************************
 * Public Functions
 *****************************************************************************/

/******************************************************************************
 *
 * Description:
 *    Initialize the EEPROM Driver
 *
 *****************************************************************************/
void eeprom_init (void)
{

}

/******************************************************************************
 *
 * Description:
 *    Read from the EEPROM
 *
 * Params:
 *   [in] buf - read buffer
 *   [in] offset - offset to start to read from
 *   [in] len - number of bytes to read
 *
 * Returns:
 *   number of read bytes or -1 in case of an error
 *
 *****************************************************************************/
int16_t eeprom_read(uint8_t* buf, uint16_t offset, uint16_t len)
{
    uint8_t addr = 0;
    int i = 0;

    uint16_t off = offset;

    if (len > EEPROM_TOTAL_SIZE || offset+len > EEPROM_TOTAL_SIZE) {
        return -1;
    }

    addr = EEPROM_I2C_ADDR1 + (offset/EEPROM_BLOCK_SIZE);
    off = offset % EEPROM_BLOCK_SIZE;

    I2CWrite((addr), (uint8_t*)&off, 1);
    for ( i = 0; i < 0x2000; i++);
    I2CRead((addr), buf, len);

    return len;

}

/******************************************************************************
 *
 * Description:
 *    Write to the EEPROM
 *
 * Params:
 *   [in] buf - data to write
 *   [in] offset - offset to start to write to
 *   [in] len - number of bytes to write
 *
 * Returns:
 *   number of written bytes or -1 in case of an error
 *
 *****************************************************************************/
int16_t eeprom_write(uint8_t* buf, uint16_t offset, uint16_t len)
{
    uint8_t addr = 0;
    int16_t written = 0;
    uint16_t wLen = 0;
    uint16_t off = offset;
    uint8_t tmp[17];

    if (len > EEPROM_TOTAL_SIZE || offset+len > EEPROM_TOTAL_SIZE) {
        return -1;
    }

    addr = EEPROM_I2C_ADDR1 + (offset/EEPROM_BLOCK_SIZE);
    off = offset % EEPROM_BLOCK_SIZE;
    wLen = ((((off >> 4) + 1) << 4) - off);
    wLen = MIN(wLen, len);

    while (len) {
        tmp[0] = off;
        memcpy(&tmp[1], (void*)&buf[written], wLen);
        I2CWrite((addr), tmp, wLen+1);

        /* delay to wait for a write cycle */
        eepromDelay();

        len     -= wLen;
        written += wLen;
        off  += wLen;

        wLen = MIN(EEPROM_PAGE_SIZE, len);

        addr += off / EEPROM_BLOCK_SIZE;
        off  = off % EEPROM_BLOCK_SIZE;
    }

    return written;
}
