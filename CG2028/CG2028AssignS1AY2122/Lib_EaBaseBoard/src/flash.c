/*****************************************************************************
 *   flash.c:  Driver for the SPI Flash (AT45DB081D-SU)
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
 ******************************************************************************/

/*
 * NOTE: SPI must have been initialized before calling any functions in
 * this file.
 *
 */

/******************************************************************************
 * Includes
 *****************************************************************************/

#include "lpc17xx_gpio.h"
#include "lpc17xx_ssp.h"
#include "flash.h"

/******************************************************************************
 * Defines and typedefs
 *****************************************************************************/

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#define FLASH_CS_OFF() GPIO_SetValue(2, 1<<2);
#define FLASH_CS_ON()  GPIO_ClearValue( 2, 1<<2 )


#define FLASH_CMD_RDID      0x9F        /* read device ID */
#define FLASH_CMD_RDSR      0xD7        /* read status register */
#define FLASH_CMD_FAST_READ 0x0B        /* read data (extra command setup time allows higher SPI clock) */

#define FLASH_CMD_SE        0x7C        /* sector erase */
#define FLASH_CMD_BE        0x50        /* block erase */
#define FLASH_CMD_PE        0x81        /* page erase */
#define FLASH_CMD_PP_BUF    0x82        /* page program through buffer 1 */

#define FLASH_CMD_DP        0xB9        /* deep power down */
#define FLASH_CMD_RES       0xAB        /* release from deep power down */


#define ATMEL_MANU_ID   0x1F
#define ATMEL_DEVICE_ID1 ((1 << 5) | (0x05) ) /* DataFlash, 8 MBit */

/* status register defines */
#define STATUS_RDY      (1 << 7)
#define STATUS_COMP     (1 << 6)
#define STATUS_PROTECT  (1 << 1)
#define STATUS_POW2     (1 << 0)



#define FLAG_IS_POW2 0x01

struct _flash_info
{
    char* name;
    uint32_t jedec_id;
    uint32_t nr_pages;
    uint16_t pagesize;
    uint16_t pageoffset;
    uint8_t flags;
};

/******************************************************************************
 * External global variables
 *****************************************************************************/


/******************************************************************************
 * Local variables
 *****************************************************************************/

static uint16_t pageSize = 0;
static uint16_t pageOffset = 0;
static uint8_t  pageIsPow2 = 0;
static uint8_t  pageSizeChanged = FALSE;
static uint32_t flashTotalSize = 0;

static struct _flash_info flash_devices[] = {
        {"AT45DB081D", 0x1F2500, 4096, 264, 9, 0},
        {"AT45DB081D", 0x1F2500, 4096, 256, 8, FLAG_IS_POW2},
        {"AT45DB081D", 0x1F2600, 4096, 528, 10, 0},
        {"AT45DB081D", 0x1F2600, 4096, 512, 9, FLAG_IS_POW2},
};

/******************************************************************************
 * Local Functions
 *****************************************************************************/

static void SSPSend(uint8_t *buf, uint32_t Length)
{
    SSP_DATA_SETUP_Type xferConfig;

	xferConfig.tx_data = buf;
	xferConfig.rx_data = NULL;
	xferConfig.length  = Length;

    SSP_ReadWrite(LPC_SSP1, &xferConfig, SSP_TRANSFER_POLLING);
}

void SSPReceive( uint8_t *buf, uint32_t Length )
{
    SSP_DATA_SETUP_Type xferConfig;

	xferConfig.tx_data = NULL;
	xferConfig.rx_data = buf;
	xferConfig.length  = Length;

    SSP_ReadWrite(LPC_SSP1, &xferConfig, SSP_TRANSFER_POLLING);
}

static void exitDeepPowerDown(void)
{
    uint8_t cmd = FLASH_CMD_RES;
    FLASH_CS_ON();

    SSPSend( (uint8_t *)&cmd, 1 );

    FLASH_CS_OFF();
}

static void readDeviceId(uint8_t *deviceId)
{
    uint8_t cmd = FLASH_CMD_RDID;

    FLASH_CS_ON();

    SSPSend( (uint8_t *)&cmd, 1 );
    SSPReceive( deviceId, 4 );

    FLASH_CS_OFF();
}

static uint8_t readStatus(void)
{
    uint8_t cmd = FLASH_CMD_RDSR;
    uint8_t status = 0;

    FLASH_CS_ON();

    SSPSend( (uint8_t *)&cmd, 1 );
    SSPReceive( &status, 1 );

    FLASH_CS_OFF();

    return status;
}


static void pollIsBusy(void)
{
  uint8_t status = 0;
  int i = 0;

  do
  {
    for (i = 0; i < 0x2000; i++);

    status = readStatus();
  }
  while ((status & STATUS_RDY) == 0);
}

static void setAddressBytes(uint8_t* addr, uint32_t offset)
{
    if (!pageIsPow2) {
        uint16_t page = offset / pageSize;
        uint16_t off  = offset % pageSize;

        /* buffer address bits */
        addr[2] = (off & 0xff);
        addr[1] = (off / pageSize);

        /* page address bits */
        addr[1] |= ((page & ((1 << (16-pageOffset))-1)) << (pageOffset-8));
        addr[0] = ((page >> (16-pageOffset)) & ((1 << (16-pageOffset))-1) );
    }
    else {

        addr[0] = ((offset >> 16) & 0xFF);
        addr[1] = ((offset >>  8) & 0xFF);
        addr[2] = ((offset >>  0) & 0xFF);

    }
}

/******************************************************************************
 * Public Functions
 *****************************************************************************/

/******************************************************************************
 *
 * Description:
 *    Initialize the SPI Flash Driver
 *
 * Returns:
 *   TRUE if initialization successful, otherwise FALSE
 *
 *****************************************************************************/
uint32_t flash_init (void)
{
    uint8_t deviceId[4];
    uint8_t status = 0;
    uint8_t flag   = 0;
    uint32_t id = 0;
    int i = 0;


    GPIO_SetDir(2, 1<<2, 1);
    FLASH_CS_OFF();

    exitDeepPowerDown();
    readDeviceId(deviceId);

    do {
        status = readStatus();
    } while ((status & STATUS_RDY) == 0);

    if ((status & STATUS_POW2) != 0)
    {
        flag = FLAG_IS_POW2;
    }

    id = (deviceId[0] << 16 | deviceId[1] << 8 | deviceId[2]);

    for (i = 0; i < sizeof(flash_devices) / sizeof(struct _flash_info); i++) {
        if (flash_devices[i].jedec_id == id
                && flash_devices[i].flags == flag)
        {
            pageSize   = flash_devices[i].pagesize;
            pageOffset = flash_devices[i].pageoffset;;
            pageIsPow2 = ((flash_devices[i].flags & FLAG_IS_POW2) != 0) ;
            flashTotalSize = flash_devices[i].pagesize * flash_devices[i].nr_pages;

            return TRUE;
        }
    }


    return FALSE;
}


/******************************************************************************
 *
 * Description:
 *    Write data to flash
 *
 * Params:
 *   [in] buf - data to write to flash
 *   [in] offset - offset into the flash
 *   [in] len - number of bytes to write
 *
 * Returns:
 *   number of written bytes
 *
 *****************************************************************************/
uint32_t flash_write(uint8_t* buf, uint32_t offset, uint32_t len)
{
    uint16_t wLen;
    uint32_t written = 0;
    uint8_t addr[4];

    if (len > flashTotalSize || len+offset > flashTotalSize) {
        return 0;
    }

    if (pageSizeChanged) {
        return 0;
    }

    /* write up to first page boundry */
    wLen = (((( (offset%pageSize) / pageSize) + 1) * pageSize)
            - (offset%pageSize));
    wLen = MIN(wLen, len);

    while (len) {

        /* write address */
        addr[0] = FLASH_CMD_PP_BUF;
        setAddressBytes(&addr[1], offset);

        FLASH_CS_ON();

        SSPSend(addr, 4);

        /* write data */
        SSPSend(&buf[written], wLen);

        FLASH_CS_OFF();

        /* delay to wait for a write cycle */
        //eepromDelay();

        len     -= wLen;
        written += wLen;
        offset  += wLen;
        wLen = MIN(pageSize, len);

        pollIsBusy();
    }

    return written;
}

/******************************************************************************
 *
 * Description:
 *    Read data from flash
 *
 * Params:
 *   [in] buf - data buffer
 *   [in] offset - offset into the flash
 *   [in] len - number of bytes to read
 *
 * Returns:
 *   number of read bytes
 *
 *****************************************************************************/
uint32_t flash_read(uint8_t* buf, uint32_t offset, uint32_t len)
{
    uint8_t addr[5];

    if (len > flashTotalSize || len+offset > flashTotalSize) {
        return 0;
    }

    if (pageSizeChanged) {
        return 0;
    }

    addr[0] = FLASH_CMD_FAST_READ;

    setAddressBytes(&addr[1], offset);
    addr[4] = (0);

    FLASH_CS_ON();

    SSPSend(addr, 5);
    SSPReceive(buf, len);

    FLASH_CS_OFF();

    return len;
}

/******************************************************************************
 *
 * Description:
 *    Get flash page size
 *
 * Returns:
 *   256 or 264
 *
 *****************************************************************************/
uint16_t flash_getPageSize(void)
{
    return pageSize;
}

/******************************************************************************
 *
 * Description:
 *    Change page size to binary page, i.e. 256 bytes.
 *
 *    NOTE: This operation cannot be undone and requires a power-cycle
 *    before taking effect.
 *
 *****************************************************************************/
void flash_setToBinaryPageSize(void)
{
    uint8_t data[4];

    data[0] = 0x3D;
    data[1] = 0x2A;
    data[2] = 0x80;
    data[3] = 0xA6;

    FLASH_CS_ON();

    SSPSend( (uint8_t *)data, 4 );

    FLASH_CS_OFF();

    pageSizeChanged = (!pageIsPow2);
}
