/*****************************************************************************
 *   led7seg.c:  Driver for the 7 segment display
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
#include "led7seg.h"

/******************************************************************************
 * Defines and typedefs
 *****************************************************************************/

#define LED7_CS_OFF() GPIO_SetValue( 2, (1<<2) )
#define LED7_CS_ON()  GPIO_ClearValue( 2, (1<<2) )


/******************************************************************************
 * External global variables
 *****************************************************************************/


/******************************************************************************
 * Local variables
 *****************************************************************************/


/* character mapping */
static uint8_t chars[] = {
        /* '-', '.' */
        0xFB, 0xDF, 0xFF,
        /* digits 0 - 9 */
        0x24, 0xAF, 0xE0, 0xA2, 0x2B, 0x32, 0x30, 0xA7, 0x20, 0x22,
        /* ':' to '@' are invalid */
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        /* A to J */
        0x21, 0x38, 0x74, 0xA8, 0x70, 0x71, 0x10, 0x29, 0x8F, 0xAC,
        /* K to T */
        0xFF, 0x7C,  0xFF, 0xB9, 0x04, 0x61, 0x03, 0xF9, 0x12, 0x78,
        /* U to Z */
        0x2C, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        /* '[' to '`' */
        0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF,
        /* a to j */
        0x21, 0x38, 0xF8, 0xA8, 0x70, 0x71, 0x02, 0x39, 0x8F, 0xAC,
        /* k to t */
        0xFF, 0x7C,  0xFF, 0xB9, 0xB8, 0x61, 0x03, 0xF9, 0x12, 0x78,
        /* u to z */
        0xBC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        /* { to '|' */
        0xFF, 0x7D,

};


/******************************************************************************
 * Local Functions
 *****************************************************************************/


/******************************************************************************
 * Public Functions
 *****************************************************************************/

/******************************************************************************
 *
 * Description:
 *    Initialize the 7 segment Display
 *
 *****************************************************************************/
void led7seg_init (void)
{
    //GPIO_SetDir( 2, 2, 1 );
	GPIO_SetDir( 2, (1<<2), 1 );
    LED7_CS_OFF();
}

/******************************************************************************
 *
 * Description:
 *    Draw a character on the 7 segment display
 *
 * Params:
 *   [in] ch - character interpreted as an ascii character. Not all ascii
 *             characters can be realized on the display. If a character
 *             can't be realized all segments are off.
 *   [in] rawMode - set to TRUE to use raw mode. In this case the ch data
 *             won't be interpreted as an ascii character.
 *
 *****************************************************************************/
void led7seg_setChar(uint8_t ch, uint32_t rawMode)
{
    uint8_t val = 0xff;
    SSP_DATA_SETUP_Type xferConfig;

   // LED7_CS_ON();

    if (ch >= '-' && ch <= '|') {
        val = chars[ch-'-'];
    }

    if (rawMode) {
        val = ch;
    }

	xferConfig.tx_data = &val;
	xferConfig.rx_data = NULL;
	xferConfig.length = 1;


    LED7_CS_ON();
    //SSP_SendData(LPC_SSP1, val);
    SSP_ReadWrite(LPC_SSP1, &xferConfig, SSP_TRANSFER_POLLING);

    LED7_CS_OFF();
}

