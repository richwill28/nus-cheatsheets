/*****************************************************************************
 *   rotary.c:  Driver for the Rotary Switch
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
 ******************************************************************************/

/*
 * NOTE: GPIOInit must have been called before using any functions in this
 * file.
 */

/******************************************************************************
 * Includes
 *****************************************************************************/


#include "lpc17xx_gpio.h"
#include "rotary.h"

/******************************************************************************
 * Defines and typedefs
 *****************************************************************************/

//#define ROTARY_READ_STATE() ( (LPC_GPIO2->DATA >> 1) & 0x03)
//#define ROTARY_READ_STATE() ( (LPC_GPIO1->DATA) & 0x03)
#define ROTARY_READ_STATE() ((GPIO_ReadValue(0) >> 24) & 0x03)

#define R_W  0
#define R_L1 1
#define R_L2 2
#define R_L3 3
#define R_R1 4
#define R_R2 5
#define R_R3 6

/******************************************************************************
 * External global variables
 *****************************************************************************/

/******************************************************************************
 * Local variables
 *****************************************************************************/

/******************************************************************************
 * Local Functions
 *****************************************************************************/

/******************************************************************************
 * Public Functions
 *****************************************************************************/

/******************************************************************************
 *
 * Description:
 *    Initialize Rotary switch driver
 *
 *****************************************************************************/
void rotary_init (void)
{
    GPIO_SetDir( 0, (1<<24 | 1 << 25), 0 );
    //GPIO_SetDir( 0, 25, 0 );
}


/******************************************************************************
 *
 * Description:
 *    Read rotary switch state
 *
 * Returns:
 *    rotary switch state
 *
 *****************************************************************************/
uint8_t rotary_read (void)
{
    /* p0.24 and p0.25 */
    uint8_t state = ROTARY_READ_STATE();
    uint8_t event = ROTARY_WAIT;
    uint8_t e = R_W;

    if (state == 0x03) {
        return ROTARY_WAIT;
    }

    while (state != 0x03) {

        switch (e) {
        case R_W:
            if (state == 0x02)
                e = R_R1;
            else if (state == 0x01)
                e = R_L1;
            break;

        case R_L1:
            if (state == 0x00)
                e = R_R2;
            break;
        case R_L2:
            if (state == 0x01) {
                e = R_R3;
                event = ROTARY_RIGHT;
            }
            break;
        case R_R1:
            if (state == 0x00)
                e = R_L2;
            break;
        case R_R2:
            if (state == 0x02) {
                e = R_L3;
                event = ROTARY_LEFT;
            }
            break;
        }

        state = ROTARY_READ_STATE();
    }

    return event;

}
