/*****************************************************************************
 *   joystick.c:  Driver for the Joystick switch
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
 ******************************************************************************/

/*
 * NOTE: GPIOInit must have been called before using any functions in this
 * file.
 *
 */


/******************************************************************************
 * Includes
 *****************************************************************************/


#include "lpc17xx_gpio.h"
#include "joystick.h"

/******************************************************************************
 * Defines and typedefs
 *****************************************************************************/

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
 *    Initialize the Joystick Driver
 *
 *****************************************************************************/
void joystick_init (void)
{
    /* set the GPIOs as inputs */
/*
    GPIO_SetDir( 0, 15, 0 );
    GPIO_SetDir( 0, 16, 0 );
    GPIO_SetDir( 0, 17, 0 );
    GPIO_SetDir( 2, 3, 0 );
    GPIO_SetDir( 2, 4, 0 );
*/
    GPIO_SetDir( 0, (1<<15|1<<16|1<<17), 0 );
    GPIO_SetDir( 2, (1<<3|1<<4), 0 );
}

/******************************************************************************
 *
 * Description:
 *    Read the joystick status
 *
 * Returns:
 *   The joystick status. The returned value is a bit mask. More than one
 *   direction may be active at any given time (e.g. UP and RIGHT)
 *
 *****************************************************************************/
uint8_t joystick_read(void)
{
    uint8_t status = 0;
    uint32_t port0 = 0;
    uint32_t port2 = 0;

    port0 = GPIO_ReadValue(0);
    port2 = GPIO_ReadValue(2);

    if (/*!GPIOGetValue( PORT2, 0)*/(port0 & (1 << 17)) == 0) {
        status |= JOYSTICK_CENTER;
    }

    if (/*!GPIOGetValue( PORT2, 1)*/ (port0 & (1 << 15)) == 0) {
        status |= JOYSTICK_DOWN;
    }

    if (/*!GPIOGetValue( PORT2, 2)*/ (port0 & (1 << 16)) == 0) {
        status |= JOYSTICK_RIGHT;
    }

    if (/*!GPIOGetValue( PORT2, 3)*/ (port2 & (1 << 3)) == 0) {
        status |= JOYSTICK_UP;
    }


    if (/*!GPIOGetValue( PORT2, 4)*/ (port2 & (1 << 4)) == 0) {
        status |= JOYSTICK_LEFT;
    }

    return status;
}





