/*****************************************************************************
 *   temp.c:  Driver for the Temp Sensor
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
#include "temp.h"

/******************************************************************************
 * Defines and typedefs
 *****************************************************************************/

/*
 * Time-Select Pin Configuration. Selected by Jumper J26 on the base board
 */
#define TEMP_TS1 0
#define TEMP_TS0 0

/*
 * Pin 0.2 or pin 1.5 can be used as input source for the temp sensor
 * Selected by jumper J25.
 */
//#define TEMP_USE_P0_2

#if TEMP_TS1 == 0 && TEMP_TS0 == 0
#define TEMP_SCALAR_DIV10 1
#define NUM_HALF_PERIODS 340
#elif TEMP_TS1 == 0 && TEMP_TS0 == 1
#define TEMP_SCALAR_DIV10 4
#define NUM_HALF_PERIODS 100
#elif TEMP_TS1 == 1 && TEMP_TS0 == 0
#define TEMP_SCALAR_DIV10 16
#define NUM_HALF_PERIODS 32
#elif TEMP_TS1 == 1 && TEMP_TS0 == 1
#define TEMP_SCALAR_DIV10 64
#define NUM_HALF_PERIODS 10
#endif


#define P0_6_STATE ((GPIO_ReadValue(0) & (1 << 6)) != 0)
#define P0_2_STATE ((GPIO_ReadValue(0) & (1 << 2)) != 0)


#ifdef TEMP_USE_P0_6
#define    GET_TEMP_STATE P0_6_STATE
#else
#define    GET_TEMP_STATE P0_2_STATE
#endif


/******************************************************************************
 * External global variables
 *****************************************************************************/

/******************************************************************************
 * Local variables
 *****************************************************************************/

static uint32_t (*getTicks)(void) = NULL;

/******************************************************************************
 * Local Functions
 *****************************************************************************/

/******************************************************************************
 * Public Functions
 *****************************************************************************/

/******************************************************************************
 *
 * Description:
 *    Initialize Temp Sensor driver
 *
 * Params:
 *   [in] getMsTicks - callback function for retrieving number of elapsed ticks
 *                     in milliseconds
 *
 *****************************************************************************/
void temp_init (uint32_t (*getMsTicks)(void))
{
#ifdef TEMP_USE_P0_6
    GPIO_SetDir( 0, (1<<6), 0 );
#else
    GPIO_SetDir( 0, (1<<2), 0 );
#endif
    getTicks = getMsTicks;
}

/******************************************************************************
 *
 * Description:
 *    Read temperature
 *
 * Returns:
 *    10 x T(c), i.e. 10 times the temperature in Celcius. Example:
 *    if the temperature is 22.4 degrees the returned value is 224.
 *
 *****************************************************************************/
int32_t temp_read (void)
{
    uint8_t state = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    int i = 0;

    /*
     * T(C) = ( period (us) / scalar ) - 273.15 K
     *
     * 10T(C) = (period (us) / scalar_div10) - 2731 K
     */

    state = GET_TEMP_STATE;

    /* get next state change before measuring time */
    while(GET_TEMP_STATE == state);
    state = !state;

    t1 = getTicks();

    for (i = 0; i < NUM_HALF_PERIODS; i++) {
        while(GET_TEMP_STATE == state);
        state = !state;
    }

    t2 = getTicks();
    if (t2 > t1) {
        t2 = t2-t1;
    }
    else {
        t2 = (0xFFFFFFFF - t1 + 1) + t2;
    }


    return ( (2*1000*t2) / (NUM_HALF_PERIODS*TEMP_SCALAR_DIV10) - 2731 );
}
