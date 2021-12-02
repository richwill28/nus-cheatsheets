/*****************************************************************************
 *   light.h:  Header file for ISL29003 Light-to-Digital Output Sensor
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
******************************************************************************/
#ifndef __LIGHT_H
#define __LIGHT_H


typedef enum
{
    LIGHT_MODE_D1,
    LIGHT_MODE_D2,
    LIGHT_MODE_D1D2
} light_mode_t;

typedef enum
{
    LIGHT_WIDTH_16BITS,
    LIGHT_WIDTH_12BITS,
    LIGHT_WIDTH_08BITS,
    LIGHT_WIDTH_04BITS
} light_width_t;

typedef enum
{
    LIGHT_RANGE_1000,
    LIGHT_RANGE_4000,
    LIGHT_RANGE_16000,
    LIGHT_RANGE_64000
} light_range_t;

typedef enum
{
    LIGHT_CYCLE_1,
    LIGHT_CYCLE_4,
    LIGHT_CYCLE_8,
    LIGHT_CYCLE_16
} light_cycle_t;


void light_init (void);
void light_enable (void);
uint32_t light_read(void);
void light_setMode(light_mode_t mode);
void light_setWidth(light_width_t width);
void light_setRange(light_range_t newRange);
void light_setHiThreshold(uint32_t luxTh);
void light_setLoThreshold(uint32_t luxTh);
void light_setIrqInCycles(light_cycle_t cycles);
uint8_t light_getIrqStatus(void);
void light_clearIrqStatus(void);
void light_shutdown(void);


#endif /* end __LIGHT_H */
/****************************************************************************
**                            End Of File
*****************************************************************************/
