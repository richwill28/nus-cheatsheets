/*****************************************************************************
 *   temp.h:  Header file for Temperature driver
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
******************************************************************************/
#ifndef __TEMP_H
#define __TEMP_H


void temp_init (uint32_t (*getMsTick)(void));
int32_t temp_read(void);


#endif /* end __TEMP_H */
/****************************************************************************
**                            End Of File
*****************************************************************************/
