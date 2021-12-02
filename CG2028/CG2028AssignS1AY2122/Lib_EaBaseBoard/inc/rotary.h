/*****************************************************************************
 *   rotary.h:  Header file for Rotary Switch driver
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
******************************************************************************/
#ifndef __ROTARY_H
#define __ROTARY_H

#define ROTARY_WAIT  0
#define ROTARY_RIGHT 1
#define ROTARY_LEFT  2


void rotary_init (void);
uint8_t rotary_read(void);


#endif /* end __ROTARY_H */
/****************************************************************************
**                            End Of File
*****************************************************************************/
