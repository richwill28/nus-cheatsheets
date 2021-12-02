/*****************************************************************************
 *   rgb.h:  Header file for RGB driver
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
******************************************************************************/
#ifndef __RGB_H
#define __RGB_H


#define RGB_RED   0x01
#define RGB_BLUE  0x02
#define RGB_GREEN 0x04


void rgb_init (void);
void rgb_setLeds (uint8_t ledMask);


#endif /* end __RGB_H */
/****************************************************************************
**                            End Of File
*****************************************************************************/
