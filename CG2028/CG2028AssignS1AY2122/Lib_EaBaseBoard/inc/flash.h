/*****************************************************************************
 *   oled.h:  Header file for OLED Display
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
******************************************************************************/
#ifndef __FLASH_H
#define __FLASH_H



uint32_t flash_init (void);
uint32_t flash_write(uint8_t* buf, uint32_t offset, uint32_t len);
uint32_t flash_read(uint8_t* buf, uint32_t offset, uint32_t len);

void flash_setToBinaryPageSize(void);
uint16_t flash_getPageSize(void);



#endif /* end __FLASH_H */
/****************************************************************************
**                            End Of File
*****************************************************************************/
