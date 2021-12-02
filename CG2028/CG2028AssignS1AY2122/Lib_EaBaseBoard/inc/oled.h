/*****************************************************************************
 *   oled.h:  Header file for OLED Display
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
******************************************************************************/
#ifndef __OLED_H
#define __OLED_H

#define OLED_DISPLAY_WIDTH  96
#define OLED_DISPLAY_HEIGHT 64


typedef enum
{
    OLED_COLOR_BLACK,
    OLED_COLOR_WHITE
} oled_color_t;


void oled_init (void);
void oled_putPixel(uint8_t x, uint8_t y, oled_color_t color);
void oled_line(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1, oled_color_t color);
void oled_circle(uint8_t x0, uint8_t y0, uint8_t r, oled_color_t color);
void oled_rect(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1, oled_color_t color);
void oled_fillRect(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1, oled_color_t color);
void oled_clearScreen(oled_color_t color);
void oled_putString(uint8_t x, uint8_t y, uint8_t *pStr, oled_color_t fb,
        oled_color_t bg);
uint8_t oled_putChar(uint8_t x, uint8_t y, uint8_t ch, oled_color_t fb, oled_color_t bg);


#endif /* end __OLED_H */
/****************************************************************************
**                            End Of File
*****************************************************************************/
