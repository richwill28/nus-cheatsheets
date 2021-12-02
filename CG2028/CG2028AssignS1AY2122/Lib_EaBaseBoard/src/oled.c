/*****************************************************************************
 *   light.c:  Driver for the ISL29003 Light-to-Digital Output Sensor
 *
 *   Copyright(C) 2009, Embedded Artists AB
 *   All rights reserved.
 *
 ******************************************************************************/

/*
 * NOTE: I2C or SPI must have been initialized before calling any functions in
 * this file.
 *
 *
 */

/******************************************************************************
 * Includes
 *****************************************************************************/

#include <string.h>
#include "lpc17xx_gpio.h"
#include "lpc17xx_i2c.h"
#include "lpc17xx_ssp.h"
#include "oled.h"
#include "font5x7.h"

/******************************************************************************
 * Defines and typedefs
 *****************************************************************************/

//#define OLED_USE_I2C

#ifdef OLED_USE_I2C
#define I2CDEV LPC_I2C2
#define OLED_I2C_ADDR (0x3c)
#else

#define OLED_CS_OFF() GPIO_SetValue( 0, (1<<6) )
#define OLED_CS_ON()  GPIO_ClearValue( 0, (1<<6) )
#define OLED_DATA()   GPIO_SetValue( 2, (1<<7) )
#define OLED_CMD()    GPIO_ClearValue( 2, (1<<7) )

#endif

/*
 * The display controller can handle a resolutiom of 132x64. The OLED
 * on the base board is 96x64.
 */
#define X_OFFSET 18

#define SHADOW_FB_SIZE (OLED_DISPLAY_WIDTH*OLED_DISPLAY_HEIGHT >> 3)

#define setAddress(page,lowerAddr,higherAddr)\
    writeCommand(page);\
    writeCommand(lowerAddr);\
    writeCommand(higherAddr);

/******************************************************************************
 * External global variables
 *****************************************************************************/


/******************************************************************************
 * Local variables
 *****************************************************************************/


/*
 * The SSD1305 doesn't support reading from the display memory when using
 * serial mode (only parallel mode). Since it isn't possible to write only
 * one pixel to the display (a minimum of one column, 8 pixels, is always
 * wriiten) a shadow framebuffer is needed to keep track of the display
 * data.
 */
static uint8_t shadowFB[SHADOW_FB_SIZE];

static uint8_t const  font_mask[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};


/******************************************************************************
 * Local Functions
 *****************************************************************************/
#ifdef OLED_USE_I2C
static int I2CWrite(uint8_t addr, uint8_t* buf, uint32_t len)
{
	I2C_M_SETUP_Type txsetup;

	txsetup.sl_addr7bit = addr;
	txsetup.tx_data = buf;
	txsetup.tx_length = len;
	txsetup.rx_data = NULL;
	txsetup.rx_length = 0;
	txsetup.retransmissions_max = 3;

	if (I2C_MasterTransferData(I2CDEV, &txsetup, I2C_TRANSFER_POLLING) == SUCCESS){
		return (0);
	} else {
		return (-1);
	}
}
#endif

/******************************************************************************
 *
 * Description:
 *    Write a command to the display
 *
 * Params:
 *   [in] data - command to write to the display
 *
 *****************************************************************************/
static void
writeCommand(uint8_t data)
{

#ifdef OLED_USE_I2C
    uint8_t buf[2];


    buf[0] = 0x00; // write Co & D/C bits
    buf[1] = data; // data

    I2CWrite(OLED_I2C_ADDR, buf, 2);

#else
    SSP_DATA_SETUP_Type xferConfig;
    OLED_CMD();
    OLED_CS_ON();

	xferConfig.tx_data = &data;
	xferConfig.rx_data = NULL;
	xferConfig.length  = 1;

    SSP_ReadWrite(LPC_SSP1, &xferConfig, SSP_TRANSFER_POLLING);
    //SSPSend( (uint8_t *)&data, 1 );

    OLED_CS_OFF();
#endif
}

/******************************************************************************
 *
 * Description:
 *    Write data to the display
 *
 * Params:
 *   [in] data - data (color) to write to the display
 *
 *****************************************************************************/
static void
writeData(uint8_t data)
{
#ifdef OLED_USE_I2C
    uint8_t buf[2];

    buf[0] = 0x40; // write Co & D/C bits
    buf[1] = data; // data

    I2CWrite(OLED_I2C_ADDR, buf, 2);


#else
    SSP_DATA_SETUP_Type xferConfig;
    OLED_DATA();
    OLED_CS_ON();

	xferConfig.tx_data = &data;
	xferConfig.rx_data = NULL;
	xferConfig.length  = 1;

    SSP_ReadWrite(LPC_SSP1, &xferConfig, SSP_TRANSFER_POLLING);
    //SSPSend( (uint8_t *)&data, 1 );

    OLED_CS_OFF();
#endif
}

/******************************************************************************
 *
 * Description:
 *    Write len number of data to the display
 *
 * Params:
 *   [in] data - data (color) to write to the display
 *   [in] len  - number of bytes to write
 *
 *****************************************************************************/
static void
writeDataLen(unsigned char data, unsigned int len)
{
    // TODO: optimize (at least from a RAM point of view)

#ifdef OLED_USE_I2C
    int i;
    uint8_t buf[140];

    buf[0] = 0x40; // write Co & D/C bits

    for (i = 1; i < len+1; i++) {
        buf[i] = data;
    }

    I2CWrite(OLED_I2C_ADDR, buf, len+1);

#else
    int i;
    uint8_t buf[140];
    SSP_DATA_SETUP_Type xferConfig;

    for (i = 0; i < len; i++) {
        buf[i] = data;
    }

    OLED_DATA();
    OLED_CS_ON();

	xferConfig.tx_data = buf;
	xferConfig.rx_data = NULL;
	xferConfig.length  = len;

    SSP_ReadWrite(LPC_SSP1, &xferConfig, SSP_TRANSFER_POLLING);

    //SSPSend( (uint8_t *)buf, len );

    OLED_CS_OFF();
#endif
}


/******************************************************************************
 *
 * Description:
 *    Run display init sequence
 *
 *****************************************************************************/
static void
runInitSequence(void)
{
    /*
     * Recommended Initial code according to manufacturer
     */

    writeCommand(0x02);//set low column address
    writeCommand(0x12);//set high column address
    writeCommand(0x40);//(display start set)
    writeCommand(0x2e);//(stop horzontal scroll)
    writeCommand(0x81);//(set contrast control register)
    writeCommand(0x32);
    writeCommand(0x82);//(brightness for color banks)
    writeCommand(0x80);//(display on)
    writeCommand(0xa1);//(set segment re-map)
    writeCommand(0xa6);//(set normal/inverse display)
    //  writeCommand(0xa7);//(set inverse display)
    writeCommand(0xa8);//(set multiplex ratio)
    writeCommand(0x3F);
    writeCommand(0xd3);//(set display offset)
    writeCommand(0x40);
    writeCommand(0xad);//(set dc-dc on/off)
    writeCommand(0x8E);//
    writeCommand(0xc8);//(set com output scan direction)
    writeCommand(0xd5);//(set display clock divide ratio/oscillator/frequency)
    writeCommand(0xf0);//
    writeCommand(0xd8);//(set area color mode on/off & low power display mode )
    writeCommand(0x05);//
    writeCommand(0xd9);//(set pre-charge period)
    writeCommand(0xF1);
    writeCommand(0xda);//(set com pins hardware configuration)
    writeCommand(0x12);
    writeCommand(0xdb);//(set vcom deselect level)
    writeCommand(0x34);
    writeCommand(0x91);//(set look up table for area color)
    writeCommand(0x3f);
    writeCommand(0x3f);
    writeCommand(0x3f);
    writeCommand(0x3f);
    writeCommand(0xaf);//(display on)
    writeCommand(0xa4);//(display on)
}


/******************************************************************************
 *
 * Description:
 *    Draw a horizontal line
 *
 * Params:
 *   [in] x0 - start x position
 *   [in] y0 - y position
 *   [in] x1 - end y position
 *   [in] color - color of the pixel
 *
 *****************************************************************************/
static void hLine(uint8_t x0, uint8_t y0, uint8_t x1, oled_color_t color)
{
    // TODO: optimize

    uint8_t bak;

    if (x0 > x1)
    {
        bak = x1;
        x1 = x0;
        x0 = bak;
    }

    while(x1 >= x0)
    {
        oled_putPixel(x0, y0, color);
        x0++;
    }
}

/******************************************************************************
 *
 * Description:
 *    Draw a vertical line
 *
 * Params:
 *   [in] x0 - x position
 *   [in] y0 - start y position
 *   [in] y1 - end y position
 *   [in] color - color of the pixel
 *
 *****************************************************************************/
static void vLine(uint8_t x0, uint8_t y0, uint8_t y1, oled_color_t color)
{
    uint8_t bak;

    if(y0 > y1)
    {
        bak = y1;
        y1 = y0;
        y0 = bak;
    }

    while(y1 >= y0)
    {
        oled_putPixel(x0, y0, color);
        y0++;
    }
    return;
}


/******************************************************************************
 * Public Functions
 *****************************************************************************/

/******************************************************************************
 *
 * Description:
 *    Initialize the OLED Display
 *
 *****************************************************************************/
void oled_init (void)
{
    int i = 0;

    //GPIO_SetDir(PORT0, 0, 1);
    GPIO_SetDir(2, (1<<1), 1);
    GPIO_SetDir(2, (1<<7), 1);
    GPIO_SetDir(0, (1<<6), 1);

    /* make sure power is off */
    GPIO_ClearValue( 2, (1<<1) );

#ifdef OLED_USE_I2C
    GPIO_ClearValue( 2, (1<<7)); // D/C#
    GPIO_ClearValue( 0, (1<<6)); // CS#
#else
    OLED_CS_OFF();
#endif

    runInitSequence();

    memset(shadowFB, 0, SHADOW_FB_SIZE);

    /* small delay before turning on power */
    for (i = 0; i < 0xffff; i++);

     /* power on */
    GPIO_SetValue( 2, (1<<1) );
}

/******************************************************************************
 *
 * Description:
 *    Draw one pixel on the display
 *
 * Params:
 *   [in] x - x position
 *   [in] y - y position
 *   [in] color - color of the pixel
 *
 *****************************************************************************/
void oled_putPixel(uint8_t x, uint8_t y, oled_color_t color) {
    uint8_t page;
    uint16_t add;
    uint8_t lAddr;
    uint8_t hAddr;
    uint8_t mask;
    uint32_t shadowPos = 0;

    if (x > OLED_DISPLAY_WIDTH) {
        return;
    }
    if (y > OLED_DISPLAY_HEIGHT) {
        return;
    }

    /* page address */
         if(y < 8)  page = 0xB0;
    else if(y < 16) page = 0xB1;
    else if(y < 24) page = 0xB2;
    else if(y < 32) page = 0xB3;
    else if(y < 40) page = 0xB4;
    else if(y < 48) page = 0xB5;
    else if(y < 56) page = 0xB6;
    else            page = 0xB7;

    add = x + X_OFFSET;
    lAddr = 0x0F & add;             // Low address
    hAddr = 0x10 | (add >> 4);      // High address

    // Calculate mask from rows basically do a y%8 and remainder is bit position
    add = y>>3;                     // Divide by 8
    add <<= 3;                      // Multiply by 8
    add = y - add;                  // Calculate bit position
    mask = 1 << add;                // Left shift 1 by bit position

    setAddress(page, lAddr, hAddr); // Set the address (sets the page,
                                    // lower and higher column address pointers)

    shadowPos = (page-0xB0)*OLED_DISPLAY_WIDTH+x;

    if(color > 0)
        shadowFB[shadowPos] |= mask;
    else
        shadowFB[shadowPos] &= ~mask;


    writeData(shadowFB[shadowPos]);
}

/******************************************************************************
 *
 * Description:
 *    Draw a line on the display starting at x0,y0 and ending at x1,y1
 *
 * Params:
 *   [in] x0 - start x position
 *   [in] y0 - start y position
 *   [in] x1 - end x position
 *   [in] y1 - end y position
 *   [in] color - color of the line
 *
 *****************************************************************************/
void oled_line(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1, oled_color_t color)
{
    int16_t   dx = 0, dy = 0;
    int8_t    dx_sym = 0, dy_sym = 0;
    int16_t   dx_x2 = 0, dy_x2 = 0;
    int16_t   di = 0;

    dx = x1-x0;
    dy = y1-y0;


    if(dx == 0)           /* vertical line */
    {
        vLine(x0, y0, y1, color);
        return;
    }

    if(dx > 0)
    {
        dx_sym = 1;
    }
    else
    {
        dx_sym = -1;
    }


    if(dy == 0)           /* horizontal line */
    {
        hLine(x0, y0, x1, color);
        return;
    }


    if(dy > 0)
    {
        dy_sym = 1;
    }
    else
    {
        dy_sym = -1;
    }

    dx = dx_sym*dx;
    dy = dy_sym*dy;

    dx_x2 = dx*2;
    dy_x2 = dy*2;

    if(dx >= dy)
    {
        di = dy_x2 - dx;
        while(x0 != x1)
        {

            oled_putPixel(x0, y0, color);
            x0 += dx_sym;
            if(di<0)
            {
                di += dy_x2;
            }
            else
            {
                di += dy_x2 - dx_x2;
                y0 += dy_sym;
            }
        }
        oled_putPixel(x0, y0, color);
    }
    else
    {
        di = dx_x2 - dy;
        while(y0 != y1)
        {
            oled_putPixel(x0, y0, color);
            y0 += dy_sym;
            if(di < 0)
            {
                di += dx_x2;
            }
            else
            {
                di += dx_x2 - dy_x2;
                x0 += dx_sym;
            }
        }
        oled_putPixel(x0, y0, color);
    }
    return;
}

/******************************************************************************
 *
 * Description:
 *    Draw a circle on the display starting at x0,y0 with radius r
 *
 * Params:
 *   [in] x0 - start x position
 *   [in] y0 - start y position
 *   [in] r - radius
 *   [in] color - color of the circle
 *
 *****************************************************************************/
void oled_circle(uint8_t x0, uint8_t y0, uint8_t r, oled_color_t color)
{
    int16_t draw_x0, draw_y0;
    int16_t draw_x1, draw_y1;
    int16_t draw_x2, draw_y2;
    int16_t draw_x3, draw_y3;
    int16_t draw_x4, draw_y4;
    int16_t draw_x5, draw_y5;
    int16_t draw_x6, draw_y6;
    int16_t draw_x7, draw_y7;
    int16_t xx, yy;
    int16_t  di;

    if(r == 0)          /* no radius */
    {
        return;
    }

    draw_x0 = draw_x1 = x0;
    draw_y0 = draw_y1 = y0 + r;
    if(draw_y0 < OLED_DISPLAY_HEIGHT)
    {
        oled_putPixel(draw_x0, draw_y0, color);     /* 90 degree */
    }

    draw_x2 = draw_x3 = x0;
    draw_y2 = draw_y3 = y0 - r;
    if(draw_y2 >= 0)
    {
        oled_putPixel(draw_x2, draw_y2, color);    /* 270 degree */
    }

    draw_x4 = draw_x6 = x0 + r;
    draw_y4 = draw_y6 = y0;
    if(draw_x4 < OLED_DISPLAY_WIDTH)
    {
        oled_putPixel(draw_x4, draw_y4, color);     /* 0 degree */
    }

    draw_x5 = draw_x7 = x0 - r;
    draw_y5 = draw_y7 = y0;
    if(draw_x5>=0)
    {
        oled_putPixel(draw_x5, draw_y5, color);     /* 180 degree */
    }

    if(r == 1)
    {
        return;
    }

    di = 3 - 2*r;
    xx = 0;
    yy = r;
    while(xx < yy)
    {

        if(di < 0)
        {
            di += 4*xx + 6;
        }
        else
        {
            di += 4*(xx - yy) + 10;
            yy--;
            draw_y0--;
            draw_y1--;
            draw_y2++;
            draw_y3++;
            draw_x4--;
            draw_x5++;
            draw_x6--;
            draw_x7++;
        }
        xx++;
        draw_x0++;
        draw_x1--;
        draw_x2++;
        draw_x3--;
        draw_y4++;
        draw_y5++;
        draw_y6--;
        draw_y7--;

        if( (draw_x0 <= OLED_DISPLAY_WIDTH) && (draw_y0>=0) )
        {
            oled_putPixel(draw_x0, draw_y0, color);
        }

        if( (draw_x1 >= 0) && (draw_y1 >= 0) )
        {
            oled_putPixel(draw_x1, draw_y1, color);
        }

        if( (draw_x2 <= OLED_DISPLAY_WIDTH) && (draw_y2 <= OLED_DISPLAY_HEIGHT) )
        {
            oled_putPixel(draw_x2, draw_y2, color);
        }

        if( (draw_x3 >=0 ) && (draw_y3 <= OLED_DISPLAY_HEIGHT) )
        {
            oled_putPixel(draw_x3, draw_y3, color);
        }

        if( (draw_x4 <= /*OLED_DISPLAY_HEIGHT*/OLED_DISPLAY_WIDTH) && (draw_y4 >= 0) )
        {
            oled_putPixel(draw_x4, draw_y4, color);
        }

        if( (draw_x5 >= 0) && (draw_y5 >= 0) )
        {
            oled_putPixel(draw_x5, draw_y5, color);
        }
        if( (draw_x6 <= OLED_DISPLAY_WIDTH) && (draw_y6 <= OLED_DISPLAY_HEIGHT) )
        {
            oled_putPixel(draw_x6, draw_y6, color);
        }
        if( (draw_x7 >= 0) && (draw_y7 <= OLED_DISPLAY_HEIGHT) )
        {
            oled_putPixel(draw_x7, draw_y7, color);
        }
    }
    return;
}

/******************************************************************************
 *
 * Description:
 *    Draw a rectangle on the display starting at x0,y0 and ending at x1,y1
 *
 * Params:
 *   [in] x0 - start x position
 *   [in] y0 - start y position
 *   [in] x1 - end x position
 *   [in] y1 - end y position
 *   [in] color - color of the rectangle
 *
 *****************************************************************************/
void oled_rect(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1, oled_color_t color)
{
    hLine(x0, y0, x1, color);
    hLine(x0, y1, x1, color);
    vLine(x0, y0, y1, color);
    vLine(x1, y0, y1, color);
}

/******************************************************************************
 *
 * Description:
 *    Fill a rectangle on the display starting at x0,y0 and ending at x1,y1
 *
 * Params:
 *   [in] x0 - start x position
 *   [in] y0 - start y position
 *   [in] x1 - end x position
 *   [in] y1 - end y position
 *   [in] color - color of the rectangle
 *
 *****************************************************************************/
void oled_fillRect(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1, oled_color_t color)
{
    uint8_t i = 0;

    if(x0 > x1)
    {
        i  = x0;
        x0 = x1;
        x1 = i;
    }

    if(y0 > y1)
    {
        i  = y0;
        y0 = y1;
        y1 = i;
    }

    if(y0 == y1)
    {
        hLine(x0, y0, x1, color);
        return;
    }

    if(x0 == x1)
    {
        vLine(x0, y0, y1, color);
        return;
    }

    while(y0 <= y1)
    {
        hLine(x0, y0, x1, color);
        y0++;
    }
    return;
}

/******************************************************************************
 *
 * Description:
 *    Clear the entire screen
 *
 * Params:
 *   [in] color - color to fill the screen with
 *
 *****************************************************************************/
void oled_clearScreen(oled_color_t color)
{
    uint8_t i;
    uint8_t c = 0;

    if (color == OLED_COLOR_WHITE)
        c = 0xff;


    for(i=0xB0;i<0xB8;i++) {            // Go through all 8 pages
        setAddress(i,0x00,0x10);
        writeDataLen(c, 132);
    }

    memset(shadowFB, c, SHADOW_FB_SIZE);
}

uint8_t oled_putChar(uint8_t x, uint8_t y, uint8_t ch, oled_color_t fb, oled_color_t bg)
{
    unsigned char data = 0;
    unsigned char i = 0, j = 0;

    oled_color_t color = OLED_COLOR_BLACK;

    if((x >= (OLED_DISPLAY_WIDTH - 8)) || (y >= (OLED_DISPLAY_HEIGHT - 8)) )
    {
        return 0;
    }

    if( (ch < 0x20) || (ch > 0x7f) )
    {
        ch = 0x20;      /* unknown character will be set to blank */
    }

    ch -= 0x20;
    for(i=0; i<8; i++)
    {
        data = font5x7[ch][i];
        for(j=0; j<6; j++)
        {
            if( (data&font_mask[j])==0 )
            {
                color = bg;
            }
            else
            {
                color = fb;
            }
            oled_putPixel(x, y, color);
            x++;
        }
        y++;
        x -= 6;
    }
    return( 1 );
}

void oled_putString(uint8_t x, uint8_t y, uint8_t *pStr, oled_color_t fb,
        oled_color_t bg)
{
  while(1)
  {
      if( (*pStr)=='\0' )
      {
          break;
      }
      if( oled_putChar(x, y, *pStr++, fb, bg) == 0 )
      {
        break;
    }
    x += 6;
  }
  return;
}
