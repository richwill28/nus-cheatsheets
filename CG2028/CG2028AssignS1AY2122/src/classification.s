    .syntax unified
    .cpu cortex-m3
    .thumb
    .align 2
    .global classification
    .thumb_func

@ CG2028 Assignment, Sem 1, AY 2021/22
@ (c) CG2028 Teaching Team, ECE NUS, 2021

@ student 1: Name: Richard Willie, Matriculation No.: A0219710L
@ student 2: Name: Hu Jialun, Matriculation No.: A0211251B

@ Register map
@ R0 - Stores the class of the nearest point
@ R1 - Stores the memory address of points[0]
@ R2 - Stores the memory address of label[0]
@ R3 - Stores the memory address of sample[0]
@ R4 - Multiple duties during different stages of the program:
@        * Stores the loop counter before passing its value to R6
@        * Stores the y-coordinate of the neighbouring data points
@        * Stores the y-difference of the data points to the sample point
@        * Stores the squared of the distances from the neighbouring points
@ R5 - Multiple duties during different stages of the program:
@        * Stores the y-coordinate of the sample point, i.e., sample[1]
@        * Stores the x-coordinate of the neighbouring data points
@        * Stores the x-difference of the data points to the sample point
@        * Stores the squared of the x-differences
@ R6 - Stores i = N-1, ..., 0, it is used as a counter for the loop
@ R7 - Stores the x-coordinate of the sample point, i.e., sample[0]
@ R12 (IP) - Stores the smallest distance from the neighbouring points

classification:
    PUSH {R4, R5, R6, R7, LR}       @ Push registers

    MVN IP, #-2147483648            @ IP = INT_MAX (32 bits)

    MOV R4, R0                      @ R4 = N
                                    @ 0b0000 0001 1010 0100 0100 0000 0000 0000
                                    @ 0x01A44000

    MOVS R0, #0                     @ R0 = 0, conveniently clearing flags
                                    @ 0b0000 0011 1011 0000 0000 0000 0000 0000
                                    @ 0x03B00000

    ADD LR, R1, #4                  @ LR = &points[1]
                                    @ 0b0000 0011 1000 0001 1110 0000 0000 0100
                                    @ 0x0381E004

.CONT:
    SUB R6, R4, #1                  @ Use R6 as loop counter, i--
                                    @ 0b0000 0010 0100 0100 0110 0000 0000 0001
                                    @ 0x02446001

    CMP R6, #0                      @ 0b0000 0011 0101 0110 0110 0000 0000 0000
                                    @ 0x03566000

    BGE .LOOP                       @ If i >= 0, jump to .LOOP
                                    @ 0b1010 1000 1000 0000 0000 0000 0000 0100
                                    @ 0xA8800004

    POP {R4, R5, R6, R7, PC}        @ Pop registers and return to caller

.LOOP:
    LDR R5, [R3, #4]                @ R5 = sample[1] = y-coordinate of the sample point
                                    @ 0b0000 0101 1001 0011 0101 0000 0000 0100
                                    @ 0x05935004

    LDR R4, [LR, R6, LSL #3]        @ R4 = points[2i+1] = y-coordinate of the ith data point

    LDR R7, [R3]                    @ R7 = sample[0] = x-coordinate of the sample point
                                    @ 0b0000 0101 1001 0011 0111 0000 0000 0000
                                    @ 0x05937000

    SUBS R4, R5                     @ R4 = y-difference
                                    @ 0b0000 0000 1001 0100 0100 0000 0000 0101
                                    @ 0x00944005

    LDR R5, [R1, R6, LSL #3]        @ R5 = points[2i] = x-coordinate of the ith data point

    SUBS R5, R7                     @ R5 = x-difference
                                    @ 0b0000 0000 1001 0101 0101 0000 0000 0111
                                    @ 0x00955007

    MULS R5, R5                     @ R5 = square of the x-difference
                                    @ 0b0000 0000 0001 0000 0101 0101 0001 0101
                                    @ 0x00105515

    MLA R4, R4, R4, R5              @ R4 = square of the distance
                                    @ 0b0000 0000 0010 0101 0100 0100 0001 0100
                                    @ 0x00254414

    CMP R4, IP                      @ 0b0000 0001 0101 0100 0100 0000 0000 1100
                                    @ 0x0154400C

    ITT LT                          @ If distance is less than current minimum
    MOVLT IP, R4                    @ Update IP to the new minimum distance
    LDRLT R0, [R2, R6, LSL #2]      @ R0 = class of the nearest neighbour

    MOV R4, R6                      @ Reassign R4 as loop counter
                                    @ 0b0000 0001 1010 0100 0100 0000 0000 0110
                                    @ 0x01A44006

    B .CONT                         @ Jump to .CONT
                                    @ 0b1110 1000 0000 0000 0000 0000 0011 0100
                                    @ 0xE8000034
