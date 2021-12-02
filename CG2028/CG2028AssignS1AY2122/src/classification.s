 	.syntax unified
 	.cpu cortex-m3
 	.thumb
 	.align 2
 	.global	classification
 	.thumb_func

@ CG2028 Assignment, Sem 2, AY 2021/21
@ (c) CG2028 Teaching Team, ECE NUS, 2021

@ student 1: Name: , Matriculation No.:
@ student 2: Name: , Matriculation No.:
@ eg: student 1: Name: John Doe , Matriculation No.: A021234N

@Register map
@R0 - N, returns class
@R1 - points
@R2 - label
@R3 - sample
@R4 - <use(s)>
@R5 - <use(s)>
@....

classification:
@ PUSH / save (only those) registers which are modified by your function
		PUSH {R1-R4,R14}
@ parameter registers need not be saved.

@ write asm function body here

@ branch to SUBROUTINE for illustration only
		BL SUBROUTINE
@ prepare value to return (class) to C program in R0
@ the #5 here is an arbitrary result
		MOVW R0, #5
@ POP / restore original register values. DO NOT save or restore R0. Why?
		POP {R1-R4,R14}
@ return to C program
		BX	LR

@ you could write your code without SUBROUTINE
SUBROUTINE:

		BX LR

@label: .word value

@.lcomm label num_bytes
