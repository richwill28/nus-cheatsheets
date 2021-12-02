   1              		.syntax unified
   2              		.cpu cortex-m3
   3              		.fpu softvfp
   4              		.eabi_attribute 20, 1
   5              		.eabi_attribute 21, 1
   6              		.eabi_attribute 23, 3
   7              		.eabi_attribute 24, 1
   8              		.eabi_attribute 25, 1
   9              		.eabi_attribute 26, 1
  10              		.eabi_attribute 30, 6
  11              		.eabi_attribute 34, 1
  12              		.eabi_attribute 18, 4
  13              		.thumb
  14              		.syntax unified
  15              		.file	"main.c"
  16              		.text
  17              	.Ltext0:
  18              		.cfi_sections	.debug_frame
  19              		.section	.rodata
  20              		.align	2
  21              	.LC3:
  22 0000 61736D3A 		.ascii	"asm: class = %d \012\000"
  22      20636C61 
  22      7373203D 
  22      20256420 
  22      0A00
  23 0012 0000     		.align	2
  24              	.LC4:
  25 0014 4320203A 		.ascii	"C  : class = %d \012\000"
  25      20636C61 
  25      7373203D 
  25      20256420 
  25      0A00
  26 0026 0000     		.align	2
  27              	.LC0:
  28 0028 23000000 		.word	35
  29 002c 00000000 		.word	0
  30 0030 00000000 		.word	0
  31 0034 0F000000 		.word	15
  32 0038 0A000000 		.word	10
  33 003c 0A000000 		.word	10
  34 0040 0A000000 		.word	10
  35 0044 00000000 		.word	0
  36 0048 1E000000 		.word	30
  37 004c 00000000 		.word	0
  38 0050 1E000000 		.word	30
  39 0054 0A000000 		.word	10
  40 0058 28000000 		.word	40
  41 005c 00000000 		.word	0
  42 0060 28000000 		.word	40
  43 0064 0A000000 		.word	10
  44              		.align	2
  45              	.LC1:
  46 0068 01000000 		.word	1
  47 006c 01000000 		.word	1
  48 0070 00000000 		.word	0
  49 0074 00000000 		.word	0
  50 0078 01000000 		.word	1
  51 007c 00000000 		.word	0
  52 0080 01000000 		.word	1
  53 0084 00000000 		.word	0
  54              		.align	2
  55              	.LC2:
  56 0088 0F000000 		.word	15
  57 008c 14000000 		.word	20
  58              		.section	.text.main,"ax",%progbits
  59              		.align	2
  60              		.global	main
  61              		.thumb
  62              		.thumb_func
  64              	main:
  65              	.LFB0:
  66              		.file 1 "../src/main.c"
   1:../src/main.c **** #include "stdio.h"
   2:../src/main.c **** #define k 1
   3:../src/main.c **** 
   4:../src/main.c **** 
   5:../src/main.c **** // CG2028 Assignment, Sem 1, AY 2021/22
   6:../src/main.c **** // (c) CG2028 Teaching Team, ECE NUS, 2021
   7:../src/main.c **** 
   8:../src/main.c **** extern int classification(int N, int* points, int* label, int* sample); // asm implementation
   9:../src/main.c **** int classification_c(int N, int* points, int* label, int* sample); // reference C implementation
  10:../src/main.c **** 
  11:../src/main.c **** int main(void)
  12:../src/main.c **** {
  67              		.loc 1 12 0
  68              		.cfi_startproc
  69              		@ args = 0, pretend = 0, frame = 112
  70              		@ frame_needed = 1, uses_anonymous_args = 0
  71 0000 B0B5     		push	{r4, r5, r7, lr}
  72              		.cfi_def_cfa_offset 16
  73              		.cfi_offset 4, -16
  74              		.cfi_offset 5, -12
  75              		.cfi_offset 7, -8
  76              		.cfi_offset 14, -4
  77 0002 9CB0     		sub	sp, sp, #112
  78              		.cfi_def_cfa_offset 128
  79 0004 00AF     		add	r7, sp, #0
  80              		.cfi_def_cfa_register 7
  13:../src/main.c **** 	//variables
  14:../src/main.c **** 	int N = 8;
  81              		.loc 1 14 0
  82 0006 0823     		movs	r3, #8
  83 0008 FB66     		str	r3, [r7, #108]
  15:../src/main.c **** 	// think of the values below as numbers of the form x.y (decimal fixed point with 1 fractional dec
  16:../src/main.c **** 	// which are scaled up to allow them to be used integers
  17:../src/main.c **** 
  18:../src/main.c **** 	int points[16] = {35, 0, 0, 15, 10, 10, 10, 0, 30, 0, 30, 10, 40, 0, 40, 10};
  84              		.loc 1 18 0
  85 000a 1C4B     		ldr	r3, .L3
  86 000c 07F12C04 		add	r4, r7, #44
  87 0010 1D46     		mov	r5, r3
  88 0012 0FCD     		ldmia	r5!, {r0, r1, r2, r3}
  89 0014 0FC4     		stmia	r4!, {r0, r1, r2, r3}
  90 0016 0FCD     		ldmia	r5!, {r0, r1, r2, r3}
  91 0018 0FC4     		stmia	r4!, {r0, r1, r2, r3}
  92 001a 0FCD     		ldmia	r5!, {r0, r1, r2, r3}
  93 001c 0FC4     		stmia	r4!, {r0, r1, r2, r3}
  94 001e 95E80F00 		ldmia	r5, {r0, r1, r2, r3}
  95 0022 84E80F00 		stmia	r4, {r0, r1, r2, r3}
  19:../src/main.c **** 	int label[8] = {1, 1, 0, 0, 1, 0, 1, 0};
  96              		.loc 1 19 0
  97 0026 164B     		ldr	r3, .L3+4
  98 0028 07F10C04 		add	r4, r7, #12
  99 002c 1D46     		mov	r5, r3
 100 002e 0FCD     		ldmia	r5!, {r0, r1, r2, r3}
 101 0030 0FC4     		stmia	r4!, {r0, r1, r2, r3}
 102 0032 95E80F00 		ldmia	r5, {r0, r1, r2, r3}
 103 0036 84E80F00 		stmia	r4, {r0, r1, r2, r3}
  20:../src/main.c **** 	int sample[2] = {15, 20};
 104              		.loc 1 20 0
 105 003a 124A     		ldr	r2, .L3+8
 106 003c 3B1D     		adds	r3, r7, #4
 107 003e 92E80300 		ldmia	r2, {r0, r1}
 108 0042 83E80300 		stmia	r3, {r0, r1}
  21:../src/main.c **** 
  22:../src/main.c **** 	// Call assembly language function to perform classification
  23:../src/main.c **** 	printf( "asm: class = %d \n", classification(N, points, label, sample) ) ;
 109              		.loc 1 23 0
 110 0046 3B1D     		adds	r3, r7, #4
 111 0048 07F10C02 		add	r2, r7, #12
 112 004c 07F12C01 		add	r1, r7, #44
 113 0050 F86E     		ldr	r0, [r7, #108]
 114 0052 FFF7FEFF 		bl	classification
 115 0056 0346     		mov	r3, r0
 116 0058 1946     		mov	r1, r3
 117 005a 0B48     		ldr	r0, .L3+12
 118 005c FFF7FEFF 		bl	printf
  24:../src/main.c **** 	printf( "C  : class = %d \n", classification_c(N, points, label, sample) ) ;
 119              		.loc 1 24 0
 120 0060 3B1D     		adds	r3, r7, #4
 121 0062 07F10C02 		add	r2, r7, #12
 122 0066 07F12C01 		add	r1, r7, #44
 123 006a F86E     		ldr	r0, [r7, #108]
 124 006c FFF7FEFF 		bl	classification_c
 125 0070 0346     		mov	r3, r0
 126 0072 1946     		mov	r1, r3
 127 0074 0548     		ldr	r0, .L3+16
 128 0076 FFF7FEFF 		bl	printf
 129              	.L2:
  25:../src/main.c **** 
  26:../src/main.c **** 	while (1); //halt
 130              		.loc 1 26 0 discriminator 1
 131 007a FEE7     		b	.L2
 132              	.L4:
 133              		.align	2
 134              	.L3:
 135 007c 28000000 		.word	.LC0
 136 0080 68000000 		.word	.LC1
 137 0084 88000000 		.word	.LC2
 138 0088 00000000 		.word	.LC3
 139 008c 14000000 		.word	.LC4
 140              		.cfi_endproc
 141              	.LFE0:
 143              		.section	.text.classification_c,"ax",%progbits
 144              		.align	2
 145              		.global	classification_c
 146              		.thumb
 147              		.thumb_func
 149              	classification_c:
 150              	.LFB1:
  27:../src/main.c **** }
  28:../src/main.c **** 
  29:../src/main.c **** int classification_c(int N, int* points, int* label, int* sample)
  30:../src/main.c **** { 	// The implementation below is meant only for verifying your results.
 151              		.loc 1 30 0
 152              		.cfi_startproc
 153              		@ args = 0, pretend = 0, frame = 56
 154              		@ frame_needed = 1, uses_anonymous_args = 0
 155              		@ link register save eliminated.
 156 0000 2DE9F003 		push	{r4, r5, r6, r7, r8, r9}
 157              		.cfi_def_cfa_offset 24
 158              		.cfi_offset 4, -24
 159              		.cfi_offset 5, -20
 160              		.cfi_offset 6, -16
 161              		.cfi_offset 7, -12
 162              		.cfi_offset 8, -8
 163              		.cfi_offset 9, -4
 164 0004 8EB0     		sub	sp, sp, #56
 165              		.cfi_def_cfa_offset 80
 166 0006 00AF     		add	r7, sp, #0
 167              		.cfi_def_cfa_register 7
 168 0008 F860     		str	r0, [r7, #12]
 169 000a B960     		str	r1, [r7, #8]
 170 000c 7A60     		str	r2, [r7, #4]
 171 000e 3B60     		str	r3, [r7]
 172              		.loc 1 30 0
 173 0010 6B46     		mov	r3, sp
 174 0012 1E46     		mov	r6, r3
  31:../src/main.c **** 	
  32:../src/main.c **** 	int i,j,n;
  33:../src/main.c **** 	int class;// returned labels of k=1 nearest neighbors
  34:../src/main.c **** 	int d[N]; // squared Euclidean distance
 175              		.loc 1 34 0
 176 0014 F968     		ldr	r1, [r7, #12]
 177 0016 4B1E     		subs	r3, r1, #1
 178 0018 7B62     		str	r3, [r7, #36]
 179 001a 0B46     		mov	r3, r1
 180 001c 1A46     		mov	r2, r3
 181 001e 4FF00003 		mov	r3, #0
 182 0022 4FEA4319 		lsl	r9, r3, #5
 183 0026 49EAD269 		orr	r9, r9, r2, lsr #27
 184 002a 4FEA4218 		lsl	r8, r2, #5
 185 002e 0B46     		mov	r3, r1
 186 0030 1A46     		mov	r2, r3
 187 0032 4FF00003 		mov	r3, #0
 188 0036 5D01     		lsls	r5, r3, #5
 189 0038 45EAD265 		orr	r5, r5, r2, lsr #27
 190 003c 5401     		lsls	r4, r2, #5
 191 003e 0B46     		mov	r3, r1
 192 0040 9B00     		lsls	r3, r3, #2
 193 0042 0333     		adds	r3, r3, #3
 194 0044 0733     		adds	r3, r3, #7
 195 0046 DB08     		lsrs	r3, r3, #3
 196 0048 DB00     		lsls	r3, r3, #3
 197 004a ADEB030D 		sub	sp, sp, r3
 198 004e 6B46     		mov	r3, sp
 199 0050 0333     		adds	r3, r3, #3
 200 0052 9B08     		lsrs	r3, r3, #2
 201 0054 9B00     		lsls	r3, r3, #2
 202 0056 3B62     		str	r3, [r7, #32]
  35:../src/main.c **** 	int d1, d2, sum=0;
 203              		.loc 1 35 0
 204 0058 0023     		movs	r3, #0
 205 005a FB61     		str	r3, [r7, #28]
  36:../src/main.c **** 
  37:../src/main.c **** 	
  38:../src/main.c **** 	// calculate the squared distance between test sample and each training data points
  39:../src/main.c **** 	for (i=0; i<N; i++){
 206              		.loc 1 39 0
 207 005c 0023     		movs	r3, #0
 208 005e 7B63     		str	r3, [r7, #52]
 209 0060 31E0     		b	.L6
 210              	.L7:
  40:../src/main.c **** 		d[i] = (points[2*i]-sample[0]) * (points[2*i]-sample[0])
 211              		.loc 1 40 0 discriminator 3
 212 0062 7B6B     		ldr	r3, [r7, #52]
 213 0064 DB00     		lsls	r3, r3, #3
 214 0066 1A46     		mov	r2, r3
 215 0068 BB68     		ldr	r3, [r7, #8]
 216 006a 1344     		add	r3, r3, r2
 217 006c 1A68     		ldr	r2, [r3]
 218 006e 3B68     		ldr	r3, [r7]
 219 0070 1B68     		ldr	r3, [r3]
 220 0072 D31A     		subs	r3, r2, r3
 221 0074 7A6B     		ldr	r2, [r7, #52]
 222 0076 D200     		lsls	r2, r2, #3
 223 0078 1146     		mov	r1, r2
 224 007a BA68     		ldr	r2, [r7, #8]
 225 007c 0A44     		add	r2, r2, r1
 226 007e 1168     		ldr	r1, [r2]
 227 0080 3A68     		ldr	r2, [r7]
 228 0082 1268     		ldr	r2, [r2]
 229 0084 8A1A     		subs	r2, r1, r2
 230 0086 02FB03F2 		mul	r2, r2, r3
  41:../src/main.c **** 		+ (points[2*i+1]-sample[1])*(points[2*i+1]-sample[1]);
 231              		.loc 1 41 0 discriminator 3
 232 008a 7B6B     		ldr	r3, [r7, #52]
 233 008c DB00     		lsls	r3, r3, #3
 234 008e 0433     		adds	r3, r3, #4
 235 0090 B968     		ldr	r1, [r7, #8]
 236 0092 0B44     		add	r3, r3, r1
 237 0094 1968     		ldr	r1, [r3]
 238 0096 3B68     		ldr	r3, [r7]
 239 0098 0433     		adds	r3, r3, #4
 240 009a 1B68     		ldr	r3, [r3]
 241 009c CB1A     		subs	r3, r1, r3
 242 009e 796B     		ldr	r1, [r7, #52]
 243 00a0 C900     		lsls	r1, r1, #3
 244 00a2 0431     		adds	r1, r1, #4
 245 00a4 B868     		ldr	r0, [r7, #8]
 246 00a6 0144     		add	r1, r1, r0
 247 00a8 0868     		ldr	r0, [r1]
 248 00aa 3968     		ldr	r1, [r7]
 249 00ac 0431     		adds	r1, r1, #4
 250 00ae 0968     		ldr	r1, [r1]
 251 00b0 411A     		subs	r1, r0, r1
 252 00b2 01FB03F3 		mul	r3, r1, r3
 253 00b6 D118     		adds	r1, r2, r3
  40:../src/main.c **** 		d[i] = (points[2*i]-sample[0]) * (points[2*i]-sample[0])
 254              		.loc 1 40 0 discriminator 3
 255 00b8 3B6A     		ldr	r3, [r7, #32]
 256 00ba 7A6B     		ldr	r2, [r7, #52]
 257 00bc 43F82210 		str	r1, [r3, r2, lsl #2]
  39:../src/main.c **** 		d[i] = (points[2*i]-sample[0]) * (points[2*i]-sample[0])
 258              		.loc 1 39 0 discriminator 3
 259 00c0 7B6B     		ldr	r3, [r7, #52]
 260 00c2 0133     		adds	r3, r3, #1
 261 00c4 7B63     		str	r3, [r7, #52]
 262              	.L6:
  39:../src/main.c **** 		d[i] = (points[2*i]-sample[0]) * (points[2*i]-sample[0])
 263              		.loc 1 39 0 is_stmt 0 discriminator 1
 264 00c6 7A6B     		ldr	r2, [r7, #52]
 265 00c8 FB68     		ldr	r3, [r7, #12]
 266 00ca 9A42     		cmp	r2, r3
 267 00cc C9DB     		blt	.L7
  42:../src/main.c **** 	}
  43:../src/main.c **** 
  44:../src/main.c **** 	// print all distances
  45:../src/main.c **** 	//for (i=0; i<N; i++){
  46:../src/main.c **** 	//	printf( "d%d = %d, class = %d \n",i+1, d[i],label[i]) ;
  47:../src/main.c **** 	//}
  48:../src/main.c **** 
  49:../src/main.c **** 	// find the k=1 nearest neighbors
  50:../src/main.c **** 
  51:../src/main.c **** 	for (j=0; j<N; j++){
 268              		.loc 1 51 0 is_stmt 1
 269 00ce 0023     		movs	r3, #0
 270 00d0 3B63     		str	r3, [r7, #48]
 271 00d2 33E0     		b	.L8
 272              	.L13:
  52:../src/main.c **** 		d1 = d[j];
 273              		.loc 1 52 0
 274 00d4 3B6A     		ldr	r3, [r7, #32]
 275 00d6 3A6B     		ldr	r2, [r7, #48]
 276 00d8 53F82230 		ldr	r3, [r3, r2, lsl #2]
 277 00dc BB61     		str	r3, [r7, #24]
  53:../src/main.c **** 		for (n=0; n<N; n++){
 278              		.loc 1 53 0
 279 00de 0023     		movs	r3, #0
 280 00e0 FB62     		str	r3, [r7, #44]
 281 00e2 24E0     		b	.L9
 282              	.L12:
  54:../src/main.c **** 			d2 = d[n];
 283              		.loc 1 54 0
 284 00e4 3B6A     		ldr	r3, [r7, #32]
 285 00e6 FA6A     		ldr	r2, [r7, #44]
 286 00e8 53F82230 		ldr	r3, [r3, r2, lsl #2]
 287 00ec 7B61     		str	r3, [r7, #20]
  55:../src/main.c **** 			if (d1<d2 && d1>=0 && d2>=0){
 288              		.loc 1 55 0
 289 00ee BA69     		ldr	r2, [r7, #24]
 290 00f0 7B69     		ldr	r3, [r7, #20]
 291 00f2 9A42     		cmp	r2, r3
 292 00f4 0CDA     		bge	.L10
 293              		.loc 1 55 0 is_stmt 0 discriminator 1
 294 00f6 BB69     		ldr	r3, [r7, #24]
 295 00f8 002B     		cmp	r3, #0
 296 00fa 09DB     		blt	.L10
 297              		.loc 1 55 0 discriminator 2
 298 00fc 7B69     		ldr	r3, [r7, #20]
 299 00fe 002B     		cmp	r3, #0
 300 0100 06DB     		blt	.L10
  56:../src/main.c **** 				class = label[j];
 301              		.loc 1 56 0 is_stmt 1
 302 0102 3B6B     		ldr	r3, [r7, #48]
 303 0104 9B00     		lsls	r3, r3, #2
 304 0106 7A68     		ldr	r2, [r7, #4]
 305 0108 1344     		add	r3, r3, r2
 306 010a 1B68     		ldr	r3, [r3]
 307 010c BB62     		str	r3, [r7, #40]
 308 010e 0BE0     		b	.L11
 309              	.L10:
  57:../src/main.c **** 			}
  58:../src/main.c **** 			else if (d1>=0 && d2>=0){
 310              		.loc 1 58 0
 311 0110 BB69     		ldr	r3, [r7, #24]
 312 0112 002B     		cmp	r3, #0
 313 0114 08DB     		blt	.L11
 314              		.loc 1 58 0 is_stmt 0 discriminator 1
 315 0116 7B69     		ldr	r3, [r7, #20]
 316 0118 002B     		cmp	r3, #0
 317 011a 05DB     		blt	.L11
  59:../src/main.c **** 				class = label[n];
 318              		.loc 1 59 0 is_stmt 1
 319 011c FB6A     		ldr	r3, [r7, #44]
 320 011e 9B00     		lsls	r3, r3, #2
 321 0120 7A68     		ldr	r2, [r7, #4]
 322 0122 1344     		add	r3, r3, r2
 323 0124 1B68     		ldr	r3, [r3]
 324 0126 BB62     		str	r3, [r7, #40]
 325              	.L11:
  53:../src/main.c **** 			d2 = d[n];
 326              		.loc 1 53 0 discriminator 2
 327 0128 FB6A     		ldr	r3, [r7, #44]
 328 012a 0133     		adds	r3, r3, #1
 329 012c FB62     		str	r3, [r7, #44]
 330              	.L9:
  53:../src/main.c **** 			d2 = d[n];
 331              		.loc 1 53 0 is_stmt 0 discriminator 1
 332 012e FA6A     		ldr	r2, [r7, #44]
 333 0130 FB68     		ldr	r3, [r7, #12]
 334 0132 9A42     		cmp	r2, r3
 335 0134 D6DB     		blt	.L12
  51:../src/main.c **** 		d1 = d[j];
 336              		.loc 1 51 0 is_stmt 1 discriminator 2
 337 0136 3B6B     		ldr	r3, [r7, #48]
 338 0138 0133     		adds	r3, r3, #1
 339 013a 3B63     		str	r3, [r7, #48]
 340              	.L8:
  51:../src/main.c **** 		d1 = d[j];
 341              		.loc 1 51 0 is_stmt 0 discriminator 1
 342 013c 3A6B     		ldr	r2, [r7, #48]
 343 013e FB68     		ldr	r3, [r7, #12]
 344 0140 9A42     		cmp	r2, r3
 345 0142 C7DB     		blt	.L13
  60:../src/main.c **** 			}
  61:../src/main.c **** 		}
  62:../src/main.c **** 	}
  63:../src/main.c **** 	
  64:../src/main.c **** 	return class;
 346              		.loc 1 64 0 is_stmt 1
 347 0144 BB6A     		ldr	r3, [r7, #40]
 348 0146 B546     		mov	sp, r6
  65:../src/main.c **** }
 349              		.loc 1 65 0
 350 0148 1846     		mov	r0, r3
 351 014a 3837     		adds	r7, r7, #56
 352              		.cfi_def_cfa_offset 24
 353 014c BD46     		mov	sp, r7
 354              		.cfi_def_cfa_register 13
 355              		@ sp needed
 356 014e BDE8F003 		pop	{r4, r5, r6, r7, r8, r9}
 357              		.cfi_restore 9
 358              		.cfi_restore 8
 359              		.cfi_restore 7
 360              		.cfi_restore 6
 361              		.cfi_restore 5
 362              		.cfi_restore 4
 363              		.cfi_def_cfa_offset 0
 364 0152 7047     		bx	lr
 365              		.cfi_endproc
 366              	.LFE1:
 368              		.text
 369              	.Letext0:
DEFINED SYMBOLS
                            *ABS*:0000000000000000 main.c
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccHpFL8k.s:20     .rodata:0000000000000000 $d
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccHpFL8k.s:59     .text.main:0000000000000000 $t
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccHpFL8k.s:64     .text.main:0000000000000000 main
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccHpFL8k.s:149    .text.classification_c:0000000000000000 classification_c
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccHpFL8k.s:135    .text.main:000000000000007c $d
/var/folders/cb/5_t_trh565vdyjfzd6gzn9vh0000gn/T//ccHpFL8k.s:144    .text.classification_c:0000000000000000 $t
                     .debug_frame:0000000000000010 $d
                           .group:0000000000000000 wm4.0.0c5e979f1ec464b8f03bc190bd321363
                           .group:0000000000000000 wm4.redlib_version.h.14.62abddb5b4efb2dd619a7dca5647eb78
                           .group:0000000000000000 wm4.libconfigarm.h.18.48d18a57a6aa6fedadbcea02294a713f
                           .group:0000000000000000 wm4.stdio.h.44.4674ea39c56924a8b52db7db756a4792

UNDEFINED SYMBOLS
classification
printf
