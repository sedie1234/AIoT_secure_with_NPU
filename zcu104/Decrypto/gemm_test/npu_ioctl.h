#ifndef _NPU_DEF_H_
#define _NPU_DEF_H_

#define NPU_IOCTL_MAGIC 'N'

#define NPU_IOCTL_POS		_IOWR(NPU_IOCTL_MAGIC, 1, void *)
#define NPU_IOCTL_RUN		_IOWR(NPU_IOCTL_MAGIC, 2, void *)

#define NPU_POS_A1		0
#define NPU_POS_A2		1
#define NPU_POS_A3		2
#define NPU_POS_B1		3
#define NPU_POS_B2		4
#define NPU_POS_C1		5
#define NPU_POS_C2		6
#define NPU_POS_C3		7
#define NPU_POS_C4		8

#endif
