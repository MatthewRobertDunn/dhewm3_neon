#include "sys/platform.h"

#include "idlib/math/Simd_NEON.h"

/*
===============================================================================

	Arm NEON implementation of idSIMDProcessor

===============================================================================
*/

#if defined(__GNUC__) && defined(__ARM_NEON)

#include <arm_neon.h>

/*
============
idSIMD_NEON::GetName
============
*/
const char *idSIMD_NEON::GetName(void) const
{
	return "Arm Advanced SIMD";
}

/*
============
idSIMD_NEON::Add

  dst[i] = constant + src[i];
============
*/
void VPCALL idSIMD_NEON::Add(float *dst, const float constant, const float *src, const int count)
{
	float32x4_t v_constant = vdupq_n_f32(constant);
	int i;

	// Add up four lanes at a time
	for (i = 0; i < count; i += 4)
	{
		float32x4_t v_src = vld1q_f32(&src[i]);
		float32x4_t v_dst = vaddq_f32(v_constant, v_src);
		vst1q_f32(&dst[i], v_dst);
	}

	// add anything left over.
	for (; i < count; i++)
	{
		dst[i] = src[i] + constant;
	}
}

/*
============
idSIMD_NEON::Add

  dst[i] = src0[i] + src1[i];
============
*/
void VPCALL idSIMD_NEON::Add(float *dst, const float *src0, const float *src1, const int count)
{
	int i;
	// Add up four lanes at a time
	for (i = 0; i < count; i += 4)
	{
		float32x4_t v_src0 = vld1q_f32(&src0[i]);
		float32x4_t v_src1 = vld1q_f32(&src1[i]);
		float32x4_t v_dst = vaddq_f32(v_src0, v_src1);
		vst1q_f32(&dst[i], v_dst);
	}

	// add anything left over.
	for (; i < count; i++)
	{
		dst[i] = src0[i] + src1[i];
	}
}

/*
============
idSIMD_NEON::Sub

  dst[i] = constant - src[i];
============
*/
void VPCALL idSIMD_NEON::Sub( float *dst, const float constant, const float *src, const int count ) {
	float32x4_t v_constant = vdupq_n_f32(constant);
	int i;

	// Add up four lanes at a time
	for (i = 0; i < count; i += 4)
	{
		float32x4_t v_src = vld1q_f32(&src[i]);
		float32x4_t v_dst = vsubq_f32(v_constant, v_src);
		vst1q_f32(&dst[i], v_dst);
	}

	// add anything left over.
	for (; i < count; i++)
	{
		dst[i] = constant - src[i] ;
	}
}

/*
============
idSIMD_NEON::Sub

  dst[i] = src0[i] - src1[i];
============
*/
void VPCALL idSIMD_NEON::Sub(float *dst, const float *src0, const float *src1, const int count)
{
	int i;
	// Add up four lanes at a time
	for (i = 0; i < count; i += 4)
	{
		float32x4_t v_src0 = vld1q_f32(&src0[i]);
		float32x4_t v_src1 = vld1q_f32(&src1[i]);
		float32x4_t v_dst = vsubq_f32(v_src0, v_src1);
		vst1q_f32(&dst[i], v_dst);
	}

	// add anything left over.
	for (; i < count; i++)
	{
		dst[i] = src0[i] - src1[i];
	}
}

/*
============
idSIMD_NEON::Mul

  dst[i] = constant * src[i];
============
*/
void VPCALL idSIMD_NEON::Mul( float *dst, const float constant, const float *src, const int count) {
	float32x4_t v_constant = vdupq_n_f32(constant);
	int i;

	// Add up four lanes at a time
	for (i = 0; i < count; i += 4)
	{
		float32x4_t v_src = vld1q_f32(&src[i]);
		float32x4_t v_dst = vmulq_f32(v_constant, v_src);
		vst1q_f32(&dst[i], v_dst);
	}

	// add anything left over.
	for (; i < count; i++)
	{
		dst[i] = constant * src[i] ;
	}
}


/*
============
idSIMD_NEON::Mul

  dst[i] = src0[i] * src1[i];
============
*/
void VPCALL idSIMD_NEON::Mul( float *dst, const float *src0, const float *src1, const int count ) {
	int i;
	// Add up four lanes at a time
	for (i = 0; i < count; i += 4)
	{
		float32x4_t v_src0 = vld1q_f32(&src0[i]);
		float32x4_t v_src1 = vld1q_f32(&src1[i]);
		float32x4_t v_dst = vmulq_f32(v_src0, v_src1);
		vst1q_f32(&dst[i], v_dst);
	}

	// add anything left over.
	for (; i < count; i++)
	{
		dst[i] = src0[i] * src1[i];
	}
}

#endif