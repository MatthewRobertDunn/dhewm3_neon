#if defined (__ARM_NEON__) && !defined(__ARM_NEON)
#define __ARM_NEON 1
#endif

#ifndef __MATH_SIMD_NEON_H__
#define __MATH_SIMD_NEON_H__

#include "idlib/math/Simd_Generic.h"

/*
===============================================================================

	Arm NEON implementation of idSIMDProcessor

===============================================================================
*/

class idSIMD_NEON : public idSIMD_Generic {
public:
    #if defined(__GNUC__) && defined(__ARM_NEON)
        virtual const char * VPCALL GetName( void ) const;
        virtual void VPCALL Add( float *dst,			const float constant,	const float *src,		const int count );
        virtual void VPCALL Add( float *dst,			const float *src0,		const float *src1,		const int count );
        virtual void VPCALL Sub( float *dst,			const float constant,	const float *src,		const int count );
        virtual void VPCALL Sub( float *dst,			const float *src0,		const float *src1,		const int count );
        virtual void VPCALL Mul( float *dst,			const float constant,	const float *src,		const int count );
	    virtual void VPCALL Mul( float *dst,			const float *src0,		const float *src1,		const int count );
    #endif

};

#endif /* !__MATH_SIMD_MMX_H__ */