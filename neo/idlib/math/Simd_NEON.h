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
    #if defined(__GNUC__) && defined(__ARM_NEON__)
        virtual const char * VPCALL GetName( void ) const;
    #endif

};

#endif /* !__MATH_SIMD_MMX_H__ */