#include "idlib/math/Simd_NEON.h"

/*
===============================================================================

	Arm NEON implementation of idSIMDProcessor

===============================================================================
*/

#if defined(__GNUC__) && defined(__ARM_NEON__)
/*
============
idSIMD_NEON::GetName
============
*/
const char * idSIMD_NEON::GetName( void ) const {
	return "Arm Advanced SIMD";
}
#endif