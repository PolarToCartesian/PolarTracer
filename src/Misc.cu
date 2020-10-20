#ifndef __PRTX__MISC_CU
#define __PRTX__MISC_CU

#include <cstdint>

namespace PRTX {

    template <typename _T>
    __host__ __device__ inline _T Clamp(const _T& x, const _T& min, const _T& max) noexcept {
        return (x > max) ? max : ((x < min) ? min : x);
    }

}; // namespace PRTX

#endif // __PRTX__MISC_CU