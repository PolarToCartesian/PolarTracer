#ifndef __POLAR_TRACER__ENUMS_CU
#define __POLAR_TRACER__ENUMS_CU

// The "Device" enum class represents the devices from which
// memory can be accessed. This is necessary because the cpu can't
// read/write directly from/to the GPU's memory and conversely.
enum class Device { CPU, GPU }; // Device

#endif // __POLAR_TRACER__ENUMS_CU