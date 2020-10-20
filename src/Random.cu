#ifndef __PRTX__RANDOM_CU
#define __PRTX__RANDOM_CU

namespace PRTX {

    __device__ size_t randomNumberIndex = 0u;
    __device__ float  randomNumbers[100] = { 0.199597f, 0.604987f, 0.255558f, 0.421514f, 0.720092f, 0.815522f, 0.192279f, 0.385067f, 0.350586f, 0.397595f, 0.357564f, 0.748578f, 0.00414681f, 0.533777f, 0.995393f, 0.907929f, 0.494525f, 0.472084f, 0.864498f, 0.695326f, 0.938409f, 0.785484f, 0.290453f, 0.13312f, 0.943201f, 0.926033f, 0.320409f, 0.0662487f, 0.25414f, 0.421945f, 0.667499f, 0.444524f, 0.838885f, 0.908202f, 0.8063f, 0.291879f, 0.114376f, 0.875398f, 0.247916f, 0.045868f, 0.535327f, 0.491882f, 0.642606f, 0.184197f, 0.154249f, 0.14628f, 0.939923f, 0.979867f, 0.503506f, 0.478285f, 0.491597f, 0.0545161f, 0.847528f, 0.0108021f, 0.934526f, 0.282655f, 0.0207591f, 0.329495f, 0.328761f, 0.560112f, 0.119835f, 0.296947f, 0.289384f, 0.83466f, 0.164883f, 0.0987901f, 0.0792031f, 0.258547f, 0.0754077f, 0.0143626f, 0.318207f, 0.483693f, 0.0715536f, 0.998425f, 0.322974f, 0.879418f, 0.261024f, 0.49866f, 0.453179f, 0.347203f, 0.638452f, 0.274543f, 0.595394f, 0.640481f, 0.798533f, 0.680735f, 0.95186f, 0.4518f, 0.969803f, 0.419822f, 0.00485671f, 0.727772f, 0.475605f, 0.816288f, 0.55194f, 0.550753f, 0.601672f, 0.908048f, 0.35448f, 0.863961f };

    __device__ float RandomFloat() noexcept {
        return randomNumbers[(randomNumberIndex++) % (sizeof(randomNumbers) / sizeof(float))];
    }

}; // namespace PRTX

#endif // __PRTX__RANDOM_CU