#include "src/scheduler/scheduler.cuh"
#include "src/scheduler/common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace eps
{
    template <typename SchedulerTraits>
    Scheduler<SchedulerTraits>::Context::Context(Params p, RunParams _rp) : p_{p}, rp{_rp}, processors{p, _rp} {}

    template <typename SchedulerTraits>
    size_t Scheduler<SchedulerTraits>::Context::getWorkspaceSize()
    {
        size_t bytes = 0;
        workspace_offsets.processors = bytes;
        bytes += ALIGN(processors.getWorkspaceSize());

        return bytes;
    }

    template <typename SchedulerTraits>
    char *Scheduler<SchedulerTraits>::Context::processors_ws()
    {
        return ws + workspace_offsets.processors;
    }

    template <typename SchedulerTraits>
    void Scheduler<SchedulerTraits>::Context::set_ws(char* ws_ptr) {
      ws = ws_ptr;
      processors.set_ws(processors_ws());
    }

    template
    class Scheduler<SchedulerTraits<float, float, float>>::Context;

    template
    class Scheduler<SchedulerTraits<half, half, half>>::Context;
    template
    class Scheduler<SchedulerTraits<half, half, float>>::Context;

    #ifdef ENABLE_BF16
    template
    class Scheduler<SchedulerTraits<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>::Context;
    template
    class Scheduler<SchedulerTraits<__nv_bfloat16, __nv_bfloat16, float>>::Context;
    #endif

    template
    class Scheduler<SchedulerTraits<__nv_fp8_e4m3, float, float>>::Context;

    template
    class Scheduler<SchedulerTraits<__nv_fp8_e4m3, half, half>>::Context;
    template
    class Scheduler<SchedulerTraits<__nv_fp8_e4m3, half, float>>::Context;

    template
    class Scheduler<SchedulerTraits<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16>>::Context;
    template
    class Scheduler<SchedulerTraits<__nv_fp8_e4m3, __nv_bfloat16, float>>::Context;
}