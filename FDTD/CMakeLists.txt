# HIP GPU backend
if(ENABLE_HIP)
  set(HIP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/gpu_backend_hip.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_ext_excitation.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_ext_upml.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_ext_mur_abc.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_ext_lorentzmaterial.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_ext_debyematerial.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_ext_lumpedRLC.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_ext_tfsf.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_ext_absorbing_bc.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_ext_steadystate.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_ext_cylinder.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_multigrid.hip
  )
  # the top level needs the list by itself: source file properties belong to the
  # directory of the target, and the openEMS and plugin targets are defined there
  set(HIP_SOURCES ${HIP_SOURCES} PARENT_SCOPE)
  if(OPENEMS_GPU_PLUGINS)
    set(HIP_PLUGIN_SOURCES ${HIP_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_plugin.cpp PARENT_SCOPE)
    # in a plugin, not in libopenEMS
    set(HIP_SOURCES "")
  endif()
endif()

# SYCL GPU backend (Intel GPUs), only built as a plugin (see the top level)
if(ENABLE_SYCL)
  set(SYCL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/gpu_backend_sycl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_ext_excitation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_ext_upml.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_ext_mur_abc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_ext_lorentzmaterial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_ext_debyematerial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_ext_lumpedRLC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_ext_tfsf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_ext_absorbing_bc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_ext_steadystate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_ext_cylinder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_multigrid.cpp
  )
  set(SYCL_SOURCES ${SYCL_SOURCES} PARENT_SCOPE)
  if(OPENEMS_GPU_PLUGINS)
    set(SYCL_PLUGIN_SOURCES ${SYCL_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/sycl/sycl_plugin.cpp PARENT_SCOPE)
    # in a plugin, not in libopenEMS
    set(SYCL_SOURCES "")
  endif()
endif()

# Metal GPU backend (Objective-C++)
if(APPLE AND ENABLE_METAL)
  set(METAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/gpu_backend_metal.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_ext_excitation.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_ext_upml.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_ext_mur_abc.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_ext_lorentzmaterial.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_ext_debyematerial.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_ext_lumpedRLC.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_ext_tfsf.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_ext_absorbing_bc.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_ext_steadystate.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_ext_cylinder.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/metal_multigrid.mm
  )
endif()

set(SOURCES
  ${SOURCES}
  ${HIP_SOURCES}
  ${SYCL_SOURCES}
  ${METAL_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/engine.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_multithread.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_cylinder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_cylinder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_sse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_sse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_sse_compressed.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_sse_compressed.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_multithread.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/excitation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_cylindermultigrid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_cylindermultigrid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_interface_fdtd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_interface_sse_fdtd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_interface_gpu_fdtd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_field_gather.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_gpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_gpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/engine_gpu_cylindermultigrid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_backend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_plugins.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_coeff_sets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_backend_reference.cpp
  PARENT_SCOPE
)
