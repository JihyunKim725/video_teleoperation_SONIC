# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated with version 12.9.0. Do not modify it directly.
cimport cuda.bindings._lib.dlfcn as dlfcn
from cuda.pathfinder import load_nvidia_dynamic_lib
from libc.stdint cimport intptr_t, uintptr_t
import threading

cdef object __symbol_lock = threading.Lock()
cdef bint __cuPythonInit = False
cdef void *__nvrtcGetErrorString = NULL
cdef void *__nvrtcVersion = NULL
cdef void *__nvrtcGetNumSupportedArchs = NULL
cdef void *__nvrtcGetSupportedArchs = NULL
cdef void *__nvrtcCreateProgram = NULL
cdef void *__nvrtcDestroyProgram = NULL
cdef void *__nvrtcCompileProgram = NULL
cdef void *__nvrtcGetPTXSize = NULL
cdef void *__nvrtcGetPTX = NULL
cdef void *__nvrtcGetCUBINSize = NULL
cdef void *__nvrtcGetCUBIN = NULL
cdef void *__nvrtcGetNVVMSize = NULL
cdef void *__nvrtcGetNVVM = NULL
cdef void *__nvrtcGetLTOIRSize = NULL
cdef void *__nvrtcGetLTOIR = NULL
cdef void *__nvrtcGetOptiXIRSize = NULL
cdef void *__nvrtcGetOptiXIR = NULL
cdef void *__nvrtcGetProgramLogSize = NULL
cdef void *__nvrtcGetProgramLog = NULL
cdef void *__nvrtcAddNameExpression = NULL
cdef void *__nvrtcGetLoweredName = NULL
cdef void *__nvrtcGetPCHHeapSize = NULL
cdef void *__nvrtcSetPCHHeapSize = NULL
cdef void *__nvrtcGetPCHCreateStatus = NULL
cdef void *__nvrtcGetPCHHeapSizeRequired = NULL
cdef void *__nvrtcSetFlowCallback = NULL

cdef int _cuPythonInit() except -1 nogil:
    global __cuPythonInit

    # Load library
    with gil, __symbol_lock:
        handle = <void*><uintptr_t>(load_nvidia_dynamic_lib("nvrtc")._handle_uint)

        # Load function
        global __nvrtcGetErrorString
        __nvrtcGetErrorString = dlfcn.dlsym(handle, 'nvrtcGetErrorString')
        global __nvrtcVersion
        __nvrtcVersion = dlfcn.dlsym(handle, 'nvrtcVersion')
        global __nvrtcGetNumSupportedArchs
        __nvrtcGetNumSupportedArchs = dlfcn.dlsym(handle, 'nvrtcGetNumSupportedArchs')
        global __nvrtcGetSupportedArchs
        __nvrtcGetSupportedArchs = dlfcn.dlsym(handle, 'nvrtcGetSupportedArchs')
        global __nvrtcCreateProgram
        __nvrtcCreateProgram = dlfcn.dlsym(handle, 'nvrtcCreateProgram')
        global __nvrtcDestroyProgram
        __nvrtcDestroyProgram = dlfcn.dlsym(handle, 'nvrtcDestroyProgram')
        global __nvrtcCompileProgram
        __nvrtcCompileProgram = dlfcn.dlsym(handle, 'nvrtcCompileProgram')
        global __nvrtcGetPTXSize
        __nvrtcGetPTXSize = dlfcn.dlsym(handle, 'nvrtcGetPTXSize')
        global __nvrtcGetPTX
        __nvrtcGetPTX = dlfcn.dlsym(handle, 'nvrtcGetPTX')
        global __nvrtcGetCUBINSize
        __nvrtcGetCUBINSize = dlfcn.dlsym(handle, 'nvrtcGetCUBINSize')
        global __nvrtcGetCUBIN
        __nvrtcGetCUBIN = dlfcn.dlsym(handle, 'nvrtcGetCUBIN')
        global __nvrtcGetNVVMSize
        __nvrtcGetNVVMSize = dlfcn.dlsym(handle, 'nvrtcGetNVVMSize')
        global __nvrtcGetNVVM
        __nvrtcGetNVVM = dlfcn.dlsym(handle, 'nvrtcGetNVVM')
        global __nvrtcGetLTOIRSize
        __nvrtcGetLTOIRSize = dlfcn.dlsym(handle, 'nvrtcGetLTOIRSize')
        global __nvrtcGetLTOIR
        __nvrtcGetLTOIR = dlfcn.dlsym(handle, 'nvrtcGetLTOIR')
        global __nvrtcGetOptiXIRSize
        __nvrtcGetOptiXIRSize = dlfcn.dlsym(handle, 'nvrtcGetOptiXIRSize')
        global __nvrtcGetOptiXIR
        __nvrtcGetOptiXIR = dlfcn.dlsym(handle, 'nvrtcGetOptiXIR')
        global __nvrtcGetProgramLogSize
        __nvrtcGetProgramLogSize = dlfcn.dlsym(handle, 'nvrtcGetProgramLogSize')
        global __nvrtcGetProgramLog
        __nvrtcGetProgramLog = dlfcn.dlsym(handle, 'nvrtcGetProgramLog')
        global __nvrtcAddNameExpression
        __nvrtcAddNameExpression = dlfcn.dlsym(handle, 'nvrtcAddNameExpression')
        global __nvrtcGetLoweredName
        __nvrtcGetLoweredName = dlfcn.dlsym(handle, 'nvrtcGetLoweredName')
        global __nvrtcGetPCHHeapSize
        __nvrtcGetPCHHeapSize = dlfcn.dlsym(handle, 'nvrtcGetPCHHeapSize')
        global __nvrtcSetPCHHeapSize
        __nvrtcSetPCHHeapSize = dlfcn.dlsym(handle, 'nvrtcSetPCHHeapSize')
        global __nvrtcGetPCHCreateStatus
        __nvrtcGetPCHCreateStatus = dlfcn.dlsym(handle, 'nvrtcGetPCHCreateStatus')
        global __nvrtcGetPCHHeapSizeRequired
        __nvrtcGetPCHHeapSizeRequired = dlfcn.dlsym(handle, 'nvrtcGetPCHHeapSizeRequired')
        global __nvrtcSetFlowCallback
        __nvrtcSetFlowCallback = dlfcn.dlsym(handle, 'nvrtcSetFlowCallback')
        __cuPythonInit = True
        return 0

# Create a very small function to check whether we are init'ed, so the C
# compiler can inline it.
cdef inline int cuPythonInit() except -1 nogil:
    if __cuPythonInit:
        return 0
    return _cuPythonInit()

cdef const char* _nvrtcGetErrorString(nvrtcResult result) except ?NULL nogil:
    global __nvrtcGetErrorString
    cuPythonInit()
    if __nvrtcGetErrorString == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetErrorString" not found')
    err = (<const char* (*)(nvrtcResult) except ?NULL nogil> __nvrtcGetErrorString)(result)
    return err

cdef nvrtcResult _nvrtcVersion(int* major, int* minor) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcVersion
    cuPythonInit()
    if __nvrtcVersion == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcVersion" not found')
    err = (<nvrtcResult (*)(int*, int*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcVersion)(major, minor)
    return err

cdef nvrtcResult _nvrtcGetNumSupportedArchs(int* numArchs) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetNumSupportedArchs
    cuPythonInit()
    if __nvrtcGetNumSupportedArchs == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetNumSupportedArchs" not found')
    err = (<nvrtcResult (*)(int*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetNumSupportedArchs)(numArchs)
    return err

cdef nvrtcResult _nvrtcGetSupportedArchs(int* supportedArchs) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetSupportedArchs
    cuPythonInit()
    if __nvrtcGetSupportedArchs == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetSupportedArchs" not found')
    err = (<nvrtcResult (*)(int*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetSupportedArchs)(supportedArchs)
    return err

cdef nvrtcResult _nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcCreateProgram
    cuPythonInit()
    if __nvrtcCreateProgram == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcCreateProgram" not found')
    err = (<nvrtcResult (*)(nvrtcProgram*, const char*, const char*, int, const char**, const char**) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcCreateProgram)(prog, src, name, numHeaders, headers, includeNames)
    return err

cdef nvrtcResult _nvrtcDestroyProgram(nvrtcProgram* prog) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcDestroyProgram
    cuPythonInit()
    if __nvrtcDestroyProgram == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcDestroyProgram" not found')
    err = (<nvrtcResult (*)(nvrtcProgram*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcDestroyProgram)(prog)
    return err

cdef nvrtcResult _nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcCompileProgram
    cuPythonInit()
    if __nvrtcCompileProgram == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcCompileProgram" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, int, const char**) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcCompileProgram)(prog, numOptions, options)
    return err

cdef nvrtcResult _nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPTXSize
    cuPythonInit()
    if __nvrtcGetPTXSize == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetPTXSize" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, size_t*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetPTXSize)(prog, ptxSizeRet)
    return err

cdef nvrtcResult _nvrtcGetPTX(nvrtcProgram prog, char* ptx) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPTX
    cuPythonInit()
    if __nvrtcGetPTX == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetPTX" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, char*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetPTX)(prog, ptx)
    return err

cdef nvrtcResult _nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetCUBINSize
    cuPythonInit()
    if __nvrtcGetCUBINSize == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetCUBINSize" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, size_t*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetCUBINSize)(prog, cubinSizeRet)
    return err

cdef nvrtcResult _nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetCUBIN
    cuPythonInit()
    if __nvrtcGetCUBIN == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetCUBIN" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, char*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetCUBIN)(prog, cubin)
    return err

cdef nvrtcResult _nvrtcGetNVVMSize(nvrtcProgram prog, size_t* nvvmSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetNVVMSize
    cuPythonInit()
    if __nvrtcGetNVVMSize == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetNVVMSize" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, size_t*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetNVVMSize)(prog, nvvmSizeRet)
    return err

cdef nvrtcResult _nvrtcGetNVVM(nvrtcProgram prog, char* nvvm) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetNVVM
    cuPythonInit()
    if __nvrtcGetNVVM == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetNVVM" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, char*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetNVVM)(prog, nvvm)
    return err

cdef nvrtcResult _nvrtcGetLTOIRSize(nvrtcProgram prog, size_t* LTOIRSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetLTOIRSize
    cuPythonInit()
    if __nvrtcGetLTOIRSize == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetLTOIRSize" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, size_t*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetLTOIRSize)(prog, LTOIRSizeRet)
    return err

cdef nvrtcResult _nvrtcGetLTOIR(nvrtcProgram prog, char* LTOIR) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetLTOIR
    cuPythonInit()
    if __nvrtcGetLTOIR == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetLTOIR" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, char*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetLTOIR)(prog, LTOIR)
    return err

cdef nvrtcResult _nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t* optixirSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetOptiXIRSize
    cuPythonInit()
    if __nvrtcGetOptiXIRSize == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetOptiXIRSize" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, size_t*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetOptiXIRSize)(prog, optixirSizeRet)
    return err

cdef nvrtcResult _nvrtcGetOptiXIR(nvrtcProgram prog, char* optixir) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetOptiXIR
    cuPythonInit()
    if __nvrtcGetOptiXIR == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetOptiXIR" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, char*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetOptiXIR)(prog, optixir)
    return err

cdef nvrtcResult _nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetProgramLogSize
    cuPythonInit()
    if __nvrtcGetProgramLogSize == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetProgramLogSize" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, size_t*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetProgramLogSize)(prog, logSizeRet)
    return err

cdef nvrtcResult _nvrtcGetProgramLog(nvrtcProgram prog, char* log) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetProgramLog
    cuPythonInit()
    if __nvrtcGetProgramLog == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetProgramLog" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, char*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetProgramLog)(prog, log)
    return err

cdef nvrtcResult _nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcAddNameExpression
    cuPythonInit()
    if __nvrtcAddNameExpression == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcAddNameExpression" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, const char*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcAddNameExpression)(prog, name_expression)
    return err

cdef nvrtcResult _nvrtcGetLoweredName(nvrtcProgram prog, const char* name_expression, const char** lowered_name) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetLoweredName
    cuPythonInit()
    if __nvrtcGetLoweredName == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetLoweredName" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, const char*, const char**) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetLoweredName)(prog, name_expression, lowered_name)
    return err

cdef nvrtcResult _nvrtcGetPCHHeapSize(size_t* ret) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPCHHeapSize
    cuPythonInit()
    if __nvrtcGetPCHHeapSize == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetPCHHeapSize" not found')
    err = (<nvrtcResult (*)(size_t*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetPCHHeapSize)(ret)
    return err

cdef nvrtcResult _nvrtcSetPCHHeapSize(size_t size) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcSetPCHHeapSize
    cuPythonInit()
    if __nvrtcSetPCHHeapSize == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcSetPCHHeapSize" not found')
    err = (<nvrtcResult (*)(size_t) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcSetPCHHeapSize)(size)
    return err

cdef nvrtcResult _nvrtcGetPCHCreateStatus(nvrtcProgram prog) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPCHCreateStatus
    cuPythonInit()
    if __nvrtcGetPCHCreateStatus == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetPCHCreateStatus" not found')
    err = (<nvrtcResult (*)(nvrtcProgram) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetPCHCreateStatus)(prog)
    return err

cdef nvrtcResult _nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, size_t* size) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcGetPCHHeapSizeRequired
    cuPythonInit()
    if __nvrtcGetPCHHeapSizeRequired == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcGetPCHHeapSizeRequired" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, size_t*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcGetPCHHeapSizeRequired)(prog, size)
    return err

cdef nvrtcResult _nvrtcSetFlowCallback(nvrtcProgram prog, void* callback, void* payload) except ?NVRTC_ERROR_INVALID_INPUT nogil:
    global __nvrtcSetFlowCallback
    cuPythonInit()
    if __nvrtcSetFlowCallback == NULL:
        with gil:
            raise RuntimeError('Function "nvrtcSetFlowCallback" not found')
    err = (<nvrtcResult (*)(nvrtcProgram, void*, void*) except ?NVRTC_ERROR_INVALID_INPUT nogil> __nvrtcSetFlowCallback)(prog, callback, payload)
    return err

cdef dict func_ptrs = None

cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    cuPythonInit()
    cdef dict data = {}

    global __nvrtcGetErrorString
    data["__nvrtcGetErrorString"] = <intptr_t>__nvrtcGetErrorString
    global __nvrtcVersion
    data["__nvrtcVersion"] = <intptr_t>__nvrtcVersion
    global __nvrtcGetNumSupportedArchs
    data["__nvrtcGetNumSupportedArchs"] = <intptr_t>__nvrtcGetNumSupportedArchs
    global __nvrtcGetSupportedArchs
    data["__nvrtcGetSupportedArchs"] = <intptr_t>__nvrtcGetSupportedArchs
    global __nvrtcCreateProgram
    data["__nvrtcCreateProgram"] = <intptr_t>__nvrtcCreateProgram
    global __nvrtcDestroyProgram
    data["__nvrtcDestroyProgram"] = <intptr_t>__nvrtcDestroyProgram
    global __nvrtcCompileProgram
    data["__nvrtcCompileProgram"] = <intptr_t>__nvrtcCompileProgram
    global __nvrtcGetPTXSize
    data["__nvrtcGetPTXSize"] = <intptr_t>__nvrtcGetPTXSize
    global __nvrtcGetPTX
    data["__nvrtcGetPTX"] = <intptr_t>__nvrtcGetPTX
    global __nvrtcGetCUBINSize
    data["__nvrtcGetCUBINSize"] = <intptr_t>__nvrtcGetCUBINSize
    global __nvrtcGetCUBIN
    data["__nvrtcGetCUBIN"] = <intptr_t>__nvrtcGetCUBIN
    global __nvrtcGetNVVMSize
    data["__nvrtcGetNVVMSize"] = <intptr_t>__nvrtcGetNVVMSize
    global __nvrtcGetNVVM
    data["__nvrtcGetNVVM"] = <intptr_t>__nvrtcGetNVVM
    global __nvrtcGetLTOIRSize
    data["__nvrtcGetLTOIRSize"] = <intptr_t>__nvrtcGetLTOIRSize
    global __nvrtcGetLTOIR
    data["__nvrtcGetLTOIR"] = <intptr_t>__nvrtcGetLTOIR
    global __nvrtcGetOptiXIRSize
    data["__nvrtcGetOptiXIRSize"] = <intptr_t>__nvrtcGetOptiXIRSize
    global __nvrtcGetOptiXIR
    data["__nvrtcGetOptiXIR"] = <intptr_t>__nvrtcGetOptiXIR
    global __nvrtcGetProgramLogSize
    data["__nvrtcGetProgramLogSize"] = <intptr_t>__nvrtcGetProgramLogSize
    global __nvrtcGetProgramLog
    data["__nvrtcGetProgramLog"] = <intptr_t>__nvrtcGetProgramLog
    global __nvrtcAddNameExpression
    data["__nvrtcAddNameExpression"] = <intptr_t>__nvrtcAddNameExpression
    global __nvrtcGetLoweredName
    data["__nvrtcGetLoweredName"] = <intptr_t>__nvrtcGetLoweredName
    global __nvrtcGetPCHHeapSize
    data["__nvrtcGetPCHHeapSize"] = <intptr_t>__nvrtcGetPCHHeapSize
    global __nvrtcSetPCHHeapSize
    data["__nvrtcSetPCHHeapSize"] = <intptr_t>__nvrtcSetPCHHeapSize
    global __nvrtcGetPCHCreateStatus
    data["__nvrtcGetPCHCreateStatus"] = <intptr_t>__nvrtcGetPCHCreateStatus
    global __nvrtcGetPCHHeapSizeRequired
    data["__nvrtcGetPCHHeapSizeRequired"] = <intptr_t>__nvrtcGetPCHHeapSizeRequired
    global __nvrtcSetFlowCallback
    data["__nvrtcSetFlowCallback"] = <intptr_t>__nvrtcSetFlowCallback

    func_ptrs = data
    return data

cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]
