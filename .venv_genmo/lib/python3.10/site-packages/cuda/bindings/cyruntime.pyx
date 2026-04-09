# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated with version 12.9.0. Do not modify it directly.
cimport cuda.bindings._bindings.cyruntime as cyruntime
cimport cython

cdef cudaError_t cudaDeviceReset() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceReset()

cdef cudaError_t cudaDeviceSynchronize() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceSynchronize()

cdef cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceSetLimit(limit, value)

cdef cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetLimit(pValue, limit)

cdef cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device)

cdef cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetCacheConfig(pCacheConfig)

cdef cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority)

cdef cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceSetCacheConfig(cacheConfig)

cdef cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetByPCIBusId(device, pciBusId)

cdef cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int length, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetPCIBusId(pciBusId, length, device)

cdef cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaIpcGetEventHandle(handle, event)

cdef cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaIpcOpenEventHandle(event, handle)

cdef cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaIpcGetMemHandle(handle, devPtr)

cdef cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaIpcOpenMemHandle(devPtr, handle, flags)

cdef cudaError_t cudaIpcCloseMemHandle(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaIpcCloseMemHandle(devPtr)

cdef cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceFlushGPUDirectRDMAWrites(target, scope)

cdef cudaError_t cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceRegisterAsyncNotification(device, callbackFunc, userData, callback)

cdef cudaError_t cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceUnregisterAsyncNotification(device, callback)

cdef cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetSharedMemConfig(pConfig)

cdef cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceSetSharedMemConfig(config)

cdef cudaError_t cudaGetLastError() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetLastError()

cdef cudaError_t cudaPeekAtLastError() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaPeekAtLastError()

cdef const char* cudaGetErrorName(cudaError_t error) except ?NULL nogil:
    return cyruntime._cudaGetErrorName(error)

cdef const char* cudaGetErrorString(cudaError_t error) except ?NULL nogil:
    return cyruntime._cudaGetErrorString(error)

cdef cudaError_t cudaGetDeviceCount(int* count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetDeviceCount(count)

cdef cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetDeviceProperties_v2(prop, device)

cdef cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetAttribute(value, attr, device)

cdef cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetDefaultMemPool(memPool, device)

cdef cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceSetMemPool(device, memPool)

cdef cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetMemPool(memPool, device)

cdef cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags)

cdef cudaError_t cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice)

cdef cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaChooseDevice(device, prop)

cdef cudaError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaInitDevice(device, deviceFlags, flags)

cdef cudaError_t cudaSetDevice(int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaSetDevice(device)

cdef cudaError_t cudaGetDevice(int* device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetDevice(device)

cdef cudaError_t cudaSetDeviceFlags(unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaSetDeviceFlags(flags)

cdef cudaError_t cudaGetDeviceFlags(unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetDeviceFlags(flags)

cdef cudaError_t cudaStreamCreate(cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamCreate(pStream)

cdef cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamCreateWithFlags(pStream, flags)

cdef cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamCreateWithPriority(pStream, flags, priority)

cdef cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamGetPriority(hStream, priority)

cdef cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamGetFlags(hStream, flags)

cdef cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamGetId(hStream, streamId)

cdef cudaError_t cudaStreamGetDevice(cudaStream_t hStream, int* device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamGetDevice(hStream, device)

cdef cudaError_t cudaCtxResetPersistingL2Cache() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaCtxResetPersistingL2Cache()

cdef cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamCopyAttributes(dst, src)

cdef cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamGetAttribute(hStream, attr, value_out)

cdef cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamSetAttribute(hStream, attr, value)

cdef cudaError_t cudaStreamDestroy(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamDestroy(stream)

cdef cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamWaitEvent(stream, event, flags)

cdef cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamAddCallback(stream, callback, userData, flags)

cdef cudaError_t cudaStreamSynchronize(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamSynchronize(stream)

cdef cudaError_t cudaStreamQuery(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamQuery(stream)

cdef cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamAttachMemAsync(stream, devPtr, length, flags)

cdef cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamBeginCapture(stream, mode)

cdef cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamBeginCaptureToGraph(stream, graph, dependencies, dependencyData, numDependencies, mode)

cdef cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaThreadExchangeStreamCaptureMode(mode)

cdef cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamEndCapture(stream, pGraph)

cdef cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamIsCapturing(stream, pCaptureStatus)

cdef cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)

cdef cudaError_t cudaStreamGetCaptureInfo_v3(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, const cudaGraphEdgeData** edgeData_out, size_t* numDependencies_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamGetCaptureInfo_v3(stream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)

cdef cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags)

cdef cudaError_t cudaStreamUpdateCaptureDependencies_v2(cudaStream_t stream, cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaStreamUpdateCaptureDependencies_v2(stream, dependencies, dependencyData, numDependencies, flags)

cdef cudaError_t cudaEventCreate(cudaEvent_t* event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEventCreate(event)

cdef cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEventCreateWithFlags(event, flags)

cdef cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEventRecord(event, stream)

cdef cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEventRecordWithFlags(event, stream, flags)

cdef cudaError_t cudaEventQuery(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEventQuery(event)

cdef cudaError_t cudaEventSynchronize(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEventSynchronize(event)

cdef cudaError_t cudaEventDestroy(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEventDestroy(event)

cdef cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEventElapsedTime(ms, start, end)

cdef cudaError_t cudaEventElapsedTime_v2(float* ms, cudaEvent_t start, cudaEvent_t end) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEventElapsedTime_v2(ms, start, end)

cdef cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaImportExternalMemory(extMem_out, memHandleDesc)

cdef cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)

cdef cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)

cdef cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDestroyExternalMemory(extMem)

cdef cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaImportExternalSemaphore(extSem_out, semHandleDesc)

cdef cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaSignalExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream)

cdef cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaWaitExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream)

cdef cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDestroyExternalSemaphore(extSem)

cdef cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaFuncSetCacheConfig(func, cacheConfig)

cdef cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaFuncGetAttributes(attr, func)

cdef cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaFuncSetAttribute(func, attr, value)

cdef cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaLaunchHostFunc(stream, fn, userData)

cdef cudaError_t cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaFuncSetSharedMemConfig(func, config)

cdef cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)

cdef cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)

cdef cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags)

cdef cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMallocManaged(devPtr, size, flags)

cdef cudaError_t cudaMalloc(void** devPtr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMalloc(devPtr, size)

cdef cudaError_t cudaMallocHost(void** ptr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMallocHost(ptr, size)

cdef cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMallocPitch(devPtr, pitch, width, height)

cdef cudaError_t cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMallocArray(array, desc, width, height, flags)

cdef cudaError_t cudaFree(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaFree(devPtr)

cdef cudaError_t cudaFreeHost(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaFreeHost(ptr)

cdef cudaError_t cudaFreeArray(cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaFreeArray(array)

cdef cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaFreeMipmappedArray(mipmappedArray)

cdef cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaHostAlloc(pHost, size, flags)

cdef cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaHostRegister(ptr, size, flags)

cdef cudaError_t cudaHostUnregister(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaHostUnregister(ptr)

cdef cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaHostGetDevicePointer(pDevice, pHost, flags)

cdef cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaHostGetFlags(pFlags, pHost)

cdef cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMalloc3D(pitchedDevPtr, extent)

cdef cudaError_t cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMalloc3DArray(array, desc, extent, flags)

cdef cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags)

cdef cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level)

cdef cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy3D(p)

cdef cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy3DPeer(p)

cdef cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy3DAsync(p, stream)

cdef cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy3DPeerAsync(p, stream)

cdef cudaError_t cudaMemGetInfo(size_t* free, size_t* total) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemGetInfo(free, total)

cdef cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaArrayGetInfo(desc, extent, flags, array)

cdef cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaArrayGetPlane(pPlaneArray, hArray, planeIdx)

cdef cudaError_t cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaArrayGetMemoryRequirements(memoryRequirements, array, device)

cdef cudaError_t cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)

cdef cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaArrayGetSparseProperties(sparseProperties, array)

cdef cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap)

cdef cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy(dst, src, count, kind)

cdef cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count)

cdef cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)

cdef cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind)

cdef cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind)

cdef cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)

cdef cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpyAsync(dst, src, count, kind, stream)

cdef cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)

cdef cudaError_t cudaMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, size_t* failIdx, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, stream)

cdef cudaError_t cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp* opList, size_t* failIdx, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy3DBatchAsync(numOps, opList, failIdx, flags, stream)

cdef cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)

cdef cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)

cdef cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)

cdef cudaError_t cudaMemset(void* devPtr, int value, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemset(devPtr, value, count)

cdef cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemset2D(devPtr, pitch, value, width, height)

cdef cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemset3D(pitchedDevPtr, value, extent)

cdef cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemsetAsync(devPtr, value, count, stream)

cdef cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemset2DAsync(devPtr, pitch, value, width, height, stream)

cdef cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemset3DAsync(pitchedDevPtr, value, extent, stream)

cdef cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPrefetchAsync(devPtr, count, dstDevice, stream)

cdef cudaError_t cudaMemPrefetchAsync_v2(const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPrefetchAsync_v2(devPtr, count, location, flags, stream)

cdef cudaError_t cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemAdvise(devPtr, count, advice, device)

cdef cudaError_t cudaMemAdvise_v2(const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemAdvise_v2(devPtr, count, advice, location)

cdef cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)

cdef cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)

cdef cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind)

cdef cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind)

cdef cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind)

cdef cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream)

cdef cudaError_t cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream)

cdef cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMallocAsync(devPtr, size, hStream)

cdef cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaFreeAsync(devPtr, hStream)

cdef cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPoolTrimTo(memPool, minBytesToKeep)

cdef cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPoolSetAttribute(memPool, attr, value)

cdef cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPoolGetAttribute(memPool, attr, value)

cdef cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPoolSetAccess(memPool, descList, count)

cdef cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPoolGetAccess(flags, memPool, location)

cdef cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPoolCreate(memPool, poolProps)

cdef cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPoolDestroy(memPool)

cdef cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMallocFromPoolAsync(ptr, size, memPool, stream)

cdef cudaError_t cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags)

cdef cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags)

cdef cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPoolExportPointer(exportData, ptr)

cdef cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaMemPoolImportPointer(ptr, memPool, exportData)

cdef cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaPointerGetAttributes(attributes, ptr)

cdef cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)

cdef cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceEnablePeerAccess(peerDevice, flags)

cdef cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceDisablePeerAccess(peerDevice)

cdef cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsUnregisterResource(resource)

cdef cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsResourceSetMapFlags(resource, flags)

cdef cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsMapResources(count, resources, stream)

cdef cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsUnmapResources(count, resources, stream)

cdef cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsResourceGetMappedPointer(devPtr, size, resource)

cdef cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel)

cdef cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource)

cdef cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetChannelDesc(desc, array)
@cython.show_performance_hints(False)
cdef cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) except* nogil:
    return cyruntime._cudaCreateChannelDesc(x, y, z, w, f)

cdef cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)

cdef cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDestroyTextureObject(texObject)

cdef cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetTextureObjectResourceDesc(pResDesc, texObject)

cdef cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetTextureObjectTextureDesc(pTexDesc, texObject)

cdef cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject)

cdef cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaCreateSurfaceObject(pSurfObject, pResDesc)

cdef cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDestroySurfaceObject(surfObject)

cdef cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject)

cdef cudaError_t cudaDriverGetVersion(int* driverVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDriverGetVersion(driverVersion)

cdef cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaRuntimeGetVersion(runtimeVersion)

cdef cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphCreate(pGraph, flags)

cdef cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)

cdef cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphKernelNodeGetParams(node, pNodeParams)

cdef cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphKernelNodeSetParams(node, pNodeParams)

cdef cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphKernelNodeCopyAttributes(hSrc, hDst)

cdef cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphKernelNodeGetAttribute(hNode, attr, value_out)

cdef cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphKernelNodeSetAttribute(hNode, attr, value)

cdef cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams)

cdef cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind)

cdef cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphMemcpyNodeGetParams(node, pNodeParams)

cdef cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphMemcpyNodeSetParams(node, pNodeParams)

cdef cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)

cdef cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams)

cdef cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphMemsetNodeGetParams(node, pNodeParams)

cdef cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphMemsetNodeSetParams(node, pNodeParams)

cdef cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)

cdef cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphHostNodeGetParams(node, pNodeParams)

cdef cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphHostNodeSetParams(node, pNodeParams)

cdef cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph)

cdef cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphChildGraphNodeGetGraph(node, pGraph)

cdef cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies)

cdef cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event)

cdef cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphEventRecordNodeGetEvent(node, event_out)

cdef cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphEventRecordNodeSetEvent(node, event)

cdef cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event)

cdef cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphEventWaitNodeGetEvent(node, event_out)

cdef cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphEventWaitNodeSetEvent(node, event)

cdef cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)

cdef cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)

cdef cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)

cdef cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)

cdef cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphMemAllocNodeGetParams(node, params_out)

cdef cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr)

cdef cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphMemFreeNodeGetParams(node, dptr_out)

cdef cudaError_t cudaDeviceGraphMemTrim(int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGraphMemTrim(device)

cdef cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceGetGraphMemAttribute(device, attr, value)

cdef cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaDeviceSetGraphMemAttribute(device, attr, value)

cdef cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphClone(pGraphClone, originalGraph)

cdef cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph)

cdef cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphNodeGetType(node, pType)

cdef cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphGetNodes(graph, nodes, numNodes)

cdef cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes)

cdef cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, size_t* numEdges) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphGetEdges(graph, from_, to, numEdges)

cdef cudaError_t cudaGraphGetEdges_v2(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, cudaGraphEdgeData* edgeData, size_t* numEdges) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphGetEdges_v2(graph, from_, to, edgeData, numEdges)

cdef cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphNodeGetDependencies(node, pDependencies, pNumDependencies)

cdef cudaError_t cudaGraphNodeGetDependencies_v2(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, cudaGraphEdgeData* edgeData, size_t* pNumDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphNodeGetDependencies_v2(node, pDependencies, edgeData, pNumDependencies)

cdef cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes)

cdef cudaError_t cudaGraphNodeGetDependentNodes_v2(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, cudaGraphEdgeData* edgeData, size_t* pNumDependentNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphNodeGetDependentNodes_v2(node, pDependentNodes, edgeData, pNumDependentNodes)

cdef cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddDependencies(graph, from_, to, numDependencies)

cdef cudaError_t cudaGraphAddDependencies_v2(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddDependencies_v2(graph, from_, to, edgeData, numDependencies)

cdef cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphRemoveDependencies(graph, from_, to, numDependencies)

cdef cudaError_t cudaGraphRemoveDependencies_v2(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphRemoveDependencies_v2(graph, from_, to, edgeData, numDependencies)

cdef cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphDestroyNode(node)

cdef cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphInstantiate(pGraphExec, graph, flags)

cdef cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphInstantiateWithFlags(pGraphExec, graph, flags)

cdef cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphInstantiateWithParams(pGraphExec, graph, instantiateParams)

cdef cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecGetFlags(graphExec, flags)

cdef cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)

cdef cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph)

cdef cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)

cdef cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)

cdef cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)

cdef cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)

cdef cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)

cdef cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)

cdef cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecUpdate(hGraphExec, hGraph, resultInfo)

cdef cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphUpload(graphExec, stream)

cdef cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphLaunch(graphExec, stream)

cdef cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecDestroy(graphExec)

cdef cudaError_t cudaGraphDestroy(cudaGraph_t graph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphDestroy(graph)

cdef cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphDebugDotPrint(graph, path, flags)

cdef cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)

cdef cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaUserObjectRetain(object, count)

cdef cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaUserObjectRelease(object, count)

cdef cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphRetainUserObject(graph, object, count, flags)

cdef cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphReleaseUserObject(graph, object, count)

cdef cudaError_t cudaGraphAddNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t cudaGraphAddNode_v2(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphAddNode_v2(pGraphNode, graph, pDependencies, dependencyData, numDependencies, nodeParams)

cdef cudaError_t cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphNodeSetParams(node, nodeParams)

cdef cudaError_t cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphExecNodeSetParams(graphExec, node, nodeParams)

cdef cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, unsigned int defaultLaunchValue, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphConditionalHandleCreate(pHandle_out, graph, defaultLaunchValue, flags)

cdef cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus)

cdef cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetDriverEntryPointByVersion(symbol, funcPtr, cudaVersion, flags, driverStatus)

cdef cudaError_t cudaLibraryLoadData(cudaLibrary_t* library, const void* code, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)

cdef cudaError_t cudaLibraryLoadFromFile(cudaLibrary_t* library, const char* fileName, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)

cdef cudaError_t cudaLibraryUnload(cudaLibrary_t library) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaLibraryUnload(library)

cdef cudaError_t cudaLibraryGetKernel(cudaKernel_t* pKernel, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaLibraryGetKernel(pKernel, library, name)

cdef cudaError_t cudaLibraryGetGlobal(void** dptr, size_t* numbytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaLibraryGetGlobal(dptr, numbytes, library, name)

cdef cudaError_t cudaLibraryGetManaged(void** dptr, size_t* numbytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaLibraryGetManaged(dptr, numbytes, library, name)

cdef cudaError_t cudaLibraryGetUnifiedFunction(void** fptr, cudaLibrary_t library, const char* symbol) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaLibraryGetUnifiedFunction(fptr, library, symbol)

cdef cudaError_t cudaLibraryGetKernelCount(unsigned int* count, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaLibraryGetKernelCount(count, lib)

cdef cudaError_t cudaLibraryEnumerateKernels(cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaLibraryEnumerateKernels(kernels, numKernels, lib)

cdef cudaError_t cudaKernelSetAttributeForDevice(cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaKernelSetAttributeForDevice(kernel, attr, value, device)

cdef cudaError_t cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetExportTable(ppExportTable, pExportTableId)

cdef cudaError_t cudaGetKernel(cudaKernel_t* kernelPtr, const void* entryFuncAddr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGetKernel(kernelPtr, entryFuncAddr)
@cython.show_performance_hints(False)
cdef cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) except* nogil:
    return cyruntime._make_cudaPitchedPtr(d, p, xsz, ysz)
@cython.show_performance_hints(False)
cdef cudaPos make_cudaPos(size_t x, size_t y, size_t z) except* nogil:
    return cyruntime._make_cudaPos(x, y, z)
@cython.show_performance_hints(False)
cdef cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) except* nogil:
    return cyruntime._make_cudaExtent(w, h, d)

cdef cudaError_t cudaGraphicsEGLRegisterImage(cudaGraphicsResource** pCudaResource, EGLImageKHR image, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsEGLRegisterImage(pCudaResource, image, flags)

cdef cudaError_t cudaEGLStreamConsumerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEGLStreamConsumerConnect(conn, eglStream)

cdef cudaError_t cudaEGLStreamConsumerConnectWithFlags(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEGLStreamConsumerConnectWithFlags(conn, eglStream, flags)

cdef cudaError_t cudaEGLStreamConsumerDisconnect(cudaEglStreamConnection* conn) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEGLStreamConsumerDisconnect(conn)

cdef cudaError_t cudaEGLStreamConsumerAcquireFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t* pCudaResource, cudaStream_t* pStream, unsigned int timeout) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEGLStreamConsumerAcquireFrame(conn, pCudaResource, pStream, timeout)

cdef cudaError_t cudaEGLStreamConsumerReleaseFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t pCudaResource, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEGLStreamConsumerReleaseFrame(conn, pCudaResource, pStream)

cdef cudaError_t cudaEGLStreamProducerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, EGLint width, EGLint height) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEGLStreamProducerConnect(conn, eglStream, width, height)

cdef cudaError_t cudaEGLStreamProducerDisconnect(cudaEglStreamConnection* conn) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEGLStreamProducerDisconnect(conn)

cdef cudaError_t cudaEGLStreamProducerPresentFrame(cudaEglStreamConnection* conn, cudaEglFrame eglframe, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEGLStreamProducerPresentFrame(conn, eglframe, pStream)

cdef cudaError_t cudaEGLStreamProducerReturnFrame(cudaEglStreamConnection* conn, cudaEglFrame* eglframe, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEGLStreamProducerReturnFrame(conn, eglframe, pStream)

cdef cudaError_t cudaGraphicsResourceGetMappedEglFrame(cudaEglFrame* eglFrame, cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsResourceGetMappedEglFrame(eglFrame, resource, index, mipLevel)

cdef cudaError_t cudaEventCreateFromEGLSync(cudaEvent_t* phEvent, EGLSyncKHR eglSync, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaEventCreateFromEGLSync(phEvent, eglSync, flags)

cdef cudaError_t cudaProfilerStart() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaProfilerStart()

cdef cudaError_t cudaProfilerStop() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaProfilerStop()

cdef cudaError_t cudaGLGetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, cudaGLDeviceList deviceList) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGLGetDevices(pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList)

cdef cudaError_t cudaGraphicsGLRegisterImage(cudaGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsGLRegisterImage(resource, image, target, flags)

cdef cudaError_t cudaGraphicsGLRegisterBuffer(cudaGraphicsResource** resource, GLuint buffer, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsGLRegisterBuffer(resource, buffer, flags)

cdef cudaError_t cudaVDPAUGetDevice(int* device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaVDPAUGetDevice(device, vdpDevice, vdpGetProcAddress)

cdef cudaError_t cudaVDPAUSetVDPAUDevice(int device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaVDPAUSetVDPAUDevice(device, vdpDevice, vdpGetProcAddress)

cdef cudaError_t cudaGraphicsVDPAURegisterVideoSurface(cudaGraphicsResource** resource, VdpVideoSurface vdpSurface, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsVDPAURegisterVideoSurface(resource, vdpSurface, flags)

cdef cudaError_t cudaGraphicsVDPAURegisterOutputSurface(cudaGraphicsResource** resource, VdpOutputSurface vdpSurface, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cyruntime._cudaGraphicsVDPAURegisterOutputSurface(resource, vdpSurface, flags)

from libc.stdint cimport uintptr_t
from cuda.pathfinder import load_nvidia_dynamic_lib
cimport cuda.bindings._lib.dlfcn as dlfcn

cdef cudaError_t getLocalRuntimeVersion(int* runtimeVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    # Load
    with gil:
        loaded_dl = load_nvidia_dynamic_lib("cudart")
        handle = <void *><uintptr_t>loaded_dl._handle_uint
    __cudaRuntimeGetVersion = dlfcn.dlsym(handle, 'cudaRuntimeGetVersion')

    if __cudaRuntimeGetVersion == NULL:
        with gil:
            raise RuntimeError(f'Function "cudaRuntimeGetVersion" not found in {loaded_dl.abs_path}')

    # Call
    cdef cudaError_t err = cudaSuccess
    err = (<cudaError_t (*)(int*) except ?cudaErrorCallRequiresNewerDriver nogil> __cudaRuntimeGetVersion)(runtimeVersion)

    # We explicitly do *NOT* cleanup the library handle here, acknowledging
    # that, yes, the handle leaks. The reason is that there's a
    # `functools.cache` on the top-level caller of this function.
    #
    # This means this library would be opened once and then immediately closed,
    # all the while remaining in the cache lurking there for people to call.
    #
    # Since we open the library one time (technically once per unique library name),
    # there's not a ton of leakage, which we deem acceptable for the 1000x speedup
    # achieved by caching (ultimately) `ctypes.CDLL` calls.
    #
    # Long(er)-term we can explore cleaning up the library using higher-level
    # Python mechanisms, like `__del__` or `weakref.finalizer`s.

    return err
