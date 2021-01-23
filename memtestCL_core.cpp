/*
 * memtestCL_core.cu
 * MemtestCL core memory test functions and OOP interfaces to tester.
 *
 * Author: Imran Haque, 2010
 * Copyright 2010, Stanford University
 *
 * This file is licensed under the terms of the LGPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

#include "memtestCL_core.h"

#include <iostream>
using namespace std;

cl_int softwaitForEvents(cl_uint num_events,const cl_event* event_list,cl_command_queue const* pcq,unsigned sleeplength,unsigned limit)
{
    #ifdef SOFTWAIT_IS_HARDWAIT
    cl_int status = clWaitForEvents(num_events,event_list);
    for (uint i = 0; i < num_events; i++) clReleaseEvent(event_list[i]);
    if (status < 0) cerr << "Status was "<<descriptionOfError(status)<<endl;
    return status;
    #else
    if (num_events == 0 || event_list == NULL) return CL_INVALID_VALUE;
    unsigned int start = getTimeMilliseconds();
    #if defined(CL_VERSION_1_1) && defined(USE_CL_11)
    if (num_events > 1) {
        cl_context ctx0,ctxn;
        clGetEventInfo(event_list[0],CL_EVENT_CONTEXT,sizeof(cl_context),&ctx0,NULL);
        if (err != CL_SUCCESS) return err;
        for (uint i = 0; i < num_events; i++) {
            cl_int err = clGetEventInfo(event_list[i],CL_EVENT_CONTEXT,sizeof(cl_context),&ctxn,NULL);
            if (ctx0 != ctxn) return CL_INVALID_CONTEXT;
            if (err != CL_SUCCESS) return err;
        }
    }
    #endif //OpenCL 1.1 new
    bool anyfailures = false;
    for (uint i = 0; i < num_events; i++) {
        cl_int status;
        cl_int err;
        // If we do not flush the queue, commands may never get issued
        cl_command_queue cq;
        if (pcq == NULL) {
            err = clGetEventInfo(event_list[0],CL_EVENT_COMMAND_QUEUE,sizeof(cl_command_queue),&cq,NULL);
            pcq = &cq;
        }
        //cerr << "Status of clGetEventInfo was "<<descriptionOfError(err)<<endl;
        err = clFlush(*pcq);
        //cerr << "Status of clFlush  was "<<descriptionOfError(err)<<endl;
        // Now wait for completion
        err = clGetEventInfo(event_list[i],CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(cl_int),&status,NULL);
        if (err != CL_SUCCESS) return err;
        while (status != CL_COMPLETE && status >= 0) {
            unsigned int current = getTimeMilliseconds();
            if ((current-start) > limit) return CL_INVALID_VALUE; // TODO: is this the best error?
            //cout << status << endl;
            SLEEPMS(sleeplength);
            err = clGetEventInfo(event_list[i],CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(cl_int),&status,NULL);
            if (err != CL_SUCCESS) return err;
        }
        clReleaseEvent(event_list[i]);
        anyfailures = anyfailures || (status < 0);
    }
    #if defined(CL_VERSION_1_1) && defined(USE_CL_11)
    return anyfailures ? CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST : CL_SUCCESS;
    #else
    return CL_SUCCESS;
    #endif
    #endif
}

inline static void checkCLErr(cl_int err, const char * name) {
   const char* errtext = descriptionOfError(err);

    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
                 << " (" << err << ") " << errtext << std::endl;
        exit(EXIT_FAILURE);
    }
}

memtestState::memtestState(cl_context context, cl_device_id device) : 
    ctx(context), dev(device), cq(clCreateCommandQueue(ctx,dev,0,NULL)),
    memtest(ctx,dev,cq),
    nBlocks(1024), nThreads(512), loopFactor(1), lcgPeriod(1024),
    allocated(false), hostTempMem(NULL), initTime(0)
{ 
    clRetainContext(ctx);
    cl_device_type devtype;
    clGetDeviceInfo(dev,CL_DEVICE_TYPE,sizeof(cl_device_type),&devtype,NULL);
    cl_uint maxdims;
    clGetDeviceInfo(dev,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,sizeof(cl_uint),&maxdims,NULL);
    size_t* maxextents = new size_t[maxdims];
    clGetDeviceInfo(dev,CL_DEVICE_MAX_WORK_ITEM_SIZES,maxdims*sizeof(size_t),maxextents,NULL);
    switch (devtype) {
        case CL_DEVICE_TYPE_GPU:
            nThreads = memtest.max_workgroup_size(); break;
            break;
        case CL_DEVICE_TYPE_CPU:
            nBlocks = 32; nThreads = 1;
            break;
        default:
            break;
    }
    loopFactor = 524288/(nBlocks*nThreads);
    //cout << nBlocks << " work-groups of "<<nThreads<<" work-items each with a loop-factor of "<<loopFactor<<endl;
    delete[] maxextents;

}
memtestState::~memtestState() {
    deallocate();
    clReleaseCommandQueue(cq);
    clReleaseContext(ctx);
}
void memtestState::deallocate() {
    if (!allocated) return;
    clReleaseMemObject(devTempMem);
    clReleaseMemObject(devTestMem);
    free(hostTempMem);
    allocated = false;
}
uint memtestState::allocate(uint mbToTest) {
		deallocate();

        initTime = getTimeMilliseconds();
		
        // Round up to nearest 2MiB
		if (mbToTest % 2) mbToTest++;

		megsToTest = mbToTest;
		loopIters = megsToTest/2;
        loopIters *= loopFactor;

		if (megsToTest == 0) return 0;
		cl_int err;
		try {
            // AMD's OpenCL will throw an error on allocation, NVIDIA on use. So both alloc and try to init.
            devTestMem = clCreateBuffer(ctx,CL_MEM_READ_WRITE,megsToTest*1048576ULL,NULL,&err);
            if (err != CL_SUCCESS) {
                cerr << "Unable to allocate OpenCL memory: "<<descriptionOfError(err)<<endl;
                throw 1;
            }
            memtest.writeConstant(nBlocks,nThreads,devTestMem,loopIters,0,err);
            if (err != CL_SUCCESS) {
                cerr << "Unable to allocate OpenCL memory: "<<descriptionOfError(err)<<endl;
                throw 1;
            }

            devTempMem = clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizeof(uint)*nBlocks,NULL,&err);
            if (err != CL_SUCCESS) {
                cerr << "Unable to allocate OpenCL memory: "<<descriptionOfError(err)<<endl;
                throw 2;
            }
            memtest.writeConstant(1,1,devTempMem,1,0,err);
            if (err != CL_SUCCESS) {
                cerr << "Unable to allocate OpenCL memory: "<<descriptionOfError(err)<<endl;
                throw 2;
            }

			hostTempMem = (uint*)malloc(sizeof(uint)*nBlocks);
            if (hostTempMem == NULL) {
                cerr << "Unable to allocate host memory: general failure"<<endl;
                throw 3;
            }
		} catch (int allocFailed) {
            // Clear CUDA error flag for outside world
            switch (allocFailed) {
                case 3:
                    clReleaseMemObject(devTempMem);
                case 2:
                    clReleaseMemObject(devTestMem);
                case 1:
                    break;
                default:
                    cerr<<"Invalid allocation failure type in memtestState::allocate\n";
                    exit(1);
            }
			return 0;
		}
		allocated = true;
		return megsToTest;
}
bool memtestState::gpuMemoryBandwidth(double& bandwidth,uint mbToTest,uint iters) {
    if (!allocated || mbToTest > max_bandwidth_size()) return false;

    cl_int err;
    cl_event *events = new cl_event[iters];
    
    uint start = getTimeMilliseconds();
    for (uint i = 0; i < iters; i++) {
        err = clEnqueueCopyBuffer(cq,devTestMem,devTestMem,0,mbToTest*1048576ULL,mbToTest*1048576ULL,0,NULL,events+i);
        if (err != CL_SUCCESS) {
            cerr << "Status of clEnqueueCopyBuffer was "<<descriptionOfError(err)<<endl;
            return false;
        }
    }
    err = softwaitForEvents(iters,events,&cq);

    uint end = getTimeMilliseconds();
	
    // Calculate bandwidth in MiB/s
	// Multiply by 2 since we are reading and writing to the same memory
    bandwidth = 2.0*((double)mbToTest*iters)/((end-start)/1000.0);
    delete[] events;
    return err == CL_SUCCESS;
}
bool memtestState::writeConstant(const uint constant) const {
	if (!allocated) return false;
    cl_int status;
	cl_event event = memtest.writeConstant(nBlocks,nThreads,devTestMem,loopIters,constant,status);
    return status == CL_SUCCESS && softwaitForEvents(1,&event,&cq) == CL_SUCCESS;
}
bool memtestState::verifyConstant(uint& errorCount,const uint constant) const {
	if (!allocated) return false;
	cl_int status;
    errorCount = memtest.verifyConstant(nBlocks,nThreads,devTestMem,loopIters,constant,devTempMem,hostTempMem,status);
    return status == CL_SUCCESS;
}
bool memtestState::gpuShortLCG0(uint& errorCount,const uint repeats) const {
	if (!allocated) return false;
    cl_int status;
	cl_event event = memtest.shortLCG0(nBlocks,nThreads,devTestMem,loopIters,repeats,lcgPeriod,status);
    if (status != CL_SUCCESS || softwaitForEvents(1,&event,&cq) != CL_SUCCESS) return false;
    
    errorCount = memtest.verifyConstant(nBlocks,nThreads,devTestMem,loopIters,0,devTempMem,hostTempMem,status);
    return status == CL_SUCCESS;
}
bool memtestState::gpuShortLCG0Shmem(uint& errorCount,const uint repeats) const {
	if (!allocated) return false;
	cl_int status;
	cl_event event = memtest.shortLCG0Shmem(nBlocks,nThreads,devTestMem,loopIters,repeats,lcgPeriod,status);
    if (status != CL_SUCCESS || softwaitForEvents(1,&event,&cq) != CL_SUCCESS) return false;
    
    errorCount = memtest.verifyConstant(nBlocks,nThreads,devTestMem,loopIters,0,devTempMem,hostTempMem,status);
    return status == CL_SUCCESS;
}
bool memtestState::gpuMovingInversionsPattern(uint& errorCount,const uint pattern) const {
	if (!allocated) return false;
    uint partialErrorCount;

    if (!writeConstant(pattern)) return false;
    if (!verifyConstant(partialErrorCount,pattern)) return false;
    errorCount = partialErrorCount;

    if (!writeConstant(~pattern)) return false;
    if (!verifyConstant(partialErrorCount,~pattern)) return false;
    errorCount += partialErrorCount;
    return true;
}
bool memtestState::gpuMovingInversionsOnesZeros(uint& errorCount) const {
    return gpuMovingInversionsPattern(errorCount,0xFFFFFFFF);
}
bool memtestState::gpuWalking8BitM86(uint& errorCount,const uint shift) const {
	if (!allocated) return false;
    // Performs the Memtest86 variation on the walking 8-bit pattern, where the same shifted pattern is
    // written into each 32-bit word in memory, verified, and its complement written and verified
    uint pattern = 1 << (shift & 0x7);
    pattern = pattern | (pattern << 8) | (pattern << 16) | (pattern << 24);

    return gpuMovingInversionsPattern(errorCount,pattern);
}
bool memtestState::gpuMovingInversionsRandom(uint& errorCount) const {
	if (!allocated) return false;
    uint pattern = (uint)rand();
    return gpuMovingInversionsPattern(errorCount,pattern);
}
bool memtestState::gpuWalking8Bit(uint& errorCount,const bool ones,const uint shift) const {
	if (!allocated) return false;
    cl_event event;
    cl_int status;
    // Implements one iteration of true walking 8-bit ones/zeros test
    uint patterns[2]={0x0,0x0};
    
    // Build the walking-ones paired pattern of 8-bits with the given shift
    uint bits = 0x1 << (shift & 0x7);
    for (uint i = 0; i < 4; i++) {
        patterns[0] = (patterns[0] << 8) | bits;
        bits = (bits == 0x80) ? 0x01 : bits<<1;
    }
    for (uint i = 0; i < 4; i++) {
        patterns[1] = (patterns[1] << 8) | bits;
        bits = (bits == 0x80) ? 0x01 : bits<<1;
    }

    if (!ones) {
        patterns[0] = ~patterns[0];
        patterns[1] = ~patterns[1];
    }

	event = memtest.writePairedConstants(nBlocks,nThreads,devTestMem,loopIters,patterns[0],patterns[1],status);
    if (status != CL_SUCCESS || softwaitForEvents(1,&event,&cq) != CL_SUCCESS) return false;

    errorCount = memtest.verifyPairedConstants(nBlocks,nThreads,devTestMem,loopIters,patterns[0],patterns[1],devTempMem,hostTempMem,status);
    return status == CL_SUCCESS;

}
bool memtestState::gpuWalking32Bit(uint& errorCount,const bool ones,const uint shift) const {
	if (!allocated) return false;
    cl_event event;
    cl_int status;

	event = memtest.writeWalking32Bit(nBlocks,nThreads,devTestMem,loopIters,ones,shift,status);
    if (status != CL_SUCCESS || softwaitForEvents(1,&event,&cq) != CL_SUCCESS) return false;

    errorCount = memtest.verifyWalking32Bit(nBlocks,nThreads,devTestMem,loopIters,ones,shift,devTempMem,hostTempMem,status);
    return status == CL_SUCCESS;
}
bool memtestState::gpuRandomBlocks(uint& errorCount,const uint seed) const {
	if (!allocated) return false;
    cl_event event;
    cl_int status;

	event = memtest.writeRandomBlocks(nBlocks,nThreads,devTestMem,loopIters,seed,status);
    if (status != CL_SUCCESS || softwaitForEvents(1,&event,&cq) != CL_SUCCESS) return false;

    errorCount = memtest.verifyRandomBlocks(nBlocks,nThreads,devTestMem,loopIters,seed,devTempMem,hostTempMem,status);
    return status == CL_SUCCESS;
}
bool memtestState::gpuModuloX(uint& errorCount,const uint shift,const uint pattern,const uint modulus,const uint overwriteIters) const {
	if (!allocated) return false;
    cl_event event;
    cl_int status;
    uint realShift = shift % modulus;
    errorCount = 0;
    uint currentPattern = pattern;

    for (int i = 0; i < 2; i++, currentPattern = ~currentPattern) {
	    event = memtest.writePairedModulo(nBlocks,nThreads,devTestMem,loopIters,realShift,currentPattern,~currentPattern,modulus,overwriteIters,status);
        if (status != CL_SUCCESS || softwaitForEvents(1,&event,&cq) != CL_SUCCESS) return false;

        errorCount += memtest.verifyPairedModulo(nBlocks,nThreads,devTestMem,loopIters,realShift,currentPattern,modulus,devTempMem,hostTempMem,status);
        if (status != CL_SUCCESS) return false;
    
    }
    return true;
}

memtestFunctions::memtestFunctions(cl_context context,cl_device_id device,cl_command_queue q): ctx(context),dev(device),cq(q),
    k_write_constant(kernels[0]),k_verify_constant(kernels[1]),k_logic(kernels[2]),k_logic_shared(kernels[3]),
    k_write_paired_constants(kernels[4]),k_verify_paired_constants(kernels[5]),k_write_w32(kernels[6]),k_verify_w32(kernels[7]),
    k_write_random(kernels[8]),k_verify_random(kernels[9]),k_write_mod(kernels[10]),k_verify_mod(kernels[11])
{
    #include "memtestCL_kernels.clh"
    size_t kernel_length = memtestCL_kernels_len;
    cl_int err;
    const char* kernelcode = (char*) &memtestCL_kernels[0];
    clRetainContext(ctx);
    clRetainCommandQueue(cq);
    code = clCreateProgramWithSource(ctx,1,&kernelcode,&kernel_length,&err);
    checkCLErr(err,"clCreateProgramWithSource");
    err = clBuildProgram(code,1,&dev,"",NULL,NULL);
    if (err != CL_SUCCESS /* || true */) { //== CL_BUILD_PROGRAM_FAILURE) {
        char buildlog[16384];
        clGetProgramBuildInfo(code,device,CL_PROGRAM_BUILD_LOG,16384,buildlog,NULL);
        std::cout<<"Error building CL kernels:\n"<<buildlog<<std::endl;
        std::cout<<"\n\n\n";
        unsigned char* binary = new unsigned char[131072];
        size_t binlen;
        clGetProgramInfo(code,CL_PROGRAM_BINARIES,131071,&binary,&binlen);
        binary[binlen] = '\0';
        std::cout << "Binary rep ("<<binlen<<"bytes):\n";
        std::cout << binary;
        delete[] binary;
        if (err != CL_SUCCESS) exit(2);
    }
    checkCLErr(err,"clBuildProgram");
    k_write_constant = clCreateKernel(code,"deviceWriteConstant",&err);
    checkCLErr(err,"k_write_constant");
    k_verify_constant = clCreateKernel(code,"deviceVerifyConstant",&err);
    checkCLErr(err,"k_verify_constant");
    k_logic = clCreateKernel(code,"deviceShortLCG0",&err);
    checkCLErr(err,"k_logic");
    k_logic_shared = clCreateKernel(code,"deviceShortLCG0Shmem",&err);
    checkCLErr(err,"k_logic_shared");
    k_write_paired_constants = clCreateKernel(code,"deviceWritePairedConstants",&err);
    checkCLErr(err,"k_write_paired_constants");
    k_verify_paired_constants = clCreateKernel(code,"deviceVerifyPairedConstants",&err);
    checkCLErr(err,"k_verify_paired_constants");
    k_write_w32 = clCreateKernel(code,"deviceWriteWalking32Bit",&err);
    checkCLErr(err,"k_write_w32");
    k_verify_w32 = clCreateKernel(code,"deviceVerifyWalking32Bit",&err);
    checkCLErr(err,"k_verify_w32");
    k_write_random = clCreateKernel(code,"deviceWriteRandomBlocks",&err);
    checkCLErr(err,"k_write_random");
    k_verify_random = clCreateKernel(code,"deviceVerifyRandomBlocks",&err);
    checkCLErr(err,"k_verify_random");
    k_write_mod = clCreateKernel(code,"deviceWritePairedModulo",&err);
    checkCLErr(err,"k_write_mod");
    k_verify_mod = clCreateKernel(code,"deviceVerifyPairedModulo",&err);
    checkCLErr(err,"k_verify_mod");
}
memtestFunctions::~memtestFunctions() {
    clReleaseKernel(k_write_constant);
    clReleaseKernel(k_verify_constant);
    clReleaseKernel(k_logic);
    clReleaseKernel(k_logic_shared);
    clReleaseKernel(k_write_paired_constants);
    clReleaseKernel(k_verify_paired_constants);
    clReleaseKernel(k_write_w32);
    clReleaseKernel(k_verify_w32);
    clReleaseKernel(k_write_random);
    clReleaseKernel(k_verify_random);
    clReleaseKernel(k_write_mod);
    clReleaseKernel(k_verify_mod);
    clReleaseProgram(code);
    clReleaseCommandQueue(cq);
    clReleaseContext(ctx);
}
cl_int memtestFunctions::setKernelArgs(cl_kernel& kernel,const int n_args,const size_t* sizes,const void** args) const {
    char kername[256];
    clGetKernelInfo(kernel,CL_KERNEL_FUNCTION_NAME,256,kername,NULL);
    for (int i = 0; i < n_args; i++) {
        cl_int status = clSetKernelArg(kernel,i,sizes[i],args[i]);
        if (status != CL_SUCCESS) {
            cout << "Error "<<descriptionOfError(status) <<" setting argument "<<i<<" of kernel "<<kername<<endl;
            break;
        }
    }
    return CL_SUCCESS;
}
uint memtestFunctions::max_workgroup_size() const {
        uint maxsize = 0xFFFFFFFF;
        size_t kernelsize;
        for (int i = 0; i < n_kernels; i++) {
            clGetKernelWorkGroupInfo(kernels[i],dev,CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t),&kernelsize,NULL);
            maxsize = (kernelsize < maxsize) ? kernelsize : maxsize;

            char kername[256];
            clGetKernelInfo(kernels[i],CL_KERNEL_FUNCTION_NAME,256,kername,NULL);
            //cout << "Max WG-size for kernel "<<kername<<" is "<<kernelsize<<endl;
        }
        //cout << "Max size possible is "<<maxsize;
        return maxsize;
}
cl_event memtestFunctions::writeConstant(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint constant,cl_int& status) const {
    cl_event event;
    const int n_args = 3;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(uint)};
    const void*  args[]  = {&base,&N,&constant};
    status = setKernelArgs(k_write_constant,n_args,sizes,args);
    if (status != CL_SUCCESS) return event;
    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    //cout << "Enqueueing writeConstant kernel with "<<total_threads<<" total threads over "<<nBlocks<<" work-groups for "<<nThreads<<" items per group"<<endl;
    status = clEnqueueNDRangeKernel(cq,k_write_constant,1,NULL,&total_threads,&local_threads,0,NULL,&event);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing writeConstant kernel"<<endl; return event;}
    return event;
}
cl_event memtestFunctions::writePairedConstants(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint constant1,const uint constant2,cl_int& status) const {
    cl_event event;
    const int n_args = 4;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(uint),sizeof(uint)};
    const void*  args[]  = {&base, &N, &constant1, &constant2};
    status = setKernelArgs(k_write_paired_constants,n_args,sizes,args);
    if (status != CL_SUCCESS) return event;

    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    status = clEnqueueNDRangeKernel(cq,k_write_paired_constants,1,NULL,&total_threads,&local_threads,0,NULL,&event);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing writePairedConstants kernel"<<endl; return event;}
    return event;
}
cl_event memtestFunctions::writeWalking32Bit(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const bool ones,const uint shift,cl_int& status) const {
    cl_event event;
    cl_int iones = (int)ones;
    const int n_args = 4;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(cl_int),sizeof(uint)};
    const void*  args[]  = {&base, &N, &iones, &shift};
    status = setKernelArgs(k_write_w32,n_args,sizes,args);
    if (status != CL_SUCCESS) return event;
    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    status = clEnqueueNDRangeKernel(cq,k_write_w32,1,NULL,&total_threads,&local_threads,0,NULL,&event);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing writeWalking32Bit kernel"<<endl; return event;}
    return event;
}
cl_event memtestFunctions::writeRandomBlocks(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint seed,cl_int& status) const {
    cl_event event;
    const int n_args = 4;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(uint),4*nThreads};
    const void*  args[]  = {&base, &N, &seed, NULL};
    status = setKernelArgs(k_write_random,n_args,sizes,args);
    if (status != CL_SUCCESS) return event;
    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    status = clEnqueueNDRangeKernel(cq,k_write_random,1,NULL,&total_threads,&local_threads,0,NULL,&event);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing writeRandomBlocks kernel"<<endl; return event;}
    return event;
}
cl_event memtestFunctions::writePairedModulo(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint shift,const uint pattern1, const uint pattern2, const uint modulus,const uint iters,cl_int& status) const {
    cl_event event;
    const int n_args = 7;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(uint),sizeof(uint),sizeof(uint),sizeof(uint),sizeof(uint)};
    const void*  args[]  = {&base, &N, &shift,&pattern1,&pattern2,&modulus,&iters};
    status = setKernelArgs(k_write_mod,n_args,sizes,args);
    if (status != CL_SUCCESS) return event;

    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    status = clEnqueueNDRangeKernel(cq,k_write_mod,1,NULL,&total_threads,&local_threads,0,NULL,&event);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing writePairedModulo kernel"<<endl; return event;}
    return event;
}
cl_event memtestFunctions::shortLCG0(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint repeats,const uint period,cl_int& status) const {
    cl_event event;
    const int n_args = 4;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(uint),sizeof(uint)};
    const void*  args[]  = {&base, &N, &repeats, &period};
    status = setKernelArgs(k_logic,n_args,sizes,args);
    if (status != CL_SUCCESS) return event;

    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    status = clEnqueueNDRangeKernel(cq,k_logic,1,NULL,&total_threads,&local_threads,0,NULL,&event);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing LCG0 kernel"<<endl; return event;}
    return event;
}
cl_event memtestFunctions::shortLCG0Shmem(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint repeats,const uint period,cl_int& status) const {
    cl_event event;
    const int n_args = 5;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(uint),sizeof(uint),sizeof(uint)*nThreads};
    const void*  args[]  = {&base, &N, &repeats, &period,NULL};
    status = setKernelArgs(k_logic_shared,n_args,sizes,args);
    if (status != CL_SUCCESS) return event;

    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    status = clEnqueueNDRangeKernel(cq,k_logic_shared,1,NULL,&total_threads,&local_threads,0,NULL,&event);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing LCG0Shmem kernel"<<endl; return event;}
    return event;
}

uint memtestFunctions::verifyConstant(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint constant,cl_mem blockErrorCount,uint* error_counts,cl_int& status) const {
    const int n_args = 5;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(uint),sizeof(cl_mem),sizeof(uint)*nThreads};
    const void*  args[]  = {&base, &N, &constant, &blockErrorCount,NULL};
    status = setKernelArgs(k_verify_constant,n_args,sizes,args);
    if (status != CL_SUCCESS) return -1;

    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    //cout << "Enqueueing verifyConstant kernel with "<<total_threads<<" total threads over "<<nBlocks<<" work-groups for "<<local_threads<<" items per group"<<endl;
    status = clEnqueueNDRangeKernel(cq,k_verify_constant,1,NULL,&total_threads,&local_threads,0,NULL,NULL);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing verifyConstant kernel"<<endl; return (uint)-1;}
    status = clEnqueueReadBuffer(cq,blockErrorCount,CL_TRUE,0,nBlocks*sizeof(uint),error_counts,0,NULL,NULL);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing verifyConstant readback"<<endl; return (uint)-1;}

    uint totalErrors = 0;
    for (uint i = 0; i < nBlocks; i++) {
         totalErrors += error_counts[i];
    }
    return totalErrors;
}
uint memtestFunctions::verifyPairedConstants(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint constant1,const uint constant2,cl_mem blockErrorCount,uint* error_counts,cl_int& status) const {
    const int n_args = 6;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(uint),sizeof(uint),sizeof(cl_mem),sizeof(uint)*nThreads};
    const void*  args[]  = {&base, &N, &constant1, &constant2, &blockErrorCount,NULL};
    status = setKernelArgs(k_verify_paired_constants,n_args,sizes,args);
    if (status != CL_SUCCESS) return -1;

    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    status = clEnqueueNDRangeKernel(cq,k_verify_paired_constants,1,NULL,&total_threads,&local_threads,0,NULL,NULL);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing verifyPairedConstants kernel"<<endl; return (uint)-1;}
    status = clEnqueueReadBuffer(cq,blockErrorCount,CL_TRUE,0,nBlocks*sizeof(uint),error_counts,0,NULL,NULL);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing verifyPairedConstants readback"<<endl; return (uint)-1;}

    uint totalErrors = 0;
    for (uint i = 0; i < nBlocks; i++) {
         totalErrors += error_counts[i];
    }
    return totalErrors;
}
uint memtestFunctions::verifyWalking32Bit(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const bool ones,const uint shift,cl_mem blockErrorCount,uint* error_counts,cl_int& status) const {
    cl_int iones = (int)ones;
    const int n_args = 6;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(cl_int),sizeof(uint),sizeof(cl_mem),sizeof(uint)*nThreads};
    const void*  args[]  = {&base, &N, &iones, &shift, &blockErrorCount,NULL};
    status = setKernelArgs(k_verify_w32,n_args,sizes,args);
    if (status != CL_SUCCESS) return -1;

    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    clEnqueueNDRangeKernel(cq,k_verify_w32,1,NULL,&total_threads,&local_threads,0,NULL,NULL);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing verifyWalking32Bit kernel"<<endl; return (uint)-1;}
    status = clEnqueueReadBuffer(cq,blockErrorCount,CL_TRUE,0,nBlocks*sizeof(uint),error_counts,0,NULL,NULL);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing verifyWalking32Bit readback"<<endl; return (uint)-1;}

    uint totalErrors = 0;
    for (uint i = 0; i < nBlocks; i++) {
         totalErrors += error_counts[i];
    }
    return totalErrors;
}
uint memtestFunctions::verifyRandomBlocks(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint seed,cl_mem blockErrorCount,uint* error_counts,cl_int& status) const {
    const int n_args = 7;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(uint),sizeof(cl_mem),sizeof(uint)*nThreads,sizeof(uint)*nThreads,sizeof(uint)*nThreads};
    const void*  args[]  = {&base, &N, &seed, &blockErrorCount,NULL,NULL,NULL};
    status = setKernelArgs(k_verify_random,n_args,sizes,args);
    if (status != CL_SUCCESS) return -1;

    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    clEnqueueNDRangeKernel(cq,k_verify_random,1,NULL,&total_threads,&local_threads,0,NULL,NULL);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing verifyRandomBlocks kernel"<<endl; return (uint)-1;}
    status = clEnqueueReadBuffer(cq,blockErrorCount,CL_TRUE,0,nBlocks*sizeof(uint),error_counts,0,NULL,NULL);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing verifyRandomBlocks readback"<<endl; return (uint)-1;}

    uint totalErrors = 0;
    for (uint i = 0; i < nBlocks; i++) {
         totalErrors += error_counts[i];
    }
    return totalErrors;
}
uint memtestFunctions::verifyPairedModulo(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint shift,const uint pattern1,const uint modulus,cl_mem blockErrorCount,uint* error_counts,cl_int& status) const {
    const int n_args = 7;
    size_t sizes[] = {sizeof(cl_mem),sizeof(uint),sizeof(uint),sizeof(uint),sizeof(uint),sizeof(cl_mem),sizeof(uint)*nThreads};
    const void* args[]  = {&base, &N, &shift, &pattern1, &modulus, &blockErrorCount,NULL};
    status = setKernelArgs(k_verify_mod,n_args,sizes,args);
    if (status != CL_SUCCESS) return -1;

    size_t total_threads = nBlocks*nThreads;
    size_t local_threads = nThreads;
    clEnqueueNDRangeKernel(cq,k_verify_mod,1,NULL,&total_threads,&local_threads,0,NULL,NULL);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing verifyModuloX kernel"<<endl; return (uint)-1;}
    status = clEnqueueReadBuffer(cq,blockErrorCount,CL_TRUE,0,nBlocks*sizeof(uint),error_counts,0,NULL,NULL);
    if (status != CL_SUCCESS) {cout << "Error "<< descriptionOfError(status) <<" queueing verifyModuloX readback"<<endl; return (uint)-1;}

    uint totalErrors = 0;
    for (uint i = 0; i < nBlocks; i++) {
         totalErrors += error_counts[i];
    }
    return totalErrors;
}

uint memtestMultiTester::allocate(uint mbToTest) {
    uint totalmb = mbToTest;
    if (totalmb & 1) totalmb++;
    try {
        while (mbToTest > 0) {
            uint amount = allocation_unit < mbToTest ? allocation_unit : mbToTest;
            //cout << "Allocating new tester of "<<amount<<" MiB \n";
            memtestState* tester = new memtestState(ctx,dev);
            if (!tester->allocate(amount)) throw 1;
            testers.push_back(tester);
            mbToTest -= amount;
        }
    } catch (int error) {
        for (list<memtestState*>::iterator i = testers.begin(); i != testers.end(); i++) {
            delete *i;
        }
        testers.clear();
        return 0;
    }
    //cout << "Allocated "<<totalmb<<" over "<<testers.size()<<" testers\n";
    return totalmb;
}
void memtestMultiTester::deallocate() {
    if (!isAllocated()) return;
    for (list<memtestState*>::iterator i = testers.begin(); i != testers.end(); i++) {
        delete *i;
    }
    testers.clear();
}
bool memtestMultiTester::gpuMemoryBandwidth(double& bandwidth,uint mbToTest,uint iters) {
    if (!isAllocated()) return false;
    if (mbToTest > max_bandwidth_size()) return false;
    return testers.front()->gpuMemoryBandwidth(bandwidth,mbToTest,iters);
}
bool memtestMultiTester::gpuShortLCG0(uint& errorCount,const uint repeats) const {
    uint partialErrorCount;
    bool status;
    errorCount = 0;
    for (list<memtestState*>::const_iterator i = testers.begin(); i != testers.end(); i++) {
        status = (*i)->gpuShortLCG0(partialErrorCount,repeats);
        errorCount += partialErrorCount;
        if (!status) return false;
    }
    return true;
}
bool memtestMultiTester::gpuShortLCG0Shmem(uint& errorCount,const uint repeats) const {
    uint partialErrorCount;
    bool status;
    errorCount = 0;
    for (list<memtestState*>::const_iterator i = testers.begin(); i != testers.end(); i++) {
        status = (*i)->gpuShortLCG0Shmem(partialErrorCount,repeats);
        errorCount += partialErrorCount;
        if (!status) return false;
    }
    return true;
}
bool memtestMultiTester::gpuMovingInversionsOnesZeros(uint& errorCount) const {
    uint partialErrorCount;
    bool status;
    errorCount = 0;
    for (list<memtestState*>::const_iterator i = testers.begin(); i != testers.end(); i++) {
        status = (*i)->gpuMovingInversionsOnesZeros(partialErrorCount);
        errorCount += partialErrorCount;
        if (!status) return false;
    }
    return true;
}
bool memtestMultiTester::gpuWalking8BitM86(uint& errorCount,const uint shift) const {
    uint partialErrorCount;
    bool status;
    errorCount = 0;
    for (list<memtestState*>::const_iterator i = testers.begin(); i != testers.end(); i++) {
        status = (*i)->gpuWalking8BitM86(partialErrorCount,shift);
        errorCount += partialErrorCount;
        if (!status) return false;
    }
    return true;
}
bool memtestMultiTester::gpuWalking8Bit(uint& errorCount,const bool ones,const uint shift) const {
    uint partialErrorCount;
    bool status;
    errorCount = 0;
    for (list<memtestState*>::const_iterator i = testers.begin(); i != testers.end(); i++) {
        status = (*i)->gpuWalking8Bit(partialErrorCount,ones,shift);
        errorCount += partialErrorCount;
        if (!status) return false;
    }
    return true;
}
bool memtestMultiTester::gpuMovingInversionsRandom(uint& errorCount) const {
    uint partialErrorCount;
    bool status;
    errorCount = 0;
    uint pattern = (uint)rand();
    // This one is different from the rest to preserve semantics of test
    for (list<memtestState*>::const_iterator i = testers.begin(); i != testers.end(); i++) {
        status = (*i)->gpuMovingInversionsPattern(partialErrorCount,pattern);
        errorCount += partialErrorCount;
        if (!status) return false;
    }
    return true;
}
bool memtestMultiTester::gpuWalking32Bit(uint& errorCount,const bool ones,const uint shift) const {
    uint partialErrorCount;
    bool status;
    errorCount = 0;
    for (list<memtestState*>::const_iterator i = testers.begin(); i != testers.end(); i++) {
        status = (*i)->gpuWalking32Bit(partialErrorCount,ones,shift);
        errorCount += partialErrorCount;
        if (!status) return false;
    }
    return true;
}
bool memtestMultiTester::gpuRandomBlocks(uint& errorCount,const uint seed) const {
    uint partialErrorCount;
    bool status;
    errorCount = 0;
    for (list<memtestState*>::const_iterator i = testers.begin(); i != testers.end(); i++) {
        status = (*i)->gpuRandomBlocks(partialErrorCount,seed);
        errorCount += partialErrorCount;
        if (!status) return false;
    }
    return true;
}
bool memtestMultiTester::gpuModuloX(uint& errorCount,const uint shift,const uint pattern,const uint modulus,const uint overwriteIters) const {
    uint partialErrorCount;
    bool status;
    errorCount = 0;
    for (list<memtestState*>::const_iterator i = testers.begin(); i != testers.end(); i++) {
        status = (*i)->gpuModuloX(partialErrorCount,shift,pattern,modulus,overwriteIters);
        errorCount += partialErrorCount;
        if (!status) return false;
    }
    return true;
}

uint memtestMultiContextTester::allocate(uint mbToTest) {	
    uint totalmb = mbToTest;
    if (totalmb & 1) totalmb++;
    cl_context_properties ctxprops[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)plat,0};
    cl_int clerror;
    try {
        while (mbToTest > 0) {
            uint amount = allocation_unit < mbToTest ? allocation_unit : mbToTest;
            //cout << "Allocating new tester of "<<amount<<" MiB in its own context\n";
            // Create a new context for this tester
            cl_context ctx = clCreateContext(ctxprops,1,&dev,NULL,NULL,&clerror);
            if (clerror != CL_SUCCESS) {
                printf("Error creating context: %s!\n",descriptionOfError(clerror));
                throw 2;
            }
            memtestState* tester = new memtestState(ctx,dev);
            if (!tester->allocate(amount)) throw 1;
            testers.push_back(tester);
            mbToTest -= amount;
        }
    } catch (int error) {
        for (list<memtestState*>::iterator i = testers.begin(); i != testers.end(); i++) {
            delete *i;
        }
        testers.clear();
        return 0;
    }
    cout << "Allocated "<<totalmb<<" over "<<testers.size()<<" testers\n";
    return totalmb;
}

const char* descriptionOfError (cl_int err) { //{{{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default: return "Unknown";
    }
} //}}}

