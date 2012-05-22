/*
 * memtestCL_core.h
 * Public API for core memory test functions for MemtestCL
 * Includes functional and OO interfaces to GPU test functions.
 *
 * Author: Imran Haque, 2010
 * Copyright 2010, Stanford University
 *
 * This file is licensed under the terms of the LGPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */
#ifndef _MEMTESTCL_CORE_H_
#define _MEMTESTCL_CORE_H_

#include <stdio.h>
#include <iostream>
#include <list>
using namespace std;

#if defined (WINDOWS) || defined (WINNV)
    #include <windows.h>
    inline unsigned int getTimeMilliseconds(void) {
        return GetTickCount();
    }
    #include <windows.h>
	#define SLEEPMS(x) Sleep(x)
#elif defined (LINUX) || defined (OSX)
    #include <sys/time.h>
    inline unsigned int getTimeMilliseconds(void) {
        struct timeval tv;
        gettimeofday(&tv,NULL);
        return tv.tv_sec*1000 + tv.tv_usec/1000;
    }
    #include <unistd.h>
    #define SLEEPMS(x) usleep(x*1000)
#else
    #error Must #define LINUX, WINDOWS, WINNV, or OSX
#endif

#if defined (__APPLE__) || defined(MACOSX) || defined(OSX)
   #include <OpenCL/opencl.h>
#else
   #include <CL/opencl.h>
#endif

cl_int softwaitForEvents(cl_uint num_events,const cl_event* event_list,cl_command_queue const* pcq=NULL,unsigned sleeplength=1,unsigned limit=15000);


const char* descriptionOfError (cl_int err);
typedef unsigned int uint;

// Low-level OO interface to MemtestCL functions
class memtestFunctions { //{{{
protected:
    cl_context ctx;
    cl_device_id dev;
    cl_command_queue cq;
    cl_program code;
    static const int n_kernels = 12;
    cl_kernel kernels[n_kernels];
    cl_kernel &k_write_constant, &k_verify_constant;
    cl_kernel &k_logic,&k_logic_shared;
    cl_kernel &k_write_paired_constants,&k_verify_paired_constants;
    cl_kernel &k_write_w32,&k_verify_w32;
    cl_kernel &k_write_random,&k_verify_random;
    cl_kernel &k_write_mod,&k_verify_mod;
    cl_int setKernelArgs(cl_kernel& kernel,const int n_args,const size_t* sizes,const void** args) const;
public:
    memtestFunctions(cl_context context,cl_device_id device,cl_command_queue q);
    ~memtestFunctions();
    uint max_workgroup_size() const;
    cl_event writeConstant(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint constant,cl_int& status) const;
    cl_event writePairedConstants(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint constant1,const uint constant2,cl_int& status) const;
    cl_event writeWalking32Bit(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const bool ones,const uint shift,cl_int& status) const;
    cl_event writeRandomBlocks(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint seed,cl_int& status) const;
    cl_event writePairedModulo(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint shift,const uint pattern1, const uint pattern2, const uint modulus,const uint iters,cl_int& status) const;
    cl_event shortLCG0(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint repeats,const uint period,cl_int& status) const;
    cl_event shortLCG0Shmem(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint repeats,const uint period,cl_int& status) const;
    uint verifyConstant(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint constant,cl_mem blockErrorCount,uint* error_counts,cl_int& status) const;
    uint verifyPairedConstants(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint constant1,const uint constant2,cl_mem blockErrorCount,uint* error_counts,cl_int& status) const;
    uint verifyWalking32Bit(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const bool ones,const uint shift,cl_mem blockErrorCount,uint* error_counts,cl_int& status) const;
    uint verifyRandomBlocks(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint seed,cl_mem blockErrorCount,uint* error_counts,cl_int& status) const;
    uint verifyPairedModulo(const uint nBlocks,const uint nThreads,cl_mem base,uint N,const uint shift,const uint pattern1,const uint modulus,cl_mem blockErrorCount,uint* error_counts,cl_int& status) const;

}; //}}}

// OO interface to MemtestCL functions
class memtestState { //{{{
    friend class memtestMultiTester;
protected:
    cl_context ctx;
    cl_device_id dev;
    cl_command_queue cq;
    memtestFunctions memtest;
	uint nBlocks;
	uint nThreads;
    uint loopFactor;
    uint loopIters;
	uint megsToTest;
    int lcgPeriod;
	cl_mem devTestMem;
	cl_mem devTempMem;
	bool allocated;
	uint* hostTempMem;
	bool writeConstant(const uint constant) const;
	bool verifyConstant(uint& errorCount,const uint constant) const;
	bool gpuMovingInversionsPattern(uint& errorCount,const uint pattern) const;
public:
    uint initTime;
	memtestState(cl_context context, cl_device_id device);
    ~memtestState();

	uint allocate(uint mbToTest);
	void deallocate();
	bool isAllocated() const {return allocated;}
	uint size() const {return megsToTest;}
    void setLCGPeriod(int period) {lcgPeriod = period;}
    int getLCGPeriod() const {return lcgPeriod;}
    uint max_bandwidth_size() const {return megsToTest/2;}
    uint workgroup_size() const {return nThreads;}

    bool gpuMemoryBandwidth(double& bandwidth,uint mbToTest,uint iters=5);
	bool gpuShortLCG0(uint& errorCount,const uint repeats) const;
	bool gpuShortLCG0Shmem(uint& errorCount,const uint repeats) const;
	bool gpuMovingInversionsOnesZeros(uint& errorCount) const;
	bool gpuWalking8BitM86(uint& errorCount,const uint shift) const;
	bool gpuWalking8Bit(uint& errorCount,const bool ones,const uint shift) const;
	bool gpuMovingInversionsRandom(uint& errorCount) const;
	bool gpuWalking32Bit(uint& errorCount,const bool ones,const uint shift) const;
	bool gpuRandomBlocks(uint& errorCount,const uint seed) const;
	bool gpuModuloX(uint& errorCount,const uint shift,const uint pattern,const uint modulus,const uint overwriteIters) const;
}; //}}}

// Simple wrapper class around memtestState to allow multiple test regions
class memtestMultiTester {
    protected:
    list<memtestState*> testers;
    cl_context ctx;
    cl_device_id dev;
    uint lcg_period;
    bool ctx_retained;
    uint allocation_unit;
    memtestMultiTester(cl_device_id device) : dev(device), lcg_period(1024), ctx_retained(false), initTime(0)
    {
        cl_ulong maxalloc;
        clGetDeviceInfo(dev,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(cl_ulong),&maxalloc,NULL);
        // in MiB
        allocation_unit = (uint)(maxalloc/1048576);
    }
    public:
    uint initTime;
	memtestMultiTester(cl_context context, cl_device_id device) : ctx(context), dev(device), lcg_period(1024), ctx_retained(true), initTime(0)
    { //{{{
        clRetainContext(ctx);
        cl_ulong maxalloc;
        clGetDeviceInfo(dev,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(cl_ulong),&maxalloc,NULL);
        // in MiB
        allocation_unit = (uint)(maxalloc/1048576);
    }; //}}}
    virtual ~memtestMultiTester() {
        deallocate();
        if (ctx_retained) clReleaseContext(ctx);
    }

    int getLCGPeriod() const {return lcg_period;}
    uint get_allocation_unit() const {return allocation_unit;}
	bool isAllocated() const {return testers.size()>0;}
	uint size() const {
        uint totalsize = 0;
        for (list<memtestState*>::const_iterator i = testers.begin(); i != testers.end(); i++) {
            totalsize += (*i)->size();
        }
        return totalsize;
    }
    uint max_bandwidth_size() const {
            if (!isAllocated()) return 0;
            return testers.front()->size()/2;
    }
    uint workgroup_size() const {
            if (!isAllocated()) return 0;
            return testers.front()->workgroup_size();
    }
    void setLCGPeriod(int period) {
        lcg_period = period;
        for (list<memtestState*>::iterator i = testers.begin(); i != testers.end(); i++) {
            (*i)->setLCGPeriod(period);
        }
    }

	virtual uint allocate(uint mbToTest);
	virtual void deallocate();
    bool gpuMemoryBandwidth(double& bandwidth,uint mbToTest,uint iters=5);
	bool gpuShortLCG0(uint& errorCount,const uint repeats) const;
	bool gpuShortLCG0Shmem(uint& errorCount,const uint repeats) const;
	bool gpuMovingInversionsOnesZeros(uint& errorCount) const;
	bool gpuWalking8BitM86(uint& errorCount,const uint shift) const;
	bool gpuWalking8Bit(uint& errorCount,const bool ones,const uint shift) const;
	bool gpuMovingInversionsRandom(uint& errorCount) const;
	bool gpuWalking32Bit(uint& errorCount,const bool ones,const uint shift) const;
	bool gpuRandomBlocks(uint& errorCount,const uint seed) const;
	bool gpuModuloX(uint& errorCount,const uint shift,const uint pattern,const uint modulus,const uint overwriteIters) const;
}; //}}}

class memtestMultiContextTester : public memtestMultiTester {
    protected:
        cl_platform_id plat;
    public:
        memtestMultiContextTester(cl_platform_id platform,cl_device_id device) : memtestMultiTester(device), plat(platform) {}
        virtual ~memtestMultiContextTester() {};
        virtual uint allocate(uint mbToTest);
};


#endif
