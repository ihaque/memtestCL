/*
 * memtestCL_cli.cpp
 * Command-line interface frontend for MemtestCL tester
 *
 * Author: Imran Haque, 2010
 * Copyright 2010, Stanford University
 *
 * This file is licensed under the terms of the LGPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <ctype.h>

#include "ezOptionParser.hpp"

#include "memtestCL_core.h"

// For isatty
#ifdef WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#elif defined(LINUX) || defined(OSX)
#include <unistd.h>
#endif

#ifdef OSX
inline size_t strnlen(const char* s,size_t maxlen) {
    size_t i;
    for (i=0;i<maxlen && s[i];i++);
    return i;
}
#endif

bool validateNumeric(const char* str) { //{{{
    size_t idlen;
    // Assumes number will not be larger than 10 digits
    if ((idlen=strnlen(str,11))==11)
        return false;
    for (size_t i = 0; i < idlen; i++) {
        if (!isdigit(str[i])) return false;
    }
    return true;
} //}}}

int getint_range(const char* prompt,const int min,const int max)
{ //{{{
    int sel, scanf_rv;
    if (!isatty(fileno(stdout))) {
        return min;
    }
    do {
        printf("%s (%d - %d): ",prompt,min,max);
        scanf_rv = scanf("%d",&sel);
    } while (scanf_rv < 1 || sel < min || sel > max);
    getchar(); // Consume the extra newline in stdin
    return sel;
} //}}}

void print_usage(void) { //{{{
    printf("     -------------------------------------------------------------\n");
    printf("     |                       MemtestCL v1.00                     |\n");
    //printf("     |             Copyright 2009, Stanford University           |\n");
    printf("     |                                                           |\n");
    printf("     | Usage: memtestCL [flags] [MB GPU RAM to test] [# iters]   |\n");
    printf("     |                                                           |\n");
    printf("     | Defaults: GPU 0, 128MB RAM, 50 test iterations            |\n");
    printf("     | Amount of tested RAM will be rounded up to nearest 2MB    |\n");
    printf("     -------------------------------------------------------------\n\n");
    printf("      Available flags:\n");
    printf("        --platform N ,-p N   : run test on the Nth (from 0) OpenCL platform\n");
    printf("        --gpu N ,-g N        : run test on the Nth (from 0) OpenCL device\n");
    printf("                               on selected platform\n");
    printf("        --license ,-l        : show license terms for this build\n");
    printf("\n");
} //}}}

void print_licensing(void) { //{{{
    printf("Copyright 2010, Stanford University\n");
    printf("Licensed under the GNU Library General Public License (LGPL), version 3.0\n");
    printf("Please see the file COPYING in the source distribution for details\n");
    printf("\n");
    #if defined(WINDOWS) || defined(WINNV)
    printf("This software incorporates by linkage code from the libintl and libiconv\n");
    printf("libraries, which are covered by the Library GNU Public License, available\n");
    printf("at http://www.gnu.org/licenses/lgpl-3.0.txt\n");
    #endif
    return;
} //}}}


void initialize_CL(cl_platform_id &plat,cl_context& ctx,cl_device_id& dev,int& device_idx_selected,int& platform_idx_selected) { //{{{
    // Set up CL
    cl_platform_id platforms[16];
    cl_uint num_platforms;
    clGetPlatformIDs(16,platforms,&num_platforms);
    if (num_platforms == 0) {
        printf("Error: No OpenCL platforms available.\n");
        exit(2);
    }
    if (platform_idx_selected == -1 && num_platforms == 1) platform_idx_selected = 0;
    printf ("Available OpenCL platforms:\n");
    for (int i = 0; i < (int) num_platforms; i++) {
        char platname[256];
        size_t namesize;
        clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,256,platname,&namesize);
        printf("\t %d: %s",i,platname);
        if (platform_idx_selected == i) printf(" (SELECTED)");
        printf("\n");
    }
    if (platform_idx_selected < 0)
        platform_idx_selected = getint_range("Please select a platform",0,num_platforms-1);

    cl_device_id devids[32];
    uint num_gpu = 0, num_cpu = 0, num_accel = 0, num_devices = 0;
    clGetDeviceIDs(platforms[platform_idx_selected],CL_DEVICE_TYPE_GPU,32,devids,&num_gpu);
    clGetDeviceIDs(platforms[platform_idx_selected],CL_DEVICE_TYPE_ACCELERATOR,32-num_gpu,devids+num_gpu,&num_accel);
    clGetDeviceIDs(platforms[platform_idx_selected],CL_DEVICE_TYPE_CPU,32-num_gpu-num_accel,devids+num_gpu+num_accel,&num_cpu);
    num_devices = num_gpu + num_accel + num_cpu;

    if (num_devices == 0) {
        printf("Error: no OpenCL devices available on selected platform.\n");
        exit(2);
    }

    if (device_idx_selected == -1 && num_devices == 1) device_idx_selected = 0;
    printf ("Available OpenCL devices on selected platform:\n");
    for (int i = 0; i < (int) num_devices;i++) {
        char devname[256];
        size_t namesize;
        clGetDeviceInfo(devids[i],CL_DEVICE_NAME,256,devname,&namesize);
        printf("\t %d: %s",i,devname);
        if (device_idx_selected == i) printf(" (SELECTED)");
        printf("\n");
    }
    printf("\n");
    if (device_idx_selected < 0)
        device_idx_selected = getint_range("Please select a device",0,num_devices-1);
    // Sanity check device ID
    if (device_idx_selected >= (int) num_devices) {
        printf("Error: Specified invalid device index (%d); %d OpenCL devices present on selected platform, numbered from zero\n",device_idx_selected,num_devices);
        exit(2);
    }

    // Create a CL context on selected device
    cl_context_properties ctxprops[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)platforms[platform_idx_selected],0};
    cl_int clerror;
    plat = platforms[platform_idx_selected];
    ctx = clCreateContext(ctxprops,1,devids+device_idx_selected,NULL,NULL,&clerror);
    dev = devids[device_idx_selected];
    if (clerror != CL_SUCCESS) {
        printf("Error creating context: %s!\n",descriptionOfError(clerror));
        exit(2);
    }
} //}}}

int main(int argc,const char** argv) {
    uint megsToTest=128;
    uint maxIters=50;
    int gpuID=-1;
    int platID=-1;
    int showLicense = 0;
    int ramclock=-1,coreclock=-1;
    int commAuthorized=-1;
    int commBanned=0;
    
    print_usage(); 
    
    ez::ezOptionParser opt;

    opt.add(
        "0", // Default.
        0, // Required?
        1, // Number of args expected.
        0, // Delimiter if expecting multiple args.
        "run test on the Nth (from 0) OpenCL platform\n", // Help description.
        "--platform",
        "-p"
    );
    
    opt.add(
        "0", // Default.
        0, // Required?
        1, // Number of args expected.
        0, // Delimiter if expecting multiple args.
        "run test on the Nth (from 0) OpenCL device\n", // Help description.
        "--gpu",
        "-g"
    );

    opt.add(
        "", // Default.
        0, // Required?
        0, // Number of args expected.
        0, // Delimiter if expecting multiple args.
        "show license terms for this build\n", // Help description.
        "-l",
        "--license"
    );

    opt.parse(argc, argv);
    std::string lastArg;
    if(opt.isSet("-p"))
        opt.get("-p")->getInt(platID);
    if(opt.isSet("-g"))
        opt.get("-g")->getInt(gpuID);
    if(opt.isSet("-l"))
        opt.get("-g")->getInt(showLicense);
    if(opt.lastArgs.size() == 0) {
        // do nothing, use default settings
    } else if(opt.lastArgs.size() == 2) {
        sscanf(opt.lastArgs[0]->c_str(),"%u",&megsToTest);
        sscanf(opt.lastArgs[1]->c_str(),"%u",&maxIters);
    } else {
        printf("Error: Bad argument for [MB GPU RAM to test] [# iters]");
    }

    if (showLicense) print_licensing();

    cl_context ctx;
    cl_device_id dev;
    cl_platform_id plat;
    char devname[256];
    initialize_CL(plat,ctx,dev,gpuID,platID);
    clGetDeviceInfo(dev,CL_DEVICE_NAME,256,devname,NULL);

    memtestMultiTester tester(ctx,dev);
    //memtestMultiContextTester tester(plat,dev);
    if (!tester.allocate(megsToTest)) {
        printf("Error: unable to allocate %u MiB of memory to test, bailing!\n",megsToTest);
        exit(2);
    } else {
        printf("Running %u iterations of tests over %u MB of memory on device %d: %s\n\n",maxIters,tester.size(),gpuID,devname);
    }

    // Run bandwidth test
    const unsigned bw_iters = 20;
    printf("Running memory bandwidth test over %u iterations of %u MB transfers...\n",bw_iters,tester.max_bandwidth_size());
    double bandwidth;
    if (!tester.gpuMemoryBandwidth(bandwidth,tester.max_bandwidth_size(),bw_iters)) {
        printf("\tTest failed!\n");
        bandwidth = 0;
    } else {
        printf("\tEstimated bandwidth %.02f MB/s\n\n",bandwidth);
    }

    uint accumulatedErrors = 0,iterErrors;
    uint errorCounts[15];
    unsigned short iterErrorCounts[13];
    memset(errorCounts,0,15*sizeof(uint));
    memset(iterErrorCounts,0,13*sizeof(unsigned short));
   
    unsigned int start,end;
    uint iter;
    const char* test;
	bool status;
    bool thisIterFailed;
    int itersfailed = 0;
    const char *testnames[] = {"Moving inversions (ones and zeros)",
                               "Memtest86 walking 8-bit",
                               "True walking zeros (8-bit)",
                               "True walking ones (8-bit)",
                               "Moving inversions (random)",
                               "True walking zeros (32-bit)",
                               "True walking ones (32-bit)",
                               "Random blocks",
                               "Memtest86 Modulo-20",
                               "Integer logic",
                               "Integer logic (4 loops)",
                               "Integer logic (local memory)",
                               "Integer logic (4 loops, local memory)"};
                            
    for (iter = 0; iter < maxIters ; iter++) {  //{{{
        thisIterFailed = false;
        printf("Test iteration %u on %d MiB of memory on device %d (%s): %u errors so far\n",iter+1,tester.size(),gpuID,devname,accumulatedErrors);
        uint errorCount;
        
        // Moving inversions, 1's and 0's {{{
        errorCount = 0;
        test = "Moving Inversions (ones and zeros)";
        start=getTimeMilliseconds();
        status = tester.gpuMovingInversionsOnesZeros(errorCount);
        if (!status) {
            printf("Could not execute test %s; quitting\n",test);
            goto loopend;
        }
        accumulatedErrors += errorCount;
        end=getTimeMilliseconds();
        errorCounts[0] += errorCount;
        iterErrorCounts[0] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        // Moving inversions, random {{{
        errorCount = 0;
        test = "Moving Inversions (random)";
        start=getTimeMilliseconds();
        status = tester.gpuMovingInversionsRandom(errorCount);
        if (!status) {
            printf("Could not execute test %s; quitting\n",test);
            goto loopend;
        }
        accumulatedErrors += errorCount;
        end=getTimeMilliseconds();
        errorCounts[4] += errorCount;
        iterErrorCounts[4] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        // Memtest86 walking 8-bit {{{
        errorCount = 0;
        test = "Memtest86 Walking 8-bit";
        start=getTimeMilliseconds();
        for (uint shift=0;shift<8;shift++){
            status = tester.gpuWalking8BitM86(iterErrors,shift);
            if (!status) {
                printf("Could not execute test %s; quitting\n",test);
                goto loopend;
            }
            errorCount += iterErrors;
        }
        end=getTimeMilliseconds();
        accumulatedErrors+=errorCount;
        errorCounts[1] += errorCount;
        iterErrorCounts[1] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        // True Walking zeros, 8-bit {{{
        errorCount = 0;
        test = "True Walking zeros (8-bit)";
        start=getTimeMilliseconds();
        for (uint shift=0;shift<8;shift++){
            status = tester.gpuWalking8Bit(iterErrors,false,shift);
            if (!status) {
                printf("Could not execute test %s; quitting\n",test);
                goto loopend;
            }
            errorCount += iterErrors;
        }
        end=getTimeMilliseconds();
        accumulatedErrors+=errorCount;
        errorCounts[2] += errorCount;
        iterErrorCounts[2] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        // True Walking ones, 8-bit {{{
        errorCount = 0;
        test = "True Walking ones (8-bit)";
        start=getTimeMilliseconds();
        for (uint shift=0;shift<8;shift++){
            status = tester.gpuWalking8Bit(iterErrors,true,shift);
            if (!status) {
                printf("Could not execute test %s; quitting\n",test);
                goto loopend;
            }
            errorCount += iterErrors;
        }
        end=getTimeMilliseconds();
        accumulatedErrors+=errorCount;
        errorCounts[3] += errorCount;
        iterErrorCounts[3] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        // Walking zeros, 32-bit {{{
        errorCount = 0;
        test ="Memtest86 Walking zeros (32-bit)";
        start=getTimeMilliseconds();
        for (uint shift=0;shift<32;shift++){
            status = tester.gpuWalking32Bit(iterErrors,false,shift);
            if (!status) {
                printf("Could not execute test %s; quitting\n",test);
                goto loopend;
            }
            errorCount += iterErrors;
        }
        end=getTimeMilliseconds();
        accumulatedErrors+=errorCount;
        errorCounts[5] += errorCount;
        iterErrorCounts[5] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        // Walking ones, 32-bit {{{
        errorCount = 0;
        test ="Memtest86 Walking ones (32-bit)";
        start=getTimeMilliseconds();
        for (uint shift=0;shift<32;shift++){
            status = tester.gpuWalking32Bit(iterErrors,true,shift);
            if (!status) {
                printf("Could not execute test %s; quitting\n",test);
                goto loopend;
            }
            errorCount += iterErrors;
        }
        end=getTimeMilliseconds();
        accumulatedErrors+=errorCount;
        errorCounts[6] += errorCount;
        iterErrorCounts[6] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        // Random blocks {{{
        errorCount = 0;
        test="Random blocks";
        start=getTimeMilliseconds();
        status = tester.gpuRandomBlocks(errorCount,rand());
        if (!status) {
            printf("Could not execute test %s; quitting\n",test);
            goto loopend;
        }
        accumulatedErrors += errorCount;
        errorCounts[7] += errorCount;
        iterErrorCounts[7] += (errorCount) ? 1 : 0;
        end=getTimeMilliseconds();
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        // Modulo-20, 32-bit {{{
        errorCount = 0;
        test ="Memtest86 Modulo-20";
        start=getTimeMilliseconds();
        for (uint shift=0;shift<20;shift++){
            status = tester.gpuModuloX(iterErrors,shift,rand(),20,2);
            if (!status) {
                printf("Could not execute test %s; quitting\n",test);
                goto loopend;
            }
            errorCount += iterErrors;
        }
        end=getTimeMilliseconds();
        accumulatedErrors+=errorCount;
        errorCounts[8] += errorCount;
        iterErrorCounts[8] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        // Logic, 1 iteration {{{
        errorCount = 0;
        test = "Logic (one iteration)";
        start=getTimeMilliseconds();
        status = tester.gpuShortLCG0(errorCount,1);
        if (!status) {
            printf("Could not execute test %s; quitting\n",test);
            goto loopend;
        }
        end=getTimeMilliseconds();
        accumulatedErrors += errorCount;
        errorCounts[9] += errorCount;
        iterErrorCounts[9] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        // Logic, 4 iterations {{{
        errorCount = 0;
        test = "Logic (4 iterations)";
        start=getTimeMilliseconds();
        status = tester.gpuShortLCG0(errorCount,4);
        if (!status) {
            printf("Could not execute test %s; quitting\n",test);
            goto loopend;
        }
        end=getTimeMilliseconds();
        accumulatedErrors += errorCount;
        errorCounts[10] += errorCount;
        iterErrorCounts[10] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
       // Logic, shared-memory, 1 iteration {{{
        errorCount = 0;
        test = "Logic (local memory, one iteration)";
        start=getTimeMilliseconds();
        status = tester.gpuShortLCG0Shmem(errorCount,1);
        if (!status) {
            printf("Could not execute test %s; quitting\n",test);
            goto loopend;
        }
        end=getTimeMilliseconds();
        accumulatedErrors += errorCount;
        errorCounts[11] += errorCount;
        iterErrorCounts[11] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        // Logic, shared-memory, 4 iterations {{{
        errorCount = 0;
        test = "Logic (local memory, 4 iterations)";
        start=getTimeMilliseconds();
        status = tester.gpuShortLCG0Shmem(errorCount,4);
        if (!status) {
            printf("Could not execute test %s; quitting\n",test);
            goto loopend;
        }
        end=getTimeMilliseconds();
        accumulatedErrors += errorCount;
        errorCounts[12] += errorCount;
        iterErrorCounts[12] += (errorCount) ? 1 : 0;
        thisIterFailed = thisIterFailed || errorCount;
        printf("\t%s: %u errors (%u ms)\n",test,errorCount,end-start);
        // }}}
        
        if (thisIterFailed) itersfailed++;
        printf("\n");
    } //}}}
    loopend:
    clReleaseContext(ctx);
    if (!status) { // One of the tests failed
        return 1;
    } else {
        printf("Test summary:\n");
        printf("-----------------------------------------\n");
        printf("%u iterations over %u MiB of memory on device %s\n",iter,tester.size(),devname);
        for (int i = 0; i < 13; i++) {
            printf("%40s: %d failed iterations\n",testnames[i],iterErrorCounts[i]);
	    printf("                                         (%d total incorrect bits)\n",errorCounts[i]);
        }
        if (itersfailed)
            printf("Final error count: %d test iterations with at least one error; %u errors total\n",itersfailed,accumulatedErrors);
        else
            printf("Final error count: 0 errors\n");
        if (isatty(fileno(stdout))) {
            int i = 0;
            printf("\nPress <enter> to quit.\n");
            i = getchar();
        }
        return (accumulatedErrors != 0);
    }
}
