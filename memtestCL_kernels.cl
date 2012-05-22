/*
 * memtestCL_kernels.cl
 * MemtestCL core memory testing kernels
 *
 * Author: Imran Haque, 2010
 * Copyright 2010, Stanford University
 *
 * This file is licensed under the terms of the LGPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

 /*
  * OpenCL grid layout: Linear in work-groups and work-items.
  * Intended usage = 1k workgroups, 512 wi/wg, with N words (iterations) per thread
  *     for devices that cannot support that many, keep (#WGs) * (#WIs) = 524288 and increase N
  *     -> 2*N MiB tested per grid
  * wi address at iteration i = base + blockIdx.x * N * blockDim.x + i*blockDim.x + threadIdx.x (in CUDA notation...)
  *
  */

#define THREAD_ADDRESS(base,N,i) (base + get_group_id(0) * N * get_local_size(0) + i * get_local_size(0) + get_local_id(0))
#define THREAD_OFFSET(N,i) (get_group_id(0) * N * get_local_size(0) + i * get_local_size(0) + get_local_id(0))
#define BITSDIFF(x,y) __popc((x) ^ (y))

#define threadIdx get_local_id(0)
#define blockIdx get_group_id(0)
#define blockDim get_local_size(0)

#define OLD_M20_SYNC
#define MODX_WITHOUT_MOD

#ifdef OLD_M20_SYNC
#define M20_SYNC() barrier(CLK_LOCAL_MEM_FENCE)
#else
#define M20_SYNC() barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#endif

// Device population count, since not defined in OpenCL
// From Wikipedia, optimized for case with few nonzero bits
unsigned __popc(uint x) { //{{{
#define f(y) if ((x &= x-1) == 0) return y;
    if (x == 0) return 0;
    f( 1) f( 2) f( 3) f( 4) f( 5) f( 6) f( 7) f( 8)
    f( 9) f(10) f(11) f(12) f(13) f(14) f(15) f(16)
    f(17) f(18) f(19) f(20) f(21) f(22) f(23) f(24)
    f(25) f(26) f(27) f(28) f(29) f(30) f(31)
    return 32;
#undef f
} //}}}


// Utility functions to write/verify pure constants in memory 
__kernel void deviceWriteConstant(__global uint* base, uint N, const uint konstant) { //{{{
    for (uint i = 0 ; i < N; i++) {      
        *(THREAD_ADDRESS(base,N,i)) = konstant;
    }
} //}}}
__kernel void deviceVerifyConstant(__global uint* base,uint N,const uint konstant,__global uint* blockErrorCount,__local uint* threadErrorCount) { //{{{
    // Verifies memory at base to make sure it has a constant pattern
    // Sums number of errors found in block and stores error count into blockErrorCount[group_id]
    // Sum-reduce this array afterwards to get total error count over tested region
    // Uses 4*blockDim bytes of shared memory

    threadErrorCount[threadIdx] = 0;

    for (uint i = 0; i < N; i++) {
        //if ( *(THREAD_ADDRESS(base,N,i)) != constant ) threadErrorCount[threadIdx]++;
        threadErrorCount[threadIdx] += BITSDIFF(*(THREAD_ADDRESS(base,N,i)),konstant);
    }
    // Parallel-reduce error counts over threads in block
    for (uint stride = blockDim>>1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadIdx < stride)
            threadErrorCount[threadIdx] += threadErrorCount[threadIdx + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (threadIdx == 0)
        blockErrorCount[blockIdx] = threadErrorCount[0];
    
    return;
}
//}}}


// Logic test //{{{
// Idea: Run a varying number of iterations (k*N) of a short-period (per=N) LCG that returns to zero (or F's) quickly
// Store only the result of the last iteration
// Compare output to the desired constant
// Compare results between varying k - memory error rate for a given pattern should be constant,
//                                     so variation should be due to logic errors in loop count
// Put the LCG loop into a macro so we don't repeat code between versions of logic tester.
// The paired XOR adds diversity to the instruction stream, and is not reduced to a NOT
// as a single XOR is (verified with decuda).
// {{{
#define LCGLOOP(var,repeats,period,a,c) for (uint rep = 0; rep < repeats; rep++) {\
    (var) = ~(var);\
    for (uint iter = 0; iter < period; iter++) {\
        (var) = ~(var);\
        (var) = (a)*(var)+(c);\
        (var) ^= 0xFFFFFFF0;\
        (var) ^= 0xF;\
    }\
    (var) = ~(var);\
}
//}}} }}}

__kernel void deviceShortLCG0(__global uint* base,uint N,uint repeats,const int period) { //{{{
    // Pick a different block for different LCG lengths
    // Short periods are useful if LCG goes inside for i in 0..N loop
    int a,c;
    switch (period) {
        case 1024: a = 0x0fbfffff; c = 0x3bf75696; break;
        case 512:  a = 0x61c8647f; c = 0x2b3e0000; break;
        case 256:  a = 0x7161ac7f; c = 0x43840000; break;
        case 128:  a = 0x0432b47f; c = 0x1ce80000; break;
        case 2048: a = 0x763fffff; c = 0x4769466f; break;
        default:   a = 0; c = 0; break;
    }
    
    uint value = 0;
    LCGLOOP(value,repeats,period,a,c)

    for (uint i = 0 ; i < N; i++) {
        *(THREAD_ADDRESS(base,N,i)) = value;
    }
} //}}} 
// _shmem version uses shared memory to store inter-iteration values
// is more sensitive to shared memory errors from (eg) shader overclocking 
__kernel void deviceShortLCG0Shmem(__global uint* base,uint N,uint repeats,const int period,__local uint* shmem) { //{{{
    // Pick a different block for different LCG lengths
    // Short periods are useful if LCG goes inside for i in 0..N loop
    int a,c;
    switch (period) {
        case 1024: a = 0x0fbfffff; c = 0x3bf75696; break;
        case 512:  a = 0x61c8647f; c = 0x2b3e0000; break;
        case 256:  a = 0x7161ac7f; c = 0x43840000; break;
        case 128:  a = 0x0432b47f; c = 0x1ce80000; break;
        case 2048: a = 0x763fffff; c = 0x4769466f; break;
        default:   a = 0; c = 0; break;
    }
    shmem[threadIdx] = 0;
    LCGLOOP(shmem[threadIdx],repeats,period,a,c)

    for (uint i = 0 ; i < N; i++) {
        *(THREAD_ADDRESS(base,N,i)) = shmem[threadIdx];

    }
} //}}} //}}}



// Writes paired constants to memory, such that each offset that is X mod 2 receives patterns[X]
// Used for true walking-ones/zeros 8-bit test
__kernel void deviceWritePairedConstants(__global uint* base,uint N,uint pattern0,uint pattern1) { //{{{
    //const uint pattern = (threadIdx & 0x1) ? pattern1 : pattern0;
    uint isodd = threadIdx & 0x1;
    isodd *= 0xFFFFFFFF;
    //const uint pattern = isodd ? pattern1: pattern0;
    const uint pattern = (isodd & pattern1) | ((~isodd) & pattern0);
    for (uint i = 0 ; i < N; i++) {      
        *(THREAD_ADDRESS(base,N,i)) = pattern;
    }

} //}}}

__kernel void deviceVerifyPairedConstants(__global uint* base,uint N,uint pattern0,uint pattern1,__global uint* blockErrorCount,__local uint* threadErrorCount) { //{{{
    // Verifies memory at base to make sure it has a correct paired-constant pattern
    // Sums number of errors found in block and stores error count into blockErrorCount[blockIdx]
    // Sum-reduce this array afterwards to get total error count over tested region
    // Uses 4*blockDim bytes of shared memory
    
    threadErrorCount[threadIdx] = 0;
    //const uint pattern = patterns[threadIdx & 0x1];
    uint isodd = threadIdx & 0x1;
    isodd *= 0xFFFFFFFF;
    //const uint pattern = isodd ? pattern1: pattern0;
    const uint pattern = (isodd & pattern1) | ((~isodd) & pattern0);
    
    for (uint i = 0; i < N; i++) {
        //if ( *(THREAD_ADDRESS(base,N,i)) != pattern ) threadErrorCount[threadIdx]++;
        threadErrorCount[threadIdx] += BITSDIFF(*(THREAD_ADDRESS(base,N,i)),pattern);
    }
    // Parallel-reduce error counts over threads in block
    for (uint stride = blockDim>>1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadIdx < stride)
            threadErrorCount[threadIdx] += threadErrorCount[threadIdx + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (threadIdx == 0)
        blockErrorCount[blockIdx] = threadErrorCount[0];
    
    return;
}
//}}}

__kernel void deviceWriteWalking32Bit(__global uint* base,uint N,int ones,uint shift) { //{{{
    // Writes one iteration of the walking-{ones/zeros} 32-bit pattern to gpu memory

    // Want to write in a 1 << (offset from base + shift % 32)
    // Since thread indices are aligned with base, this reduces to
    // 1 << ((threadIdx+shift) & 0x1f)
    // With conditional inversion for walking zeros
    uint pattern = 1 << ((threadIdx + shift) & 0x1f);
    pattern = ones ? pattern : ~pattern;
    
    for (uint i = 0; i < N; i++) {
        *(THREAD_ADDRESS(base,N,i)) = pattern;
    }
} //}}}

__kernel void deviceVerifyWalking32Bit(__global uint* base,uint N,int ones,uint shift,__global uint* blockErrorCount,__local uint* threadErrorCount) { //{{{
    // Verifies memory at base to make sure it has a constant pattern
    // Sums number of errors found in block and stores error count into blockErrorCount[blockIdx]
    // Sum-reduce this array afterwards to get total error count over tested region
    // Uses 4*blockDim bytes of shared memory
    
    threadErrorCount[threadIdx] = 0;

    uint pattern = 1 << ((threadIdx + shift) & 0x1f);
    pattern = ones ? pattern : ~pattern;
    
    for (uint i = 0; i < N; i++) {
        //if ( *(THREAD_ADDRESS(base,N,i)) != pattern ) threadErrorCount[threadIdx]++;
        threadErrorCount[threadIdx] += BITSDIFF(*(THREAD_ADDRESS(base,N,i)),pattern);
    }
    // Parallel-reduce error counts over threads in block
    for (uint stride = blockDim>>1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadIdx < stride)
            threadErrorCount[threadIdx] += threadErrorCount[threadIdx + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (threadIdx == 0)
        blockErrorCount[blockIdx] = threadErrorCount[0];
    
    return;
}
//}}}

// Math functions modulo the Mersenne prime 2^31 -1 {{{
void deviceMul3131 (uint v1, uint v2,uint* LO, uint* HI)
{
    // Given v1, v2 < 2^31
    // Emulate a 31-bit integer multiply by doing instead a 32-bit multiply into LO and HI
    // And shifting bits around to make it look right.
    *LO = v1*v2;
    *HI = mul_hi(v1,v2);
    *HI <<= 1;
    *HI |= ((*LO) & 0x80000000) >> 31;
    *LO &= 0x7FFFFFFF;
    
}

uint deviceModMP31(uint LO,uint HI) {
    // Modulo a 62-bit number HI<<31 + LO, mod 2^31-1
    // Encyclopedia of Cryptography and Security By Henk C. A. van Tilborg
    // page 381, Mersenne Primes
    uint sum = LO+HI;
    if (sum >= 0x80000000) {
        // If a+b > 2^31, then high bit will be set
        return sum - 0x80000000 + 1;
    } else {
        return sum;
    }
}
uint deviceMulMP31(uint a,uint b) {
    // Multiplies a pair of 31-bit integers a and b mod the Mersenne prime 2^31-1
    // Takes result through a 62-bit intermediate
    uint LO,HI;
    deviceMul3131(a,b,&LO,&HI);
    return deviceModMP31(LO,HI);
}

uint deviceExpoModMP31(uint base,uint exponent) {
    uint result = 1;
    while (exponent > 0) {
        if (exponent & 1) {
            result = deviceMulMP31(result,base);
        }
        exponent >>= 1;
        base = deviceMulMP31(base,base);
    }
    return result;
}
//}}}
// deviceRan0p: Parallelized closed-form version of NR's ran0  {{{
uint deviceRan0p(int seed,int n) { // 
    uint an = deviceExpoModMP31(16807,n+1);
    return deviceMulMP31(an,seed);
}
//}}}
// deviceIrbit2: random bit generation, from NR {{{
int deviceIrbit2(uint* seed) {
    const uint IB1  = 1;
    const uint IB2  = 2;
    const uint IB5  = 16;
    const uint IB18 = 131072;
    const uint MASK = IB1+IB2+IB5;
    if ((*seed) & IB18) {
        *seed = (((*seed) ^ MASK) << 1) | IB1;
        return 1;
    } else {
        *seed <<= 1;
        return 0;
    }
}
int deviceIrbit2_local(__local uint* seed) {
    const uint IB1  = 1;
    const uint IB2  = 2;
    const uint IB5  = 16;
    const uint IB18 = 131072;
    const uint MASK = IB1+IB2+IB5;
    if ((*seed) & IB18) {
        *seed = (((*seed) ^ MASK) << 1) | IB1;
        return 1;
    } else {
        *seed <<= 1;
        return 0;
    }
}
//}}}
__kernel void deviceWriteRandomBlocks(__global uint* base,uint N,int seed,__local uint* randomBlock) { //{{{
    // Requires 4*nThreads bytes of local memory
    // Make sure seed is not zero.
    if (seed == 0) seed = 123459876+blockIdx;
    uint bitSeed = deviceRan0p(seed + threadIdx,threadIdx);

    for (uint i=0; i < N; i++) {
        // Generate a block of random numbers in parallel using closed-form expression for ran0
        // OR in a random bit because Ran0 will never have the high bit set
        randomBlock[threadIdx] = deviceRan0p(seed,threadIdx) | (deviceIrbit2(&bitSeed) << 31);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Set the seed for the next round to the last number calculated in this round
        seed = randomBlock[blockDim-1];

        // Prevent a race condition in which last work-item can overwrite seed before others have read it
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Blit shmem block out to global memory
        *(THREAD_ADDRESS(base,N,i)) = randomBlock[threadIdx];
    }
}
//}}}
__kernel void deviceVerifyRandomBlocks(__global uint* base,uint N,int seed,__global uint* blockErrorCount,__local uint* threadErrorCount,__local uint* randomBlock,__local uint* bitSeeds) { //{{{
    // Verifies memory at base to make sure it has a correct random pattern given the seed
    // Sums number of errors found in block and stores error count into blockErrorCount[blockIdx]
    // Sum-reduce this array afterwards to get total error count over tested region
    // Uses 12*blockDim bytes of local memory
    
    threadErrorCount[threadIdx] = 0;

    // Make sure seed is not zero.
    if (seed == 0) seed = 123459876+blockIdx;
    //uint bitSeed = deviceRan0p(seed + threadIdx,threadIdx);
    bitSeeds[threadIdx] = deviceRan0p(seed + threadIdx,threadIdx);
    for (uint i = 0; i < N; i++) {
        // Generate a block of random numbers in parallel using closed-form expression for ran0
        // OR in a random bit because Ran0 will never have the high bit set
        //randomBlock[threadIdx] = deviceRan0p(seed,threadIdx) | (deviceIrbit2(bitSeed) << 31);
        randomBlock[threadIdx] = deviceRan0p(seed,threadIdx) | (deviceIrbit2_local(bitSeeds+threadIdx) << 31);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Set the seed for the next round to the last number calculated in this round
        seed = randomBlock[blockDim-1];
        
        //if ( randomBlock[threadIdx] != *(THREAD_ADDRESS(base,N,i))) threadErrorCount[threadIdx]++;
        threadErrorCount[threadIdx] += BITSDIFF(*(THREAD_ADDRESS(base,N,i)),randomBlock[threadIdx]);
        
    }

    // Parallel-reduce error counts over threads in block
    for (uint stride = blockDim>>1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadIdx < stride)
            threadErrorCount[threadIdx] += threadErrorCount[threadIdx + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (threadIdx == 0)
        blockErrorCount[blockIdx] = threadErrorCount[0];
    
    return;
}
//}}}

#ifndef MODX_WITHOUT_MOD
__kernel void deviceWritePairedModulo(__global uint* base,const uint N,const uint shift,const uint pattern1,const uint pattern2,const uint modulus,const uint iters) { //{{{
    // First writes pattern1 into every offset that is 0 mod modulus
    // Next  (iters times) writes ~pattern1 into every other address
    uint offset;
    for (uint i = 0 ; i < N; i++) {      
        offset = THREAD_OFFSET(N,i);
        if ((offset % modulus) == shift) *(base+offset) = pattern1;
    }
    M20_SYNC();
    for (uint j = 0; j < iters; j++) {
        for (uint i = 0 ; i < N; i++) {      
            offset = THREAD_OFFSET(N,i);
            if ((offset % modulus) != shift) *(base+offset) = pattern2;
        }
    }
} //}}}
#else
__kernel void deviceWritePairedModulo(__global uint* base,const uint N,const uint shift,const uint pattern1,const uint pattern2,const uint modulus,const uint iters) { //{{{
    // First writes pattern1 into every offset that is 0 mod modulus
    // Next  (iters times) writes ~pattern1 into every other address

    // We will consider the memory as a Kx[modulus] matrix (maybe with a partial last row)
    // In the first loop, we only write if our column index == shift
    // In the second loop we only write if column index != shift
    // Each thread is guaranteed N iterations so bounds checking is not a problem

    const uint startoff = (get_group_id(0) * N * get_local_size(0) + threadIdx);
    const uint startrow = startoff / modulus;
    const uint startcol = startoff - (startrow*modulus); // threadIdx % modulus
    const uint row_per_workgroup = blockDim / modulus;
    const uint col_per_workgroup = blockDim - (modulus * row_per_workgroup);
    uint offset;
    uint row, col;
    row = startrow;
    col = startcol;
    for (uint i = 0 ; i < N; i++) {
        offset = row * modulus + col;
        if (col == shift) *(base+offset) = pattern1;
        row += row_per_workgroup;
        col += col_per_workgroup;
        if (col >= modulus) {
            col -= modulus;
            row++;
        }
    }
    M20_SYNC();
    row = startrow;
    col = startcol;
    for (uint j = 0; j < iters; j++) {
        for (uint i = 0 ; i < N; i++) {
            offset = row * modulus + col;
            if (col != shift) *(base+offset) = pattern2;
            row += row_per_workgroup;
            col += col_per_workgroup;
            if (col >= modulus) {
                col -= modulus;
                row++;
            }
        }
    }
} //}}}
#endif
__kernel void deviceVerifyPairedModulo(__global uint* base,uint N,const uint shift,const uint pattern1,const uint modulus,__global uint* blockErrorCount,__local uint* threadErrorCount) { //{{{
    // Verifies that memory at each (offset mod modulus == shift) stores pattern1
    // Sums number of errors found in block and stores error count into blockErrorCount[blockIdx]
    // Sum-reduce this array afterwards to get total error count over tested region
    // Uses 4*blockDim bytes of shared memory
    threadErrorCount[threadIdx] = 0;
    uint offset;
    
    for (uint i = 0; i < N; i++) {
        offset = THREAD_OFFSET(N,i);
        //if (((offset % modulus) == shift) && (*(base+offset) != pattern1)) threadErrorCount[threadIdx]++;
        if ((offset % modulus) == shift) threadErrorCount[threadIdx] += BITSDIFF(*(base+offset),pattern1);
    }
    // Parallel-reduce error counts over threads in block
    for (uint stride = blockDim>>1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadIdx < stride)
            threadErrorCount[threadIdx] += threadErrorCount[threadIdx + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (threadIdx == 0)
        blockErrorCount[blockIdx] = threadErrorCount[0];
    
    return;
}
//}}}
