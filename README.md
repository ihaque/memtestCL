README for MemtestCL open source edition
Version 1.00
Imran Haque
12 Aug 2010

CONTENTS
1. Description
2. How to build
3. Using MemtestCL as a library
4. CLI Standalone Basic Usage
5. CLI Standalone Advanced Usage
6. FAQ
7. Licensing


1. DESCRIPTION

MemtestCL is a program to test the memory and logic of OpenCL-enabled
GPUs, CPUs, and accelerators for errors. It is an OpenCL port of our CUDA-
based tester for NVIDIA GPUs, MemtestG80.

This is the open-source version of MemtestCL, implementing the same memory
tests as the closed-source version. The intended usage is as a library so that
other software developers can use the MemtestCL tests to validate the correct
operation of GPUs or accelerators in their own code. In addition to the core
memory testing libraries, this package contains the source code to a limited
version of the command-line interface standalone tester included in the closed-
source build; certain capabilities, such as the ability to transmit results 
back to Stanford, are not present in the open-source version.

Up-to-date versions of both the closed- and open-source versions of MemtestCL
may be downloaded at https://simtk.org/home/memtest. The closed-source version
is available as precompiled binaries; the open-source version is available only
as a source package.

This document concerns the open-source version.

2. HOW TO BUILD

First, ensure that you have installed an OpenCL SDK (typically, either that in
the NVIDIA CUDA toolkits (from 3.0 onwards) or the ATI Stream SDK. Binaries
built using either SDK should execute on any OpenCL implementation (ie, it is
not required that the binary be built using the ATI Stream SDK to run on ATI
GPUs). Common paths for SDK installations are included in the Makefiles; it
may be necessary to adjust such paths to correspond to your installation.

Makefiles for 32- and 64-bit Linux, Mac OS X, and 32-bit Windows are included
On Linux and OSX, it should be possible to build MemtestCL by executing the 
following from the root of the source package:

    make -f Makefiles/Makefile.OS

where OS is one of linux32, linux64, osx. On Windows, the Visual Studio C++
compiler and make system is required (tested under VS2005); execute the
following command to build:

    nmake -f Makefiles\Makefile.windows

The resulting executable, memtestCL, should be immediately executable
on Linux and OS X platforms. On Windows, libiconv-2.dll, libintl-2.dll, and
popt1.dll must be copied from the popt/win32 subdirectory into a directory in
the DLL search path (most conveniently, the root of the source distribution).
MemtestCL uses the MIT/X licensed popt library to handle command line
arguments; precompiled static libraries are provided for Linux and OS X, but
dynamic libraries for Windows.

3. USING MEMTESTCL AS A LIBRARY

We encourage software developers to use MemtestCL as a code library in their
programs to verify the correct operation of hardware on which they execute. The
code is licensed under the LGPL, so developers of both open- and closed-source
software can use it - developers of closed-source software are required to link
to MemtestCL via a shared library (.so, .dll) mechanism; open-source software
can integrate it via static linkage.

The API for the memory tests is defined in memtestCL_core.h. There are two
APIs - a low-level API defined by the memtestFunctions class (which is a thin
wrapper around the underlyinh OpenCL kernel invocations), and a high-level API
defined by the memtestState and memtestMultiTester classes. At the lowest level
the tests are implemented by the kernels in memtestCL_kernels.cl.

The recommended interface is the memtestMultiTester class, which automatically encapsulates details such as the maximum per-buffer allocation in a particular OCL library. An example of the API's usage can be found in the standalone tester,
memtestCL_cli.cu.

4. CLI STANDALONE BASIC USAGE

MemtestCL is available for Windows, Linux, and Mac OS X-based machines. In the
following directions, please replace "MemtestCL" with the name of the program
included in the distribution for your operating system.

MemtestCL is a command line application; to run it, start it from a command
prompt (Start->Run->cmd in Windows, Terminal.app in OS X). For basic operation,
just run it from the command prompt:

    MemtestCL

By default, MemtestCL will test 128 megabytes of memory on the first OpenCL
device on the first OpenCL platform found,running 50 iterations of its tests.
On typical machines, each iteration will complete in under 10 seconds with
these parameters (the speed will vary both with the speed of the card tested
and the amount of memory tested). The amount of memory tested and number of 
test iterations can be modified by adding command line parameters as follows:

    MemtestCL [amount of RAM in megabytes] [number of test iterations]

For example, to run MemtestCL over 256 megabytes of RAM, with 100 test
iterations, execute the following command:
    
    MemtestCL 256 100

Be aware that not all of the memory on your video card can be tested, as part
of it is reserved for use by the operating system, and (as of this writing)
both ATI and NVIDIA OpenCL drivers severely restrict the amount of memory 
available to an OpenCL program running on a GPU. If too large a test region
is specified, the program will print a warning and quit. Also, if the tested
GPU is currently driving a graphical desktop, the driver may impose time
limits on test execution such that tests over very large test regions will
time out. Timeouts or other execution errors will be trapped and will cause
the test to terminate early. Due to the currently immature state of OpenCL
implementations, they may also cause the program to crash.

If you suspect that your graphics card is having issues (for example, it fails
running Folding@home work units), we strongly recommend that you test as large
a memory region as is practical, and run thousands of test iterations. In our
testing, we have found that even "problematic" cards may only fail sporadically
(e.g., once every 50,000 test iterations). Like other stress testing tools,
to properly verify stability MemtestCL should be run for an extended period of
time.

5. CLI STANDALONE ADVANCED USAGE

MemtestCL supports the use of various command line flags to enable
advanced functionality. Flags may be issued in any order, and may precede
or follow the memory size and iteration count parameters (but the memory size
must always precede the iteration count).

To run MemtestCL on an OpenCL platform other than the first (e.g., if you have
both the AMD and NVIDIA OpenCL implementations installed), use the --platform
or -p flags, passing the index of the platform to test (starting at zero). If
you do not know the index of the OpenCL platform you want, just run MemtestCL
with no parameters - a list of all platforms found will be printed immediately
following the usage summary. For example, to run MemtestCL on the second
platform in a system:
    
    MemtestCL --platform 1

To run MemtestCL on an OpenCL device (e.g., GPU) other than the first one on
the selected platform, use the --gpu or -g flags, passing the index of the
device to test (starting at zero). MemtestCL prints a list of all devices on
the selected platform (and their indices) before running tests. For example,
to run MemtestCL on the third OpenCL device on the default platform (platform
index 0):
    
    MemtestCL --gpu 2

The --platform and --gpu flags may be combined, to select a different platform
and device. This may be necessary in multi-vendor or multi-GPU configurations.
Refer to the list of platforms and devices on the current platform printed at
program start to determine the right combination. To select the third GPU on
the second platform:

    MemtestCL --platform 1 --gpu 2

At the beginning of test execution, MemtestCL issues a prompt to receive
consent to transmit the results of test data back to Stanford. No personally
identifying information is transmitted. To assist in automation, this answer
can be provided at the command prompt. To implicitly answer "yes" (that is, to
transmit results back), use the --forcecomm or -f options:
    
    MemtestCL --forcecomm

To implicitly answer "no" (that is, to forbid the tester from communicating
with Stanford) use the --bancomm or -b options:

    MemtestCL --bancomm

If transmitting data back to Stanford, the memory and core (non-shader) clock
speeds of the card are very useful data. The tester will normally prompt for
these. To provide them at the command line, use the --coreclock/-c and
--ramclock/-r options:

    MemtestCL --forcecomm --ramclock 700 --coreclock 650

Finally, to display the license agreement for MemtestCL, provide the --license
or -l options:

    MemtestCL -l

6. Frequently Asked Questions

    - I have an {ATI 2xxx/3xxx ,NVIDIA 5/6/7-series} video card and it doesn't
      work!
        - Only OpenCL-capable hardware is supported. As of this writing, this
          includes any NVIDIA GeForce 8-series or newer graphics cards (e.g.,
          GeForce 8-, 9-, GT- GTS-, GTX-series; Quadro FX, and Tesla boards),
          and the ATI Radeon 4xxx and 5xxx-series graphics cards. MemtestCL
          has also been tested on Intel and AMD CPUs using the AMD OpenCL
          implementation. OpenCL implementations exist for the Cell processor,
          but this has not been tested.

    - I have an OpenCL-enabled card, but it still doesn't work!
        - You must have an OpenCL-enabled graphics driver and OpenCL runtime
          installed. For NVIDIA hardware, this requires a version 195 ForceWare
          driver or newer; ATI requires BOTH a v9.12 or newer Catalyst driver,
          as well as (at the time of writing) the ATI Stream SDK.
        
    - I get an error complaining about a missing "OpenCL.dll" on Windows!
        - You must install OpenCL. NVIDIA bundles OpenCL.dll with recent
          (v195 or newer) drivers. For ATI GPUs, you must install an OpenCL-
          capable video driver, and also install the ATI Stream SDK
          (http://developer.amd.com/gpu/atistreamsdk/). For CPU support,
          installing only the ATI Stream SDK is sufficient.

6. Licensing

The source code to the open-source edition of MemtestCL is Copyright 2010,
Stanford University, and is licensed under the terms of the GNU Lesser General
Public License, version 3, reproduced below:

		   GNU LESSER GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.


  This version of the GNU Lesser General Public License incorporates
the terms and conditions of version 3 of the GNU General Public
License, supplemented by the additional permissions listed below.

  0. Additional Definitions.

  As used herein, "this License" refers to version 3 of the GNU Lesser
General Public License, and the "GNU GPL" refers to version 3 of the GNU
General Public License.

  "The Library" refers to a covered work governed by this License,
other than an Application or a Combined Work as defined below.

  An "Application" is any work that makes use of an interface provided
by the Library, but which is not otherwise based on the Library.
Defining a subclass of a class defined by the Library is deemed a mode
of using an interface provided by the Library.

  A "Combined Work" is a work produced by combining or linking an
Application with the Library.  The particular version of the Library
with which the Combined Work was made is also called the "Linked
Version".

  The "Minimal Corresponding Source" for a Combined Work means the
Corresponding Source for the Combined Work, excluding any source code
for portions of the Combined Work that, considered in isolation, are
based on the Application, and not on the Linked Version.

  The "Corresponding Application Code" for a Combined Work means the
object code and/or source code for the Application, including any data
and utility programs needed for reproducing the Combined Work from the
Application, but excluding the System Libraries of the Combined Work.

  1. Exception to Section 3 of the GNU GPL.

  You may convey a covered work under sections 3 and 4 of this License
without being bound by section 3 of the GNU GPL.

  2. Conveying Modified Versions.

  If you modify a copy of the Library, and, in your modifications, a
facility refers to a function or data to be supplied by an Application
that uses the facility (other than as an argument passed when the
facility is invoked), then you may convey a copy of the modified
version:

   a) under this License, provided that you make a good faith effort to
   ensure that, in the event an Application does not supply the
   function or data, the facility still operates, and performs
   whatever part of its purpose remains meaningful, or

   b) under the GNU GPL, with none of the additional permissions of
   this License applicable to that copy.

  3. Object Code Incorporating Material from Library Header Files.

  The object code form of an Application may incorporate material from
a header file that is part of the Library.  You may convey such object
code under terms of your choice, provided that, if the incorporated
material is not limited to numerical parameters, data structure
layouts and accessors, or small macros, inline functions and templates
(ten or fewer lines in length), you do both of the following:

   a) Give prominent notice with each copy of the object code that the
   Library is used in it and that the Library and its use are
   covered by this License.

   b) Accompany the object code with a copy of the GNU GPL and this license
   document.

  4. Combined Works.

  You may convey a Combined Work under terms of your choice that,
taken together, effectively do not restrict modification of the
portions of the Library contained in the Combined Work and reverse
engineering for debugging such modifications, if you also do each of
the following:

   a) Give prominent notice with each copy of the Combined Work that
   the Library is used in it and that the Library and its use are
   covered by this License.

   b) Accompany the Combined Work with a copy of the GNU GPL and this license
   document.

   c) For a Combined Work that displays copyright notices during
   execution, include the copyright notice for the Library among
   these notices, as well as a reference directing the user to the
   copies of the GNU GPL and this license document.

   d) Do one of the following:

       0) Convey the Minimal Corresponding Source under the terms of this
       License, and the Corresponding Application Code in a form
       suitable for, and under terms that permit, the user to
       recombine or relink the Application with a modified version of
       the Linked Version to produce a modified Combined Work, in the
       manner specified by section 6 of the GNU GPL for conveying
       Corresponding Source.

       1) Use a suitable shared library mechanism for linking with the
       Library.  A suitable mechanism is one that (a) uses at run time
       a copy of the Library already present on the user's computer
       system, and (b) will operate properly with a modified version
       of the Library that is interface-compatible with the Linked
       Version.

   e) Provide Installation Information, but only if you would otherwise
   be required to provide such information under section 6 of the
   GNU GPL, and only to the extent that such information is
   necessary to install and execute a modified version of the
   Combined Work produced by recombining or relinking the
   Application with a modified version of the Linked Version. (If
   you use option 4d0, the Installation Information must accompany
   the Minimal Corresponding Source and Corresponding Application
   Code. If you use option 4d1, you must provide the Installation
   Information in the manner specified by section 6 of the GNU GPL
   for conveying Corresponding Source.)

  5. Combined Libraries.

  You may place library facilities that are a work based on the
Library side by side in a single library together with other library
facilities that are not Applications and are not covered by this
License, and convey such a combined library under terms of your
choice, if you do both of the following:

   a) Accompany the combined library with a copy of the same work based
   on the Library, uncombined with any other library facilities,
   conveyed under the terms of this License.

   b) Give prominent notice with the combined library that part of it
   is a work based on the Library, and explaining where to find the
   accompanying uncombined form of the same work.

  6. Revised Versions of the GNU Lesser General Public License.

  The Free Software Foundation may publish revised and/or new versions
of the GNU Lesser General Public License from time to time. Such new
versions will be similar in spirit to the present version, but may
differ in detail to address new problems or concerns.

  Each version is given a distinguishing version number. If the
Library as you received it specifies that a certain numbered version
of the GNU Lesser General Public License "or any later version"
applies to it, you have the option of following the terms and
conditions either of that published version or of any later version
published by the Free Software Foundation. If the Library as you
received it does not specify a version number of the GNU Lesser
General Public License, you may choose any version of the GNU Lesser
General Public License ever published by the Free Software Foundation.

  If the Library as you received it specifies that a proxy can decide
whether future versions of the GNU Lesser General Public License shall
apply, that proxy's public statement of acceptance of any version is
permanent authorization for you to choose that version for the
Library.
