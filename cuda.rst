Kernels
=======

Kernels are executed by blocks of threads which look like wavefronts. A set of blocks is a grid.
Blocks can be grouped into clusters after compute 9.

Launching a kernel looks like this.

::
  kern<<<nblocks, nthreads_per_block>>>

Defining the cluster setup for a kernel is compile time with ``__cluster_dims__``, or using the
``cudaLaunchKernel`` api.

Memory
======

Threads in a block can share memory ('shared memory'), threads in a cluster can share memory
('distributed shared memory'). Global memory is shared between all threads.

There is also texture and constant memory for specific uses, obviously. These, and global memory
are persistent across kernel launches (by the same app, obviously).

Unified memory provides 'managed memory' which is a single coherent memory image with a common
address space, which seems equivalent to Vulkan memory allocated from a heap with HOST_COHERENT and
HOST_VISIBLE flags, which you can access via a regular pointer.

Async
=====

A cuda threads is the lowest abstraction over computation and memory operations.

Async work is that which is initiated by a cuda thread, and executed asynchronously *as-if* by
another thread (unclear if this means that the work is always done on the initiating thread? Or if
the work could be handed to someone else? Unclear if this matters at all, or if people rely on
either of these cases).

Synchronisation of an async operation has the following scopes, which are intuitive:

+-----------------------------------------+---------------------------------------------------------------------------------------------+
| Thread scope                            | Description                                                                                 |
+=========================================+=============================================================================================+
| cuda::thread_scope::thread_scope_thread | Only the CUDA thread which initiated asynchronous operations synchronizes.                  |
+-----------------------------------------+---------------------------------------------------------------------------------------------+
| cuda::thread_scope::thread_scope_block  | All or any CUDA threads within the same thread block as the initiating thread synchronizes. |
+-----------------------------------------+---------------------------------------------------------------------------------------------+
| cuda::thread_scope::thread_scope_device | All or any CUDA threads in the same GPU device as the initiating thread synchronizes.       |
+-----------------------------------------+---------------------------------------------------------------------------------------------+
| cuda::thread_scope::thread_scope_system | All or any CUDA or CPU threads in the same system as the initiating thread synchronizes.    |
+-----------------------------------------+---------------------------------------------------------------------------------------------+

Compute Capability
==================

The names of the Nvidia arches and what 'compute capability' they map to.

+----------------------------------+---------------------------------+
| Major Revision Number            | NVIDIA GPU Architecture         |
+==================================+=================================+
| 9                                | NVIDIA Hopper GPU Architecture  |
+----------------------------------+---------------------------------+
| 8                                | NVIDIA Ampere GPU Architecture  |
+----------------------------------+---------------------------------+
| 7                                | NVIDIA Volta GPU Architecture   |
+----------------------------------+---------------------------------+
| 6                                | NVIDIA Pascal GPU Architecture  |
+----------------------------------+---------------------------------+
| 5                                | NVIDIA Maxwell GPU Architecture |
+----------------------------------+---------------------------------+
| 3                                | NVIDIA Kepler GPU Architecture  |
+----------------------------------+---------------------------------+

Some incremental thing that I am just noting for the completeness and pedanticness of it all.

+----------------------------------+--------------------------------+-------------------------------+
| Compute Capability               | NVIDIA GPU Architecture        | Based On                      |
+==================================+================================+===============================+
| 7.5                              | NVIDIA Turing GPU Architecture | NVIDIA Volta GPU Architecture |
+----------------------------------+--------------------------------+-------------------------------+

Compute capability is not the same as cuda version, although some cuda versions will stop supporting older arches.

Programming Interface
=====================

Runtime api allows allocating and deallocating device memory and launching kernels. The driver api
is a superset of the runtime, providing access to 'cuda contexts': an "analogue of host processes
for the device" (I guess this means - in the unix explanation - that a process is just a set of
resources that are being used by some progam); and cuda modules: dynamic libraries for the device
(intuitive).

PTX
===

"Kernels can be written using the CUDA instruction set architecture, called PTX, which is described
in the PTX reference manual. It is however usually more effective to use a high-level programming
language such as C++" - LOL, "don't use PTX, better to avoid it".

Compilation
===========

Interesting: NVCC "modifies the host code" replacing <<<...>>> with cuda runtime function calls for
loading and launching kernels. Looks like it removes this shit from the source code before handing
the remaining source code off to the host compiler.

  The modified host code is output either as C++ code that is left to be compiled using another tool
  or as object code directly by letting nvcc invoke the host compiler during the last compilation
  stage.

JIT
===

In cuda this refers to the device driver compiling PTX code loaded by the app at runtime into binary
code.

Ah, interesting: while this (obviously) increases load times, it means that an app compiled to PTX
code can run on future devices, and benefit from future compiler optimisations. That makes good
sense.

This compilation is cached and invalidated when the driver updates.

Binary Compat
=============

Controlled by the ``-code`` flag.

Binary compatibility is guaranteed forwards for minor versions, but not backwards, and not for major
releases. So a binary for ``8.5`` would work with ``8.6``, but not ``8.4``.

PTX Compat
==========

Controlled by the ``-arch`` flag.

The flag can take a compute capability (e.g. ``compute_50``), a specific arch (e.g. ``sm_90a``,
``compute_90a``), or a specific family (e.g. ``sm_100f``). Compute capability compilation is forward
compatible, arch specific is only compatible on the exact physical arch, and family specific runs on
the exact arch and arches in the same family.

App Compat
==========

The ``-gencode`` flag can be used to embed code for various architectures in the same binary, the
most appropriate of which is selected at runtime.

The ``__CUDA_ARCH__``, ``__CUDA_ARCH_FAMILY_SPECIFIC__`` and ``__CUDA_ARCH_SPECIFIC__`` macros can
be used to control source code compilation.

Initialization
==============

A context gets created for each device: these are the 'primary device contexts'. A context is shared
between all host application threads (like a Vulkan VkDevice it seems).

JIT'ing device code and loading it into device memory happens as a part of context creation.

A device's primary context can be accessed through the driver API.

``cudaDeviceReset()`` destroys the primary context of the current device, and the next runtime
call from any thread which has the same current device will result in the creation of a new primary
context for the device.

Device Memory
=============

Can be allocated either as linear memory, or cuda arrays, the latter of which are and opaque layout
optimized for texture fetches. Linear memory is allocated from a unified address space, so separate
allocations can reference eachother via pointers (so just the x64 contiguous block of virtual pages
type shit).

Per arch address spaces:

+------------------------------------------+----------------+-----------------+-----------------------+
|                                          | x86_64 (AMD64) | POWER (ppc64le) | ARM64                 |
+==========================================+================+=================+=======================+
| up to compute capability 5.3 (Maxwell)   | 40bit          | 40bit           | 40bit                 |
+------------------------------------------+----------------+-----------------+-----------------------+
| compute capability 6.0 (Pascal) or newer | up to 47bit    | up to 49bit     | up to 48bit           |
+------------------------------------------+----------------+-----------------+-----------------------+

``cudaMallocPitch`` and ``cudaMalloc3D`` ensure alignment requirements for 2D or 3D array memory
copies, improving performance.

``cudaMemcpy<To|From>Symbol`` facilitate the use of constant and global memory spaces, which are
declared as::

  __constant__ float const_data[N];
  __device__ float device_data[N];

``cudaGetSymbolAddress()`` and ``cudaGetSymbolSize()`` implement queries regarding global data.

L2 Memory Access
================

When accessing global data or cuda graph nodes, single accesses are considered "streamed", and
repeated access is considered persistent. The likelihood that such data can be cache resident can be
increased using the ``accessPolicyWindow`` struct in ``cudaStreamAttrValue`` and
``cudaKernelNodeAttrValue``. Some data range can have its likelihood have its chance of a cache hit
regulated by the `hitRatio`_ field.

Global memory accesses can also be controlled with ``cudaAccessPropertyStreaming`` and
``cudaAccessPropertyPersisting`` which inform how likely it is that an access will be repeated, or
individual.

.. _hitRatio: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-policy-for-persisting-accesses

If regulating the persistence of L2 cache lines, it is important to explicitly reset memory
persistence as cache lines may *continue to persist for a long time*.

Page-Locked (Pinned) Host Memory
================================

``cudaHostAlloc``, ``cudaFreehost``, ``cudaHostRegister``

Facilitates mapping ranges into the device's address space, removing the need for copies, and
can increase bandwidth (although this last point seems irrelevant since it is specific to a
front-side bus, but this seems old as shit?[#]_). Also

  Copies between page-locked host memory and device memory can be performed concurrently with kernel
  execution for some devices as mentioned in.

which I don't quite get: I don't know why pinning is requirement here. Maybe because the kernel can
execute since it doesn't have to worry about the memory not being there?

Note that the benefits above are only available by default to the device that was current when the
pinned memory was allocated. In order to apply the benefits to all devices,
``cudaHostAllocPortable`` must be specified.

Performance of pinned memory can be further improved with ``cudaHostAllocWriteCombined`` (as long as
the host *only ever writes* to this memory).

.. [#] "The front-side bus was used in all Intel Atom, Celeron, Pentium, Core 2, and Xeon processor models through about 2008 and was eliminated in 2009" - https://en.wikipedia.org/wiki/Front-side_bus#Evolution
