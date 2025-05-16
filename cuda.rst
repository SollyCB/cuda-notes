Kernels
=======

Kernels are executed by blocks of threads which look like wavefronts. A set of blocks is a grid.
Blocks can be grouped into clusters after compute 9.

Launching a kernel looks like this

.. code:: C
  :number-lines:

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
language such as C++" - LOL, "don't write PTX yourself, just leave it to the compiler".

Compilation
===========

Interesting: NVCC "modifies the host code" replacing ``<<<...>>>`` with cuda runtime function calls for
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
declared as

.. code:: C
  :number-lines:

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

Host Memory
===========

Page-Locked (Pinned)
--------------------

``cudaHostAlloc``, ``cudaFreehost``, ``cudaHostRegister``

Facilitates mapping ranges into the device's address space, removing the need for copies, and
can increase bandwidth (although this last point seems irrelevant since it is specific to a
front-side bus, but this seems old as shit? [#]_). Also

  Copies between page-locked host memory and device memory can be performed concurrently with kernel
  execution for some devices as mentioned in.

which I don't quite get: I don't know why pinning is requirement here. Maybe because the kernel can
execute since it doesn't have to worry about the memory not being there?

Note that the benefits above are only available by default to the device that was current when the
pinned memory was allocated. In order to apply the benefits to all devices,
``cudaHostAllocPortable`` must be specified.

Performance of pinned memory can be further improved with ``cudaHostAllocWriteCombined`` (as long as
the host *only ever writes* to this memory).

.. [#] "The front-side bus was used in all Intel Atom, Celeron, Pentium, Core 2, and Xeon processor
   models through about 2008 and was eliminated in 2009" -
   https://en.wikipedia.org/wiki/Front-side_bus#Evolution

Mapped
------

Memory mapping works as expected (basically the same as Vulkan).

Domains
=======

These facilitate narrowing synchronisation scopes.

In the case

.. code:: C
  :number-lines:

  __managed__ int x = 0;
  __device__  cuda::atomic<int, cuda::thread_scope_device> a(0);
  __managed__ cuda::atomic<int, cuda::thread_scope_system> b(0);

  /* Thread 1 (SM) */

  x = 1;
  a = 1;

  /* Thread 2 (SM) */

  while (a != 1) ;
  assert(x == 1);
  b = 1;

  /* Thread 3 (CPU) */

  while (b != 1) ;
  assert(x == 1);

the asserts are true due to memory ordering ensuring that the write to ``x`` is visible before the
the write to ``a``. However, this can lead to inefficiencies where the GPU cannot flush its writes
until it can be sure that it has waited for other writes, as they may be a part of the sync scope of
the atomic store.

Using domains, when kernels are launched, they are tagged with an ID, and fence operations will only
be ordered against those kernels who are tagged with the ID matching the fence's domain. As such, it
is insufficient to use ``thread_scope_device`` to order operations between kernels outside of a
fence's doamin: ``thread_scope_system`` must be used instead. While this changes the definition of
``thread_scope_device``, kernels will default to ID 0, so backwards compatibility is not broken.

Using Domains
-------------

+-----------------------------------------+-------------------------------------------+
| ``cudaLaunchAttributeMemSyncDomain``    | Select between remote and default domains |
+-----------------------------------------+-------------------------------------------+
| ``cudaLaunchAttributeMemSyncDomainMap`` | Map logical to physical domains           |
+-----------------------------------------+-------------------------------------------+
| ``cudaLaunchMemSyncDomainDefault``      | Default domain                            |
+-----------------------------------------+-------------------------------------------+
| ``cudaLaunchMemSyncDomainRemote``       | Isolate remote memory traffic from local  |
+-----------------------------------------+-------------------------------------------+

``cudaLaunchMemSyncDomainDefault`` and ``cudaLaunchMemSyncDomainRemote`` are logical domains. They
allow, for instance, a library to logically separate its kernels without having to consider the
environment that might be going on around it. Then user code can map logical domains to physical
domains in order to manage how the separation actually occurs. For instance, the user might have two
different streams, and he separates out these streams using physical domains; then the library code
getting called further down the stack only knows that it has separated out its kernels, while the
user knows that the way the work is being managed at a higher level is distinct.

There are 4 physical domains on Hopper (compute 9, cuda 12), older arches will just always report 1
from ``cudaDevAttrMemSyncDomainCount``, so portable code will just always map kernels to the same
physical domain.

Async Concurrent Execution
==========================

Independent tasks which can operate concurrently:

- Computation on the host;
- Computation on the device;
- Memory transfers from the host to the device;
- Memory transfers from the device to the host;
- Memory transfers within the memory of a given device;
- Memory transfers among devices.

Operations which can be launched from the host, with control returned to the host before the
operation has completed:

- Kernel launches;
- Memory copies within a single device’s memory;
- Memory copies from host to device of a memory block of 64 KB or less;
- Memory copies performed by functions that are suffixed with ``Async``;
- Memory set function calls.

Note that:

- **``Async`` memory copies might also be synchronous if they involve host memory that is not
  page-locked.**
- Kernel launches are synchronous if hardware counters are collected via a profiler (Nsight, Visual
  Profiler) unless concurrent kernel profiling is enabled.

Concurrent Kernels
------------------

Supported at 2.x and above, but:

  A kernel from one CUDA context cannot execute concurrently with a kernel from another CUDA context.
  The GPU may time slice to provide forward progress to each context. If a user wants to run kernels
  from multiple process simultaneously on the SM, one must enable MPS.

Also kernels with lots of memory are less likely to run concurrently (intuitive).

Memory copies can happen async with kernel execution, resembling Vulkan dedicated transfer queues.

Memory download and upload can also be overlapped, but involved host memory must be pinned.

Streams
=======

Streams are just Vulkan command buffers: you submit them in sequence, but they can execute
concurrently, out of order with eachother, etc. Commands start executing when their dependencies are
met, which can be within stream or cross stream. Work on a stream can overlap according the rules
described above.

Calling ``cudaStreamDestroy`` while the device is still chewing through it will cause the function
to immediately return with the stream's resources being cleaned up automatically later.

Default Stream
--------------

Not specifying a stream or passing 0 will use the default stream. This doesn't seem any different
just basically using a single command buffer for all your shit, but I might wrong because

  For code that is compiled using the --default-stream per-thread compilation flag (or that defines
  the CUDA_API_PER_THREAD_DEFAULT_STREAM macro before including CUDA headers (cuda.h and
  cuda_runtime.h)), the default stream is a regular stream and each host thread has its own default
  stream.

which could imply that the default stream otherwise is not regular? But an earlier quote

  Kernel launches... are issued to the default stream. They are therefore executed in order.

in using 'therefore' implies that the default stream without the aforementioned switches is still a
regular stream, and the "executed in order" only refers to the fact that work in a stream is
initiated in the order that it appears in the stream, but does not necessarily complete in the order
in which it was submitted.

I am going with "the default stream is a regular stream, and per-thread default streams are also
just streams, but they are used when a stream is not specified per-thread, not globally".

If code is compiled without specifying a ``--default-stream``, ``--default-stream legacy`` is
assumed, which causes each device to have a single *NULL stream*, shared by all host threads, which
has implicit synchronisation (see below).

Synchronisation
---------------

Explicit
^^^^^^^^

- ``cudaDeviceSynchronize``
  Block host until all streams in all threads have completed.
- ``cudaStreamSynchronize``
  Block host until given stream has completed.
- ``cudaStreamWaitEvent``
  Like a hardcore, zero granularity pipeline barrier: all commands in the stream after this call
  must wait for all commands before the call to complete.
- ``cudaStreamQuery``
  Ask if preceding commands in a stream have completed.

Implicit
^^^^^^^^

The NULL stream causes total stream sync:

  Two operations from different streams cannot run concurrently if any CUDA operation on the NULL
  stream is submitted in-between them, unless the streams are non-blocking streams (created with the
  cudaStreamNonBlocking flag).

So don't mix async stream submissions and NULL stream submissions, is the very obvious tip that the
docs give following this quote.

Host Callbacks
--------------

Host functions can be inserted into a stream and will run once commands preceding it in the stream
have completed. Commands later in the stream do not execute until the host function has returned.

Priority
--------

Streams can be given a priority which hints the GPU about what to schedule first. Stream priority
does not provide any ordering guarantees and cannot preempt or interrupt work.

Programmatic Dependent Launch
=============================

A fancy way of saying 'Vulkan pipeline barriers': it allows a kernel to begin execution before its
dependencies have completed if the kernel has work that it can do that is not dependent (like how
Vulkan pipeline barriers allow you to wait on specific stages, as opposed to having to wait for an
entire pipeline).

This is achieved via ``cudaTriggerProgrammaticLaunchCompletion`` and
``cudaGridDependencySynchronize``, where the latter is called on a dependent kernel, and blocks
until it sees the former, which will be called in the earlier kernel once it has completed all the
work that the later kernel actually depends on (the call itself is a flush). If the earlier kernel
does not call the explicit signal, it is implicitly called when the kernel completes.

Concurrency is not guaranteed, only being applied opportunistically.

Use with graphs
---------------

+---------------------------------------------------------------------+---------------------------------------------------------------------+
| Stream Code                                                         | Graph Edge                                                          |
+=====================================================================+=====================================================================+
| | cudaLaunchAttribute attribute;                                    | | cudaGraphEdgeData edgeData;                                       |
| | attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;| | edgeData.type = cudaGraphDependencyTypeProgrammatic;              |
| | attribute.val.programmaticStreamSerializationAllowed = 1;         | | edgeData.from_port = cudaGraphKernelNodePortProgrammatic;         |
+---------------------------------------------------------------------+---------------------------------------------------------------------+
| | cudaLaunchAttribute attribute;                                    | | cudaGraphEdgeData edgeData;                                       |
| | attribute.id = cudaLaunchAttributeProgrammaticEvent;              | | edgeData.type = cudaGraphDependencyTypeProgrammatic;              |
| | attribute.val.programmaticEvent.triggerAtBlockStart = 0;          | | edgeData.from_port = cudaGraphKernelNodePortProgrammatic;         |
+---------------------------------------------------------------------+---------------------------------------------------------------------+
| | cudaLaunchAttribute attribute;                                    | | cudaGraphEdgeData edgeData;                                       |
| | attribute.id = cudaLaunchAttributeProgrammaticEvent;              | | edgeData.type = cudaGraphDependencyTypeProgrammatic;              |
| | attribute.val.programmaticEvent.triggerAtBlockStart = 1;          | | edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion;     |
+---------------------------------------------------------------------+---------------------------------------------------------------------+

Graphs
======

Resemble Vulkan subpasses, where you program in the depedency edges, and the driver inserts in the
synchronisation, whereas normally in Vulkan you are both defining the depedency edges and inserting
the synchronisation yourself.

The rationale behind graphs is that when submitting a kernel on a stream, the driver has to do a
bunch of setup for that kernel without much of the context about how it fits into the broader
workflow. In this way, one cannot consider Vulkan command buffers as CUDA streams, because the
Vulkan driver needn't do any of this same setup: a command buffer in Vulkan is low-level enough that
you are able to describe the graph yourself, the driver just passes the instructions to the GPU for
chewing, since all of the setup is on you.

With a CUDA graph, the driver still has to do all the work for you, but it has more information with
which it can reason about the work. Graph workflow is also in three stages, the second of which is
bake/compilation, meaning that the driver doesn't have to keep doing setup work over and over, since
it does the work once, and then that work is reusable.

The three stages are BS: definition, compilation, launching. It is just Vulkan command buffer, but
the driver makes it for you: a resusable set of work that can be passed to the GPU with less driver
overhead.

Nodes
-----

A node on a graph is scheduling any time after its dependencies are met.

A node is any of the following operations:

- kernel
- CPU function call
- memory copy
- memset
- empty node
- waiting on an event
- recording an event
- signalling an external semaphore
- waiting on an external semaphore
- conditional node
- child graph

Edge Data
---------

This is exactly Vulkan pipeline dependencies: edge data is defined by an outgoing port, an incoming
port, and a type. This is just Vulkan execution scopes and how they are grouped: like a memory copy
could map be something like a buffer upload waited on by a vertex shader:

+-----------+-------------------------+---------------------------------------------------------------------+
| CUDA Name | Vulkan Equivalent Name  | Vulkan Data Value                                                   |
+===========+=========================+=====================================================================+
|  type     | VkAccessFlags           | VK_ACCESS_MEMORY_WRITE_BIT                                          |
+-----------+-------------------------+---------------------------------------------------------------------+
|  outgoing | VkPipelineStageFlagBits | VK_PIPELINE_STAGE_2_TRANSFER_BIT                                    |
+-----------+-------------------------+---------------------------------------------------------------------+
|  incoming | VkPipelineStageFlagBits | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT                                |
+-----------+-------------------------+---------------------------------------------------------------------+

Where the 'ports' are Vulkan 'synchronisation scopes', and the 'type' defines the access scope [#]_
(although I am not sure what direction incoming and outgoing are, as it depends on how you consider
the direction that the edges are pointing in).

.. [#] https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#synchronization-dependencies

Edge Data From Stream Capture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. TODO: Come back to this with more info

There is also some weirdness to do with getting the edge data using stream capture API which seems
to have some potential gotchas to do with edges that do not wait for full completion (this section
will be expanded when I have more info, which I assume I will get once I read the stream capture
section).

Graph API
---------

Creating a graph with the api seems trivial and intuitive:

.. code:: C
  :number-lines:

  cudaGraphCreate(&graph, 0);
  cudaGraphAddKernelNode(&a, graph, 0, 0, &node_info);
  cudaGraphAddKernelNode(&b, graph, 0, 0, &node_info);
  cudaGraphAddDependencies(graph, &a, &b, 1); // A->B

Stream Capture
--------------

Stream capture is literally Vulkan command buffers: calling ``cudaStreamBeginCapture`` before
enqueueing work to a stream puts the it in a recording mode which builds an internal graph. This
resembles the Vulkan command buffer lifecycle (record, then submit, as opposed to typical cuda
streams which are actually streaming work as it is put in the stream). Calling
``cudaStreamEndCapture`` is the ``vkEndCommandBuffer``

Any stream can be captured except the NULL stream.

Use ``cudaStreamBeginCaptureToGraph`` to use a user declared graph rather than an internal one.

Captured Events
---------------

If waiting on an event in a captured stream, that event must have been recorded into the same
capture graph (best clarified by the code example below).

If another stream waits on an event which was recorded in a captured stream, that stream becomes a
captured stream, and is now a part of the other captured stream's graph.

An event recorded on a captured stream (the docs call this a *captured event*) can be seen as
representing some set of nodes in the graph. So when another stream waits on a captured event, it is
waiting on that set of nodes.

When other streams become a part of a captured graph, ``cudaStreamEndCapture`` must still be called
on the original stream (docs call this the *origin stream*).

Other streams must be joined with the origin stream before capture is ended (this means that the
origin stream must wait for other streams to complete - see the below code example).

.. code:: C
  :number-lines:

  // stream1 is the origin stream
  cudaStreamBeginCapture(stream1);

  kernel_A<<< ..., stream1 >>>(...);

  // Event is captured by stream1's graph
  cudaEventRecord(event1, stream1);

  // stream2 enters the graph
  cudaStreamWaitEvent(stream2, event1);

  // kernel_B is synced with kernel_A according to the rules in Concurrent Kernels
  kernel_B<<< ..., stream1 >>>(...);

  // kernel_C will wait on kernel_A as event1 represents its completion
  kernel_C<<< ..., stream2 >>>(...);

  // Join stream1 and stream2, i.e. make stream1 wait for stream2 to idle before ending capture
  cudaEventRecord(event2, stream2);
  cudaStreamWaitEvent(stream1, event2);

  // More work can be done on stream1, stream2 is still idle
  kernel_D<<< ..., stream1 >>>(...);

  // End capture in the origin stream, since 
  cudaStreamEndCapture(stream1, &graph);

  // stream1 and stream2 no longer in capture mode

The resulting graph looks like:

.. code:: C
  :number-lines:

                                                 A
                                                / \
                                               v   v
                                               B   C
                                                \ /
                                                 v
                                                 D

Note that when a stream leaves capture mode, the first non-captured item has a dependency on the
most recent non-captured item. The captured items are dropped as dependencies as if they were not a
part of the stream. This is probably intuitive, since a captured stream is clearly nothing like what
a typical stream is: I think they just strapped the capturing on to streams because it makes graphs
easier to implement in existing code, despite graphs and streams being pretty disparate.

Illegal Operations
------------------

It is illegal to sync or query the execution status of a stream which is being captured, since no
execution is actually happening: a graph is just being built. This extends to handles which
encompass stream capture, like device and context handles.

Similarly, use of the legacy stream is invalid while a stream is being captured (if it was not
created with ``cudaStreamNonBlocking``) as such usage would require synchronisation with captured
streams. Synchronous APIs, like ``cudaMemcpy``, are also therefore invalid, since they use the
legacy stream.

A graph waiting on an event from

- another capture graph is illegal
- a stream that is not captured requires ``cudaEventWaitExternal``

Also

  A small number of APIs that enqueue asynchronous operations into streams are not currently supported
  in graphs and will return an error if called with a stream which is being captured, such as
  cudaStreamAttachMemAsync()

but I cannot see an exhaustive list documenting all exceptions.

When an illegal operations is performed on a stream that is being captured, further use of streams
or captured events associated with the capture graph is invalid until the capture is ended, which
will return an error and a NULL graph.

User Objects
------------

These are a way to to associate a destructor callback with a reference count that the graph can use
to know when to clean shit up.

This took a second to get completely at first just because the section is long, but it is literally
just what it says on the tin. The only thing to note is that a ref count owned by a child graph is
associated with the child, and not the parent, which is intuitive.

Graph Updates
-------------

Baked graphs can be updated as long as their topology does not change. This has the obvious
performance benefits that come with skipping all the checks and logic that must happen in order to
re-instantiate a graph.

Graph nodes can be updated individually, or an entire graph can be swapped with a topologically
equivalent one. The former is faster, since checks for topological equivalence and unaffected nodes
do not need to run, but is not always possible (e.g. the graph came from a library, so its topology
and node handles are are not available to the user) or so many nodes need to update that going node
by node is impractical.

Limitations
^^^^^^^^^^^

These are pretty intuitive if you consider the operations that a graph would encode.

I am just going to paste in the full section, as it is already information dense

.. code::
  :number-lines:

  Kernel nodes:

   - The owning context of the function cannot change.

   - A node whose function originally did not use CUDA dynamic parallelism cannot be updated to a
     function which uses CUDA dynamic parallelism.

  cudaMemset and cudaMemcpy nodes:

   - The CUDA device(s) to which the operand(s) was allocated/mapped cannot change.

   - The source/destination memory must be allocated from the same context as the original
   - source/destination memory.

   - Only 1D cudaMemset/cudaMemcpy nodes can be changed.

  Additional memcpy node restrictions:

   - Changing either the source or destination memory type (i.e., cudaPitchedPtr, cudaArray_t, etc.),
     or the type of transfer (i.e., cudaMemcpyKind) is not supported.

  External semaphore wait nodes and record nodes:

   - Changing the number of semaphores is not supported.

  Conditional nodes:

   - The order of handle creation and assignment must match between the graphs.

   - Changing node parameters is not supported (i.e. number of graphs in the conditional, node context,
     etc).

   - Changing parameters of nodes within the conditional body graph is subject to the rules above.


In order to do a full graph swap, the following rules apply (also just a copy paste)

.. code::
  :number-lines:

    1) For any capturing stream, the API calls operating on that stream must be made in the same order,
       including event wait and other api calls not directly corresponding to node creation.

    2) The API calls which directly manipulate a given graph node’s incoming edges (including captured
       stream APIs, node add APIs, and edge addition / removal APIs) must be made in the same order.
       Moreover, when dependencies are specified in arrays to these APIs, the order in which the
       dependencies are specified inside those arrays must match.

    3) Sink nodes must be consistently ordered. Sink nodes are nodes without dependent nodes / outgoing
       edges in the final graph at the time of the cudaGraphExecUpdate() invocation. The following
       operations affect sink node ordering (if present) and must (as a combined set) be made in the
       same order:

        - Node add APIs resulting in a sink node.

        - Edge removal resulting in a node becoming a sink node.

        - cudaStreamUpdateCaptureDependencies(), if it removes a sink node from a capturing stream’s
          dependency set.

        - cudaStreamEndCapture().

Updating individual nodes follows only the rules laid out earlier (at the beginning of
`Limitations`_). Each update type has its own dedicated API call

.. code::
  :number-lines:

  cudaGraphExecKernelNodeSetParams()
  cudaGraphExecMemcpyNodeSetParams()
  cudaGraphExecMemsetNodeSetParams()
  cudaGraphExecHostNodeSetParams()
  cudaGraphExecChildGraphNodeSetParams()
  cudaGraphExecEventRecordNodeSetEvent()
  cudaGraphExecEventWaitNodeSetEvent()
  cudaGraphExecExternalSemaphoresSignalNodeSetParams()
  cudaGraphExecExternalSemaphoresWaitNodeSetParams()

Individual nodes can also be enabled or disabled, enabling the creation of graphs which contain a
superset of some desired functionality, which can have nops swapped into in order to create the
exact subset of work that an app desires at any given time.

Usage Info
----------

``cudaGraph_t`` objects are not internally synchronized.

``cudaGraphExec_t`` objects cannot run concurrently with itself.

Graph execution happens in streams but for ordering only, creating no contraints on internal
parallelism on the graph or where its nodes can execute.

Device Graphs
-------------

On systems supporting unified addressing (discussed later), graphs can be launched from the device,
enabling data dependent decisions without the round trip from host to device.

There are limitations on such 'device graphs' that do not affect 'host graphs':

- Only graphs explicity created as device graphs can be launched from the device (as well as from
  the host).
- Device graphs cannot be launched simultaneously: such launches from the device will return an
  error code, such launches performed from the host and device simultaneously is undefined.

Instantiating device graphs also has requirements:

Meta Info
=========


Bookmark
--------
