import numpy as np
import ctypes as ct
import cgen

from pdb import set_trace
from sympy import Mod
from functools import reduce
from collections import OrderedDict

from devito.tools import timed_pass
from devito.symbolics import (CondEq, CondNe, Macro, String)
from devito.symbolics.extended_sympy import (FieldFromPointer, Byref)
from devito.types import CustomDimension, Array, Symbol, Pointer, FILE, Timer, NThreads, TimeDimension
from devito.ir.iet import (Expression, Increment, Iteration, List, Conditional, SyncSpot,
                           Section, HaloSpot, ExpressionBundle, Call, Conditional, CallableBody, 
                           Callable, FindSymbols, FindNodes, Transformer, Return)
from devito.ir.equations import IREq, ClusterizedEq
from devito.ir.support import (Interval, IntervalGroup, IterationSpace, Backward, PARALLEL, AFFINE)

__all__ = ['iet_build']


@timed_pass(name='build')
def iet_build(stree, **kwargs):
    """
    Construct an Iteration/Expression tree(IET) from a ScheduleTree.
    """

    ooc = kwargs['options']['out-of-core']
    is_mpi = kwargs['options']['mpi']
    
    nsections = 0
    queues = OrderedDict()
    for i in stree.visit():
        if i == stree:
            # We hit this handle at the very end of the visit
            iet_body = queues.pop(i)
            if(ooc):
                iet_body = _ooc_build(iet_body, kwargs['sregistry'].nthreads, kwargs['profiler'], ooc.function, ooc.mode, is_mpi)
                return List(body=iet_body)
            else:                
                return List(body=iet_body)

        elif i.is_Exprs:
            exprs = []
            for e in i.exprs:
                if e.is_Increment:
                    exprs.append(Increment(e))
                else:
                    exprs.append(Expression(e, operation=e.operation))
            body = ExpressionBundle(i.ispace, i.ops, i.traffic, body=exprs)

        elif i.is_Conditional:
            body = Conditional(i.guard, queues.pop(i))

        elif i.is_Iteration:
            iteration_nodes = [queues.pop(i)]
            if isinstance(i.dim, TimeDimension) and ooc and ooc.mode == 'forward':
                iteration_nodes.append(Section("write_temp"))
            elif isinstance(i.dim, TimeDimension) and ooc and ooc.mode == 'gradient':
                iteration_nodes.append(Section("read_temp"))

            body = Iteration(iteration_nodes, i.dim, i.limits, direction=i.direction,
                             properties=i.properties, uindices=i.sub_iterators)

        elif i.is_Section:
            body = Section('section%d' % nsections, body=queues.pop(i))
            nsections += 1

        elif i.is_Halo:
            body = HaloSpot(queues.pop(i), i.halo_scheme)

        elif i.is_Sync:
            body = SyncSpot(i.sync_ops, body=queues.pop(i, None))

        queues.setdefault(i.parent, []).append(body)

    assert False


@timed_pass(name='ooc_build')
def _ooc_build(iet_body, nthreads, profiler, func, out_of_core, is_mpi):
    """
    This private method builds a iet_body (list) with out-of-core nodes.

    Args:
        iet_body (List): a list of nodes
        nthreads (NThreads): an object of devito threads
        profiler (Profiler): a devito profiler
        func (Function): the function
        out_of_core (string): 'forward' or 'gradient'
        is_mpi (bool): flag of MPI execution

    Returns:
        List : iet_body is a list of nodes
    """
    
    # Creates nthreads once again in order to enable the ignoreDefinition flag
    #nthreads = NThreads(ignoreDefinition=True)
    
    is_forward = out_of_core == 'forward'

    ######## Dimension and symbol for iteration spaces ########
    nthreadsDim = CustomDimension(name="i", symbolic_size=nthreads)    
    iSymbol = Symbol(name="i", dtype=np.int32)


    ######## Build files and counters arrays ########
    filesArray = Array(name='files', dimensions=[nthreadsDim], dtype=np.int32)
    countersArray = Array(name='counters', dimensions=[nthreadsDim], dtype=np.int32)


    ######## Build open section ########
    openSection = open_build(filesArray, countersArray, nthreadsDim, nthreads, is_forward, iSymbol)
    

    ######## Build func_size var ########
    func_name = func.name
    func_size = Symbol(name=func_name+"_size", dtype=np.uint64) 
    funcSizeExp, floatSizeInit = func_size_build(func, func_size)

    ######## Build write/read section ########
    dims = FindSymbols("dimensions").visit(iet_body)
    t0 = next((dim for dim in dims if dim.name == "t0"), None)
    write_or_read_build(iet_body, is_forward, nthreads, filesArray, iSymbol, func_size, func, t0, countersArray, is_mpi)
    

    ######## Build close section ########
    closeSection = close_build(nthreads, filesArray, iSymbol, nthreadsDim)
    

    ######## Build write_size var ########
    size_name = 'write_size' if is_forward else 'read_size'
    ioSize = Symbol(name=size_name, dtype=np.int64)
    ioSizeExp = io_size_build(ioSize, func_size, func)
    

    ######## Build save call ########
    timerProfiler = Timer(profiler.name, [], ignoreDefinition=True)
    saveCall = Call(name='save_temp', arguments=[nthreads, timerProfiler, ioSize])
    
    #TODO: Generate blank lines between sections
    iet_body.insert(0, funcSizeExp)
    iet_body.insert(0, floatSizeInit)
    iet_body.insert(0, openSection)
    iet_body.append(closeSection)
    iet_body.append(ioSizeExp)
    iet_body.append(saveCall)

    return iet_body

def open_build(filesArray, countersArray, nthreadsDim, nthreads, is_forward, iSymbol):
    """
    This method inteds to code open section for both Forward and Gradient operators.
    
    Args:
        filesArray (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        countersArray (counters): pointer of allocated memory of nthreads dimension. Each place has a size of int
        nthreadsDim (CustomDimension): dimension from 0 to nthreads 
        nthreads (NThreads): number of threads
        is_forward (bool): True for the Forward operator; False for the Gradient operator

    Returns:
        Section: open section
    """
    
    # Test files array and exit if get wrong
    filesArrCond = array_alloc_check(filesArray) #  Forward
    
    #Call open_thread_files
    open_thread_call = Call(name='open_thread_files_temp', arguments=[filesArray, nthreads])

    # Open section body
    body = [filesArrCond, open_thread_call]
    
    if not is_forward:
        countersArrCond = array_alloc_check(countersArray) # gradient
        body.append(countersArrCond)
        
        intervalGroup = IntervalGroup((Interval(nthreadsDim, 0, nthreads)))
        cNewCountersEq = ClusterizedEq(IREq(countersArray[iSymbol], 1), ispace=IterationSpace(intervalGroup))
        openIterationGrad = Iteration(Expression(cNewCountersEq, None, False), nthreadsDim, nthreads-1)
        body.append(openIterationGrad)
        
    return Section("open", body)

def write_build(nthreads, filesArray, iSymbol, func_size, funcStencil, t0, uVecSize1, is_mpi):
    """
    This method inteds to code gradient.c write section.
    Obs: maybe the desciption of the variables should be better    

    Args:
        nthreads (NThreads): symbol of number of threads
        filesArray (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        iSymbol (Symbol): symbol of the iterator index i
        func_size (Symbol): the funcStencil size
        funcStencil (u): a stencil we call u
        t0 (ModuloDimension): time t0
        uVecSize1 (FieldFromPointer): size of a vector u

    Returns:
        Section: complete wrie section
    """
    
    uSizeDim = CustomDimension(name="i", symbolic_size=uVecSize1)
    interval = Interval(uSizeDim, 0, uVecSize1)
    intervalGroup = IntervalGroup((interval))
    ispace = IterationSpace(intervalGroup)
    itNodes = []

    tid = Symbol(name="tid", dtype=np.int32)
    tidEq = IREq(tid, Mod(iSymbol, nthreads))
    cTidEq = ClusterizedEq(tidEq, ispace=ispace)
    itNodes.append(Expression(cTidEq, None, True))
    
    ret = Symbol(name="ret", dtype=np.int32)
    writeCall = Call(name="write", arguments=[filesArray[tid], funcStencil[t0, iSymbol], func_size], retobj=ret)
    itNodes.append(writeCall)
    
    pstring = String("\"Write size mismatch with function slice size\"")

    condNodes = [Call(name="perror", arguments=pstring)]
    condNodes.append(Call(name="exit", arguments=1))
    cond = Conditional(CondNe(ret, func_size), condNodes)
    itNodes.append(cond)

    # TODO: Pragmas should depend on the user's selected optimization options and be generated by the compiler
    if is_mpi:
        pragma = cgen.Pragma("omp parallel for schedule(static,1)")
    else:
        pragma = cgen.Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
    writeIteration = Iteration(itNodes, uSizeDim, uVecSize1-1, pragmas=[pragma])

    return Section("write", writeIteration)

def read_build(nthreads, filesArray, iSymbol, func_size, funcStencil, t0, uVecSize1, counters):
    """
    This method inteds to code gradient.c read section.
    Obs: maybe the desciption of the variables should be better    

    Args:
        nthreads (NThreads): symbol of number of threads
        filesArray (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        iSymbol (Symbol): symbol of the iterator index i
        func_size (Symbol): the funcStencil size
        funcStencil (u): a stencil we call u
        t0 (ModuloDimension): time t0
        uVecSize1 (FieldFromPointer): size of a vector u
        counters (array): pointer of allocated memory of nthreads dimension. Each place has a size of int

    Returns:
        section (Section): complete read section
    """
    
    #  pragma omp parallel for schedule(static,1) num_threads(nthreads)
    #  0 <= i <= u_vec->size[1]-1
    #  TODO: Pragmas should depend on user's selected optimization options and generated by the compiler
    pragma = cgen.Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
    iDim = CustomDimension(name="i", symbolic_size=uVecSize1)
    interval = Interval(iDim, 0, uVecSize1)
    intervalGroup = IntervalGroup((interval))
    ispace = IterationSpace(intervalGroup)
    itNodes = []

    # int tid = i%nthreads;
    tid = Symbol(name="tid", dtype=np.int32)
    tidEq = IREq(tid, Mod(iSymbol, nthreads))
    cTidEq = ClusterizedEq(tidEq, ispace=ispace)
    itNodes.append(Expression(cTidEq, None, True))
    
    # off_t offset = counters[tid] * func_size;
    # lseek(files[tid], -1 * offset, SEEK_END);
    # TODO: make offset be a off_t
    offset = Symbol(name="offset", dtype=np.int32)
    offsetEq = IREq(offset, (-1)*counters[tid]*func_size)
    cOffsetEq = ClusterizedEq(offsetEq, ispace=ispace)
    itNodes.append(Expression(cOffsetEq, None, True))    
    itNodes.append(Call(name="lseek", arguments=[filesArray[tid], offset, Macro("SEEK_END")]))

    # int ret = read(files[tid], u[t0][i], func_size);
    ret = Symbol(name="ret", dtype=np.int32)
    readCall = Call(name="read", arguments=[filesArray[tid], funcStencil[t0, iSymbol], func_size], retobj=ret)
    itNodes.append(readCall)

    # printf("%d", ret);
    # perror("Cannot open output file");
    # exit(1);
    pret = String("'%d', ret")
    pstring = String("\"Cannot open output file\"")
    condNodes = [
        Call(name="printf", arguments=pret),
        Call(name="perror", arguments=pstring), 
        Call(name="exit", arguments=1)
    ]
    cond = Conditional(CondNe(ret, func_size), condNodes) # if (ret != func_size)
    itNodes.append(cond)
    
    # counters[tid] = counters[tid] + 1
    newCountersEq = IREq(counters[tid], 1)
    cNewCountersEq = ClusterizedEq(newCountersEq, ispace=ispace)
    itNodes.append(Increment(cNewCountersEq))
        
    readIteration = Iteration(itNodes, iDim, uVecSize1-1, direction=Backward, pragmas=[pragma])
    
    section = Section("read", readIteration)

    return section

def close_build(nthreads, filesArray, iSymbol, nthreadsDim):
    """
    This method inteds to code gradient.c close section.
    Obs: maybe the desciption of the variables should be better

    Args:
        nthreads (NThreads): symbol of number of threads
        filesArray (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        iSymbol (Symbol): symbol of the iterator index i
        nthreadsDim (CustomDimension): dimension from 0 to nthreads

    Returns:
        section (Section): complete close section
    """

    # close(files[i]);
    itNode = Call(name="close", arguments=[filesArray[iSymbol]])    
    
    # for(int i=0; i < nthreads; i++) --> for(int i=0; i <= nthreads-1; i+=1)
    closeIteration = Iteration(itNode, nthreadsDim, nthreads-1)
    
    section = Section("close", closeIteration)
    
    return section

def array_alloc_check(array):
    """
    Checks wether malloc worked for array allocation.

    Args:
        array (Array): array (files or counters)

    Returns:
        Conditional: condition to handle allocated array
    """
    
    pstring = String("\"Error to alloc\"")
    printfCall = Call(name="printf", arguments=pstring)
    exitCall = Call(name="exit", arguments=1)
    return Conditional(CondEq(array, Macro('NULL')), [printfCall, exitCall])
    

def write_or_read_build(iet_body, is_forward, nthreads, filesArray, iSymbol, func_size, funcStencil, t0, countersArray, is_mpi):
    """
    Builds the read or write section of the operator, depending on the out_of_core mode.
    Replaces the temporary section at the end of the time iteration by the read or write section.   

    Args:
        iet_body (List): list of IET nodes 
        is_forward (bool): True for the Forward operator; False for the Gradient operator
        nthreads (NThreads): symbol of number of threads
        filesArray (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        iSymbol (Symbol): symbol of the iterator index i
        func_size (Symbol): the funcStencil size
        funcStencil (u): a stencil we call u
        t0 (ModuloDimension): time t0
        countersArray (array): pointer of allocated memory of nthreads dimension. Each place has a size of int

    """

    if is_forward:
        ooc_section = write_build(nthreads, filesArray, iSymbol, func_size, funcStencil, t0, funcStencil.symbolic_shape[1], is_mpi)
        temp_name = 'write_temp'
    else: # gradient
        ooc_section = read_build(nthreads, filesArray, iSymbol, func_size, funcStencil, t0, funcStencil.symbolic_shape[1], countersArray)
        temp_name = 'read_temp'  

    sections = FindNodes(Section).visit(iet_body)
    temp_sec = next((section for section in sections if section.name == temp_name), None)
    mapper={temp_sec: ooc_section}

    timeIndex = next((i for i, node in enumerate(iet_body) if isinstance(node, Iteration) and isinstance(node.dim, TimeDimension)), None)
    transformedIet = Transformer(mapper).visit(iet_body[timeIndex])
    iet_body[timeIndex] = transformedIet


def func_size_build(funcStencil, func_size):
    """
    Generates float_size init call and the init function size expression.

    Args:
        funcStencil (AbstractFunction): I/O function
        func_size (Symbol): Symbol representing the I/O function size

    Returns:
        funcSizeExp: Expression initializing the function size
        floatSizeInit: Call initializing float_size
    """

    floatSize = Symbol(name="float_size", dtype=np.uint64)
    floatString = String(r"float")
    floatSizeInit = Call(name="sizeof", arguments=[floatString], retobj=floatSize)
    
    # TODO: Function name must come from user?
    sizes = funcStencil.symbolic_shape[2:]
    funcEq = IREq(func_size, (reduce(lambda x, y: x * y, sizes) * floatSize))
    funcSizeExp = Expression(ClusterizedEq(funcEq, ispace=None), None, True)

    return funcSizeExp, floatSizeInit

def io_size_build(ioSize, func_size, funcStencil):
    """
    Generates init expression calculating io_size.

    Args:
        ioSize (Symbol): Symbol representing the total amount of I/O data
        func_size (Symbol): Symbol representing the I/O function size

    Returns:
        funcSizeExp: Expression initializing ioSize
    """

    #TODO: Time limits must be retrieved
    # time_M = Symbol(name="time_M", dtype=np.int32)
    # time_m = Symbol(name="time_m", dtype=np.int32)
    time_M = funcStencil.time_dim.symbolic_max
    time_m = funcStencil.time_dim.symbolic_min
    #TODO: Field and pointer must be retrieved from somewhere
    funcSize1 = FieldFromPointer(funcStencil.dimensions[1], funcStencil._C_name)
    
    ioSizeEq = IREq(ioSize, ((time_M - time_m+1) * funcSize1 * func_size))

    return Expression(ClusterizedEq(ioSizeEq, ispace=None), None, True)
