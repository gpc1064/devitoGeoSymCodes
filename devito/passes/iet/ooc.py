import numpy as np
from sympy import Mod

from devito.passes.iet.engine import iet_pass
from devito.symbolics import (CondEq, String)
from devito.types import CustomDimension, Array, Symbol
from devito.ir.iet import (Expression, Iteration, Conditional, Call, Conditional, CallableBody, Callable, FindNodes, Transformer)
from devito.ir.equations import IREq, ClusterizedEq

__all__ = ['ooc_efuncs']


def open_threads_build(nthreads, filesArray, iSymbol, iDim, is_forward):
    """
    This method generates the function open_thread_files according to the operator used.

    Args:
        nthreads (NThreads): number of threads
        filesArray (Array): array of files
        iSymbol (Symbol): symbol of the iterator index i 
        iDim (CustomDimension): dimension i from 0 to nthreads
        is_forward (bool): True for the Forward operator; False for the Gradient operator

    Returns:
        Callable: the callable function open_thread_files
    """
    
    nvme_id = Symbol(name="nvme_id", dtype=np.int32)
    ndisks = Symbol(name="NDISKS", dtype=np.int32)
    nvmeIdEq = IREq(nvme_id, Mod(iSymbol, ndisks))
    cNvmeIdEq = ClusterizedEq(nvmeIdEq, ispace=None) # ispace

    itNodes=[]
    itNodes.append(Expression(cNvmeIdEq, None, True))

    nameDim = [CustomDimension(name="nameDim", symbolic_size=100)]
    nameArray = Array(name='name', dimensions=nameDim, dtype=np.byte)

    pstring = String(r"'data/nvme%d/thread_%d.data'")
    itNodes.append(Call(name="sprintf", arguments=[nameArray, pstring, nvme_id, iSymbol]))
    
    if is_forward:
        pstring = String(r"'Creating file %s\n'")
    else:
        pstring = String(r"'Reading file %s\n'")
        
    itNodes.append(Call(name="printf", arguments=[pstring, nameArray]))

    ifNodes=[]
    pstring = String(r"'Cannot open output file\n'")
    ifNodes.append(Call(name="perror", arguments=pstring))

    ifNodes.append(Call(name="exit", arguments=1))

    opFlagsStr = String("OPEN_FLAGS")
    flagsStr = String("S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH")
    openCall = Call(name="open", arguments=[nameArray, opFlagsStr, flagsStr], retobj=filesArray[iSymbol])
    itNodes.append(openCall)

    openCond = Conditional(CondEq(filesArray[iSymbol], -1), ifNodes)
    
    itNodes.append(openCond)

    openIteration = Iteration(itNodes, iDim, nthreads-1)
    
    body = CallableBody(openIteration)
    callable = Callable("open_thread_files", body, "void", [filesArray, nthreads])

    return callable

@iet_pass
def ooc_efuncs(iet, **kwargs):
    saveCallable = saveB()

    new_save_call = Call(name="saveData", arguments=[])
    calls = FindNodes(Call).visit(iet)
    save_call = next((call for call in calls if call.name == 'save'), None)
    mapper={save_call: new_save_call}
    iet = Transformer(mapper).visit(iet)
    efuncs=[saveCallable]
    return iet, {'efuncs': efuncs}