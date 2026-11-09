"""Native shell batch/thread timings; run with the selected build on PYTHONPATH."""
import json
import time
import statistics
import numpy as np
from litepgo import fem, dof, mesh, tbb_thread_limit


def main():
    n = 20
    nodes = np.array([[x/n, y/n, .05*x*y/n**2] for y in range(n+1) for x in range(n+1)])
    faces = []
    for y in range(n):
        for x in range(n):
            a = y*(n+1)+x
            faces.extend([[a,a+1,a+n+2],[a,a+n+2,a+n+1]])
    topology = mesh.TriangleTopology(len(nodes), np.array(faces, dtype=np.int32))
    stencil = mesh.make_triangle_mid_edge_stencils(topology)
    mapping = dof.StencilDofMap(len(nodes), stencil.vertices, 3)
    rest = [fem.TriangleMidEdgeRestData(mapping.gather(e,nodes.ravel()),stencil.opposite[e]) for e in range(len(faces))]
    u = (.01*nodes).ravel()
    reference = {}
    for threads in (1,4):
        with tbb_thread_limit(threads):
            for batch in (1,256):
                obj = fem.ShellFemEnergy(fem.PointwiseShellAdapter[fem.KoiterStVK](),mapping,rest,[.1]*len(faces),[np.array([2.,3.])]*len(faces),batch_size=batch)
                for op in ('value','gradient','hessian'):
                    fn = getattr(obj,'eval_'+op)
                    value = fn(u)
                    array = value.toarray() if op=='hessian' else value
                    if op not in reference: reference[op] = array
                    np.testing.assert_allclose(array,reference[op],rtol=1e-10,atol=1e-10)
                    fn(u)
                    samples=[]
                    for _ in range(20):
                        start=time.perf_counter(); value=fn(u); samples.append((time.perf_counter()-start)*1000)
                    print(json.dumps(dict(faces=len(faces),threads=threads,batch=batch,operation=op,median_ms=statistics.median(samples),samples_ms=samples)),flush=True)

if __name__=='__main__': main()
