"""Reproducible native JVP/VJP microbenchmarks with explicit-Jacobian checks."""
import json
import statistics
import time
import numpy as np
from litepgo import fem, dof, mesh, energy, tbb_thread_limit


def cases():
    p = np.array([[1.1,.04,0],[.02,.95,.03],[0,.02,1.03]]).ravel(order='F')
    theta = np.array([2.,3.])
    constant = lambda x: energy.AffineResponse(np.zeros((len(x),3)),x)
    for B,width in [(fem.HexQ1Basis,3),(fem.HexHermiteBasis,24)]:
        corners = np.array([fem.HexQ1Basis.coefficient_info(i).point for i in range(8)])*[1.7,.8,1.2]
        count=64
        topology=mesh.HexTopology(8*count,np.arange(8*count,dtype=np.int32).reshape(-1,8))
        mapping=dof.VertexDofMap(topology,width)
        geometry=[fem.InterpolatedResponse[fem.HexQ1Basis](corners)]*count
        u=.002*np.sin(np.arange(8*count*width))
        for batch in (1,256):
            for kind,size,x,fixed in [('Material',2,theta,p),('Fp',9,p,theta)]:
                cls=getattr(fem,'VolumetricFemForce'+kind+'Response')[B,fem.GaussLegendre3D[2]]
                obj=cls(fem.PointwiseVolumetricAdapter[fem.NeoHookean](),dof.ConstantDofMap(topology,size),mapping,geometry,u,[constant(fixed)]*count,batch)
                yield f'{B.__name__}/{kind}/{batch}',obj,x
        geometry=fem.InterpolatedResponse[fem.HexQ1Basis](corners)
        C=.01*np.sin(np.arange(B.num_basis_functions*3)).reshape(-1,3)
        yield f'spatial/{B.__name__}',fem.DeformationGradientResponse(fem.InterpolatedResponse[B](C),geometry),np.array([.2,.3,.4])
    n=12
    nodes=np.array([[x/n,y/n,.05*x*y/n**2] for y in range(n+1) for x in range(n+1)])
    faces=[]
    for y in range(n):
        for x in range(n):
            a=y*(n+1)+x
            faces.extend([[a,a+1,a+n+2],[a,a+n+2,a+n+1]])
    topology=mesh.TriangleTopology(len(nodes),np.array(faces,dtype=np.int32))
    stencil=mesh.make_triangle_mid_edge_stencils(topology)
    mapping=dof.StencilDofMap(len(nodes),stencil.vertices,3)
    rest=[fem.TriangleMidEdgeRestData(mapping.gather(e,nodes.ravel()),stencil.opposite[e]) for e in range(len(faces))]
    for batch in (1,256):
        for kind,size,x in [('Material',2,theta),('Fp',9,p)]:
            args=[fem.PointwiseShellAdapter[fem.KoiterStVK](),dof.ConstantDofMap(topology,size),mapping,rest,.01*nodes.ravel(),[.1]*len(faces)]
            if kind=='Fp': args.append([theta]*len(faces))
            obj=getattr(fem,'ShellFemForce'+kind+'Response')(*args,batch_size=batch)
            yield f'shell/{kind}/{batch}',obj,x


def main():
    with tbb_thread_limit(1):
        for name,obj,x in cases():
            J=obj.evaluate(x).jacobian()
            for op,d in [('jvp',np.sin(np.arange(obj.num_inputs))),('vjp',np.cos(np.arange(obj.num_outputs)))]:
                fn=lambda: getattr(obj.evaluate(x),op)(d)
                expected=J@d if op=='jvp' else J.T@d
                np.testing.assert_allclose(fn(),expected,rtol=1e-9,atol=1e-10)
                samples=[]
                for _ in range(30):
                    start=time.perf_counter()
                    for _ in range(10): fn()
                    samples.append((time.perf_counter()-start)*100)
                print(json.dumps(dict(case=name,operation=op,median_ms=statistics.median(samples),samples_ms=samples)),flush=True)

if __name__=='__main__': main()
