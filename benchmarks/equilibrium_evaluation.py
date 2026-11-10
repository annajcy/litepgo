"""End-to-end response timings; selected build is supplied through PYTHONPATH."""
from litepgo.solver import AbsoluteConvergence
import argparse
import json
import statistics
import time
import numpy as np
import scipy.sparse as sp
import torch
from litepgo import dof, energy, fem, mesh, solver, tbb_thread_limit
from litepgo import torch as lt


def problem(family, n, kind, counts):
    theta=np.array([2.,3.])
    if family.startswith('shell'):
        nodes=np.array([[x/n,y/n,.03*x*y/n**2] for y in range(n+1) for x in range(n+1)])
        faces=[]
        for y in range(n):
            for x in range(n):
                a=y*(n+1)+x
                faces.extend([[a,a+1,a+n+2],[a,a+n+2,a+n+1]])
        topology=mesh.TriangleTopology(len(nodes),np.array(faces,dtype=np.int32))
        st=mesh.make_triangle_mid_edge_stencils(topology)
        mapping=dof.StencilDofMap(len(nodes),st.vertices,3)
        rest=[fem.TriangleMidEdgeRestData(mapping.gather(e,nodes.ravel()),st.opposite[e]) for e in range(len(faces))]
        material=fem.PointwiseShellAdapter[fem.KoiterStVK]()
        elastic=lambda p: fem.ShellFemEnergy(material,mapping,rest,[.1]*len(faces),[p]*len(faces), Fp=[np.eye(3)] * len(faces))
        force=lambda u: fem.ShellFemForceMaterialResponse(material,dof.ConstantDofMap(topology,2),mapping,rest,u,[.1]*len(faces))
        if family=='shell_fp':
            theta=np.eye(3).ravel(order='F')
            elastic=lambda p: fem.ShellFemEnergy(material,mapping,rest,[.1]*len(faces),[np.array([2.,3.])]*len(faces),[p.reshape(3,3,order='F')]*len(faces))
            force=lambda u: fem.ShellFemForceFpResponse(material,dof.ConstantDofMap(topology,9),mapping,rest,u,[.1]*len(faces),[np.array([2.,3.])]*len(faces))
    else:
        nodes=np.array([[x/n,y/n,z/n] for z in range(n+1) for y in range(n+1) for x in range(n+1)])
        cells=[]
        for z in range(n):
            for y in range(n):
                for x in range(n):
                    a=z*(n+1)**2+y*(n+1)+x;b=a+(n+1)**2
                    h=[a,a+1,a+n+2,a+n+1,b,b+1,b+n+2,b+n+1]
                    if family=='tet':
                        for t in [(0,1,2,6),(0,2,3,6),(0,3,7,6),(0,7,4,6),(0,4,5,6),(0,5,1,6)]: cells.append([h[i] for i in t])
                    else: cells.append(h)
        B=fem.TetP1Basis if family=='tet' else fem.HexHermiteBasis
        G=fem.TetP1Basis if family=='tet' else fem.HexQ1Basis
        Q=fem.TetCentroidQuadrature if family=='tet' else fem.GaussLegendre3D[2]
        topology=(mesh.TetTopology if family=='tet' else mesh.HexTopology)(len(nodes),np.array(cells,dtype=np.int32))
        mapping=dof.VertexDofMap(topology,3 if family=='tet' else 24)
        geometry=[fem.InterpolatedResponse[G](nodes[c]) for c in cells]
        plastic=[energy.AffineResponse(sp.csc_matrix((9,3)),np.eye(3).ravel())]*len(cells)
        material=fem.PointwiseVolumetricAdapter[fem.NeoHookean]()
        elastic=lambda p: fem.VolumetricFemEnergy[B,Q](material,mapping,geometry,plastic,[energy.AffineResponse(sp.csc_matrix((2,3)),p)]*len(cells))
        force=lambda u: fem.VolumetricFemForceMaterialResponse[B,Q](material,dof.ConstantDofMap(topology,2),mapping,geometry,u,plastic)
    width=24 if family=='hermite' else 3
    fixed_ids=[width*i+j for i in range(len(nodes)) if nodes[i,0]==0 for j in range(width)]
    initial=np.zeros(mapping.num_dofs)
    load=np.zeros_like(initial)
    for i in range(len(nodes)):
        if nodes[i,0]==1: load[width*i+2]=-.001/len(nodes)
    # Small elastic foundation stabilizes shell bending and underintegrated Hermite modes.
    foundation=energy.QuadraticEnergy(sp.eye(len(initial),format='csc')*.01,-load)
    def make_energy(p):
        counts['solves']+=1
        return energy.WeightedSumEnergy([(elastic(p),1.),(foundation,1.)])
    eq=solver.StaticEquilibriumResponse(len(theta),dof.DofReduction(len(initial),fixed_ids),initial,solver.NewtonOptions(max_iterations=100,project_hessian=kind),make_energy,force, convergence=AbsoluteConvergence(1e-9))
    return eq,theta,make_energy,np.setdiff1d(np.arange(len(initial)),fixed_ids)


def main():
    args=argparse.ArgumentParser();args.add_argument('--samples',type=int,default=7);a=args.parse_args()
    for threads in (1,4):
        with tbb_thread_limit(threads):
            for family in ('tet','hermite','shell','shell_fp'):
                for n in (1,2):
                    for kind in (False,True):
                        counts={'solves':0}
                        eq,p,potential,free=problem(family,n,kind,counts)
                        u=eq.evaluate(p).value
                        residual=float(np.max(np.abs(potential(p).evaluate(u).gradient()[free])))
                        assert residual <= 1.1e-9
                        if n==1:
                            d=np.sin(np.arange(len(p))+.3);h=1e-5
                            np.testing.assert_allclose(eq.evaluate(p).jvp(d),(eq.evaluate(p+h*d).value-eq.evaluate(p-h*d).value)/(2*h),rtol=2e-4,atol=2e-7)
                        for mode in ('vjp','torch','composed'):
                            model=energy.ComposedResponse(energy.AffineResponse(sp.eye(eq.num_outputs,format='csc')),eq) if mode=='composed' else eq
                            def run():
                                if mode=='vjp': return model.evaluate(p).vjp(np.ones(model.num_outputs))
                                x=torch.tensor(p,dtype=torch.float64,requires_grad=True)
                                y=lt.response(model,x);y.sum().backward()
                                return x.grad.numpy()
                            ref=run();samples=[];calls=[]
                            for _ in range(a.samples):
                                counts['solves']=0;t=time.perf_counter();g=run();samples.append(1000*(time.perf_counter()-t));calls.append(counts['solves'])
                                np.testing.assert_allclose(g,ref,atol=1e-10,rtol=1e-8)
                            print(json.dumps(dict(family=family,n=n,threads=threads,hessian=str(kind),mode=mode,median_ms=statistics.median(samples),samples_ms=samples,solves=calls,gradient=ref.tolist(),state=u.tolist(),residual=residual)),flush=True)
    with tbb_thread_limit(1):
        for name,obj,x in [('affine',energy.AffineResponse(sp.eye(3,format='csc')),np.ones(3)),('exp',energy.ExpResponse(3),np.ones(3))]:
            samples=[]
            for _ in range(20):
                t=time.perf_counter()
                for _ in range(1000): obj.evaluate(x).value
                samples.append((time.perf_counter()-t))
            print(json.dumps(dict(case=name,median_us=statistics.median(samples)*1000,samples_us=[s*1000 for s in samples])),flush=True)

if __name__=='__main__': main()
