#!/usr/bin/env python3
"""Validated structured meshes and the native/IPCTK scaling matrix (standard library only)."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/contact_scaling'


def generate(n, bounds, path):
    vertices = list(itertools.product(*( [lo + (hi-lo)*i/n for i in range(n+1)] for lo, hi in bounds)))
    def index(p):
        return (p[0]*(n+1)+p[1])*(n+1)+p[2]
    tets = []
    for cell in itertools.product(range(n), repeat=3):
        for perm in itertools.permutations(range(3)):
            p = list(cell)
            tet = [index(p)]
            for axis in perm:
                p[axis] += 1
                tet.append(index(p))
            a,b,c,d = (vertices[i] for i in tet)
            u,v,w = ([q[i]-a[i] for i in range(3)] for q in (b,c,d))
            det = sum(u[i]*(v[(i+1)%3]*w[(i+2)%3]-v[(i+2)%3]*w[(i+1)%3]) for i in range(3))
            if det < 0:
                tet[1],tet[2] = tet[2],tet[1]
                det = -det
            assert det > 0
            tets.append((tet, det/6))
    volume = sum(v for _,v in tets)
    expected = math.prod(hi-lo for lo,hi in bounds)
    assert math.isclose(volume, expected, rel_tol=1e-11)
    assert len(tets) == 6*n**3 and len(vertices) == (n+1)**3
    assert len({tuple(sorted(t)) for t,_ in tets}) == len(tets)
    faces = Counter(tuple(sorted(f)) for t,_ in tets for f in itertools.combinations(t,3))
    assert set(faces.values()) == {1,2}
    boundary = [f for f,count in faces.items() if count == 1]
    assert len(boundary) == 12*n*n
    edges = Counter(tuple(sorted(e)) for f in boundary for e in itertools.combinations(f,2))
    assert set(edges.values()) == {2}
    for face in boundary:
        assert any(all(math.isclose(vertices[v][axis], end, abs_tol=1e-14) for v in face)
                   for axis in range(3) for end in bounds[axis])
    with path.open('w') as f:
        f.write(f'*VERTICES\n{len(vertices)} 3 0 0\n')
        for i,p in enumerate(vertices,1):
            f.write(f'{i} '+ ' '.join(format(v,'.17g') for v in p)+'\n')
        f.write(f'*ELEMENTS\nTET\n{len(tets)} 4 0\n')
        for i,(tet,_) in enumerate(tets,1):
            f.write(f'{i} '+ ' '.join(str(v+1) for v in tet)+'\n')
    return dict(subdivisions=n, stack_vertices=2*len(vertices), stack_tets=2*len(tets),
                box_volume=volume, expected_volume=expected, inverted=0, duplicate_tets=0,
                boundary_triangles=len(boundary), surface_closed=True, bounds=bounds)


def main(skip_build=False):
    OUT.mkdir(parents=True, exist_ok=True)
    lines=(ROOT/'assets/box-tet.veg').read_text().splitlines()
    start=lines.index('*VERTICES'); count=int(lines[start+1].split()[0])
    vertices=[list(map(float,line.split()[1:4])) for line in lines[start+2:start+2+count]]
    bounds=[(min(v[i] for v in vertices),max(v[i] for v in vertices)) for i in range(3)]
    top=sum(abs(v[1]-bounds[1][1]) < 1e-8 for v in vertices)
    meshes=[('baseline', ROOT/'assets/box-tet.veg')]
    validation={}
    for n in (9,15):
        path=OUT/f'grid{n}.veg'
        validation[f'grid{n}']=generate(n,bounds,path)
        meshes.append((f'grid{n}',path))
    (OUT/'mesh_validation.json').write_text(json.dumps(dict(meshes=validation,total_top_stiffness=top),indent=2)+'\n')
    for index,(name,mesh) in enumerate(meshes):
        for threads in (1,4,8):
            output=OUT/f'{name}_t{threads}'
            args=[sys.executable,str(ROOT/'scripts/contact_benchmark.py'),'--mesh',str(mesh),
                  '--threads',str(threads),'--repetitions','3','--output',str(output),
                  '--total-stiffness',str(top)]
            if skip_build or index or threads != 1:
                args.append('--skip-build')
            print(f'RUN {name} threads={threads}',flush=True)
            with (OUT/f'{name}_t{threads}.log').open('w') as log:
                subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(f'DONE {name} threads={threads}',flush=True)


def report():
    def read(path):
        with path.open() as f:
            return list(csv.DictReader(f))
    table=[]
    comparisons=[]
    for mesh,vertices in [('baseline',486),('grid9',2000),('grid15',8192)]:
        one=json.loads((OUT/f'{mesh}_t1/summary.json').read_text())
        for threads in (1,4,8):
            path=OUT/f'{mesh}_t{threads}'
            summary=json.loads((path/'summary.json').read_text())
            runs=read(path/'runs.csv')
            shared=read(path/'shared_states.csv')
            ccd=read(path/'shared_ccd.csv')
            traces=read(path/'iterations.csv')
            for backend in ('native','ipctk'):
                stats=summary[backend]
                measured=[r for r in runs if r['backend']==backend and int(r['round'])>0]
                table.append(dict(mesh=mesh,vertices=vertices,threads=threads,backend=backend,
                    median_seconds=stats['seconds_median'],min_seconds=stats['seconds_min'],max_seconds=stats['seconds_max'],
                    speedup_vs_1=one[backend]['seconds_median']/stats['seconds_median'],
                    iterations_min=min(int(r['iterations']) for r in measured),iterations_max=max(int(r['iterations']) for r in measured),
                    max_residual=max(float(r['free_residual']) for r in runs if r['backend']==backend),
                    max_repeat_displacement=stats['max_repeat_displacement']))
            comp=dict(mesh=mesh,threads=threads,**summary['displacement_comparison'])
            comp['displacement_relative_l2']=float(comp['rms'])*math.sqrt(vertices*3)/float(comp['native_norm'])
            comp['max_abs_over_target']=float(comp['max_abs'])/0.1
            for backend in ('native','ipctk'):
                comp[backend+'_max_residual']=max(float(r['free_residual']) for r in runs if r['backend']==backend)
            for key in ('energy_abs','gradient_max_abs','gradient_relative_l2','hessian_max_abs','hessian_relative_frobenius'):
                comp['shared_'+key]=max(float(r[key]) for r in shared)
            comp['max_ccd_step_difference']=max(abs(float(r['native_step'])-float(r['ipctk_step'])) for r in ccd)
            comp['ipctk_over_native_seconds']=summary['ipctk']['seconds_median']/summary['native']['seconds_median']
            for backend in ('native','ipctk'):
                first=[r for r in traces if r['backend']==backend and r['round']=='1']
                comp[backend+'_feasible_steps']=';'.join(r['feasible_step'] for r in first)
                comp[backend+'_line_search_evaluations']=sum(int(r['line_search_evaluations']) for r in first)
            comparisons.append(comp)
    for name,data in [('timings',table),('comparisons',comparisons)]:
        with (OUT/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    env=json.loads((OUT/'baseline_t1/environment.json').read_text())
    lines=['# Native versus IPCTK contact scaling','',
        f"Machine: {env.get('cpu',env['machine'])}; {env['platform']}. Release build with both backends.",
        '', '## Protocol', '',
        'Each of nine mesh/thread cases uses one warmup and three measured Newton solves per backend. Order alternates each round (warmup native/IPCTK, then IPCTK/native, native/IPCTK, IPCTK/native). TBB maximum parallelism is 1, 4 or 8. VECLIB_MAXIMUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 are fixed for every process; Accelerate is held at one thread to isolate TBB scaling.',
        '', 'Times enclose newton_solve only, including symbolic analysis and preallocated in-memory iteration recording; exclude construction, file I/O, shared-state diagnostics and displacement export. Each solve gets fresh contact and linear-solver instances. Native local contact evaluation/projection, output-owned gradient/Hessian assembly and CCD queries use TBB with fixed-order buffers; scalar energy and CCD minimum reductions remain sequential. Broadphase and active-set construction remain serial. IPCTK contact uses parallel work. End-to-end Newton measurements do not attribute time to individual phases.',
        '', '## Geometry and loading', '',
        'Baseline is the original 486-vertex two-box asset. Structured grids have 2,000 and 8,192 stack vertices (8,748 and 40,500 tets), built by six consistent Freudenthal tetrahedra per cube with positive orientation. mesh_validation.json records counts, bounds, volume 0.125 per box, zero inverted/duplicate tets, internal face incidence two, boundary-only exterior faces and closed surface edge incidence two.',
        '', 'All meshes use the original 0.5 × 0.5 × 0.5 box dimensions, 0.02 gap, lateral offsets, material, 0.1 downward target, dhat=0.005, kappa=1000 and gradient tolerance 1e-8. Total top-face spring stiffness stays 41: per-vertex values are 1 (41 asset vertices), 0.41 (100 grid9 vertices), and 41/256 (256 grid15 vertices). Equal vertex weights change the spatial load quadrature across meshes; FEM discretization and unweighted contact primitive counts also change. Cross-size results are scaling experiments, not mesh-convergence claims. Within each case both backends use exactly the same mesh, forces and options.',
        '', '## Newton times', '', '| Vertices | Threads | Native median (s) | IPCTK median (s) | IPCTK/native | Native speedup | IPCTK speedup |', '|---:|---:|---:|---:|---:|---:|---:|']
    for i in range(0,len(table),2):
        n,t=table[i:i+2]
        lines.append(f"| {n['vertices']} | {n['threads']} | {n['median_seconds']:.6f} | {t['median_seconds']:.6f} | {t['median_seconds']/n['median_seconds']:.2f} | {n['speedup_vs_1']:.2f} | {t['speedup_vs_1']:.2f} |")
    lines += ['', 'timings.csv includes min/max times, iteration ranges, residuals and repeat displacement errors.', '', '## Same-case numerical comparison','',
        '| Mesh | Threads | Displacement max abs | Displacement RMS | Shared energy abs | Shared gradient relative L2 | Shared Hessian relative Frobenius |', '|---|---:|---:|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append(f"| {c['mesh']} | {c['threads']} | {float(c['max_abs']):.3e} | {float(c['rms']):.3e} | {c['shared_energy_abs']:.3e} | {c['shared_gradient_relative_l2']:.3e} | {c['shared_hessian_relative_frobenius']:.3e} |")
    lines += ['', '| Mesh | Threads | Displacement relative L2 | Max error / 0.1 target | Native residual max | IPCTK residual max |', '|---|---:|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append(f"| {c['mesh']} | {c['threads']} | {c['displacement_relative_l2']:.3e} | {c['max_abs_over_target']:.3e} | {c['native_max_residual']:.3e} | {c['ipctk_max_residual']:.3e} |")
    lines += ['', 'All 72 solves passed the benchmark convergence checks. See timings.csv for actual iteration ranges and repeat displacement errors.', '']
    lines += ['', 'Relative L2 uses the native displacement norm. Residuals are maxima across warmup and measured runs against the 1e-8 stopping tolerance. Agreement of shared-state derivatives does not prove the cause of differing final solutions. No tighter-tolerance solves were performed.', '', 'Shared derivatives compare kappa-weighted contact at identical initial, native final, IPCTK final and midpoint states, with both exact and projected Hessians. Table reports the largest error across these eight comparisons. Displacement compares warmup equilibria; all measured repeats are checked separately. These shared-state checks assess formula agreement separately from final-solution and Newton/CCD-path differences.', '',
        'Every run must converge with free residual <=1e-8 and positive active contact distance below dhat. Each case saves runs.csv (energies, convergence, active pairs, distance), iterations.csv (every iteration, feasible/accepted steps, line-search counts, pattern changes, shifts), full-precision displacement CSVs, shared_states.csv and shared_ccd.csv. comparisons.csv also records first measured feasible-step paths and line-search totals. Separate Newton paths are permitted; equal iteration counts are not assumed.', '',
        '## Limits and reproduction','',
        'Run `python3 scripts/contact_scaling.py --output output/contact_scaling` from the repository; files at that output root are replaced. Historical serial results are preserved at output/contact_scaling_serial. For one case use scripts/contact_benchmark.py --threads N --mesh PATH --repetitions N --output PATH --total-stiffness 41. Original invocation retains one thread, the asset, five measured repetitions and original output directory.', '',
        'Timing comparisons with output/contact_scaling_serial are historical versus current, not controlled simultaneous experiments. Background processes are not terminated. Per-case processes_before.txt snapshots record competing load; scheduling and thermal effects may also affect timings. Three measured samples on a shared desktop are limited evidence. No per-phase profiler or tighter-tolerance solves were added.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def compare_serial():
    """Compare saved binary64 values; historical timings are not a controlled trial."""
    import struct
    serial = ROOT / 'output/contact_scaling_serial'
    if not serial.exists() or OUT == serial:
        return

    def read(path):
        with path.open() as stream:
            return list(csv.DictReader(stream))

    def bits(value):
        return struct.pack('d', float(value))

    timing, differences = [], []
    for mesh in ('baseline', 'grid9', 'grid15'):
        for threads in (1, 4, 8):
            case = f'{mesh}_t{threads}'
            old, new = serial / case, OUT / case
            old_env = json.loads((old / 'environment.json').read_text())
            new_env = json.loads((new / 'environment.json').read_text())
            for key in ('mesh_sha256', 'threads', 'repetitions', 'total_stiffness', 'thread_environment'):
                assert old_env[key] == new_env[key], (case, key)
            assert (old / 'build.txt').read_bytes() == (new / 'build.txt').read_bytes(), case
            for source, digest in old_env['source_sha256'].items():
                if source != 'cpp/litepgo/contact/ipc_contact_energy.cpp':
                    assert new_env['source_sha256'][source] == digest, (case, source)
            before = json.loads((old / 'summary.json').read_text())
            after = json.loads((new / 'summary.json').read_text())
            timing.append(dict(case=case, serial_native=before['native']['seconds_median'],
                               parallel_native=after['native']['seconds_median'],
                               historical_ipctk=before['ipctk']['seconds_median'],
                               current_ipctk=after['ipctk']['seconds_median']))
            for name in [f'{backend}_{r}_displacement.csv' for backend in ('native', 'ipctk') for r in range(4)] + ['shared_states.csv', 'shared_ccd.csv', 'iterations.csv', 'runs.csv']:
                a, b = read(old / name), read(new / name)
                assert len(a) == len(b) and a[0].keys() == b[0].keys(), (case, name)
                for key in a[0]:
                    if key == 'seconds':
                        continue
                    try:
                        va, vb = [float(r[key]) for r in a], [float(r[key]) for r in b]
                    except ValueError:
                        assert [r[key] for r in a] == [r[key] for r in b], (case, name, key)
                        continue
                    assert all(math.isfinite(v) for v in va + vb), (case, name, key)
                    differences.append(dict(case=case, file=name, field=key,
                        bitwise_equal=all(bits(x) == bits(y) for x, y in zip(va, vb)),
                        max_abs=max(abs(x-y) for x, y in zip(va, vb))))
    for name, rows in [('serial_parallel_timings', timing), ('serial_parallel_numerics', differences)]:
        with (OUT / f'{name}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines = ['', '## Historical serial versus current parallel', '',
             'The complete prior output is preserved in ../contact_scaling_serial. Mesh hashes, solver metadata, thread settings, repetitions and total stiffness were checked identical. Timings below are medians of three measured solves; these were separate sessions under uncontrolled background load, not paired speedup measurements.', '',
             '| Case | Historical serial native (s) | Current parallel native (s) | Historical IPCTK (s) | Current IPCTK (s) | Historical/current native ratio |',
             '|---|---:|---:|---:|---:|---:|']
    for row in timing:
        lines.append(f"| {row['case']} | {row['serial_native']:.6f} | {row['parallel_native']:.6f} | {row['historical_ipctk']:.6f} | {row['current_ipctk']:.6f} | {row['serial_native']/row['parallel_native']:.3f} |")
    lines += ['', '| Case | Native displacement max abs, all 4 rounds | Native displacement bitwise identical | Shared diagnostics max abs change | CCD max abs change |', '|---|---:|---|---:|---:|']
    for row in timing:
        ds = [d for d in differences if d['case'] == row['case']]
        disp = [d for d in ds if d['file'].startswith('native_') and d['field'] == 'displacement']
        shared = [d for d in ds if d['file'] == 'shared_states.csv']
        ccd = [d for d in ds if d['file'] == 'shared_ccd.csv']
        lines.append(f"| {row['case']} | {max(d['max_abs'] for d in disp):.3e} | {all(d['bitwise_equal'] for d in disp)} | {max(d['max_abs'] for d in shared):.3e} | {max(d['max_abs'] for d in ccd):.3e} |")
    lines += ['', 'serial_parallel_numerics.csv checks each numeric field by its parsed binary64 bytes, including every saved displacement in all four rounds, iteration paths, run diagnostics, shared exact/projected energy and derivative diagnostics, and CCD steps. Full shared derivative arrays were not saved in the old run: historical derivative comparisons are of the recorded norms and native/IPCTK difference metrics, not a bytewise comparison of every old matrix entry. New C++ tests separately compare every native sparse Hessian entry and gradient byte across 1/4/8 threads.', '']
    with (OUT / 'report.md').open('a') as stream:
        stream.write('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--skip-build', action='store_true')
    args = parser.parse_args()
    OUT = args.output.resolve()
    main(skip_build=args.skip_build)
    report()
    compare_serial()
