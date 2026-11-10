"""Release-build FEM batch/thread sweep; verifies results before timing."""
import csv
import sys
import time
from statistics import median
import numpy as np
from litepgo import dof, energy, fem, mesh, tbb_thread_limit

writer = csv.writer(sys.stdout)
writer.writerow(['elements', 'batch', 'threads', 'operation', 'median_ms'])
for count in (128, 4096):
    topology = mesh.TetTopology(4 * count, np.arange(4 * count, dtype=np.int32).reshape(-1, 4))
    mapping = dof.VertexDofMap(topology, 3)
    geometry = fem.InterpolatedResponse[fem.TetP1Basis](np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]))
    fp = energy.AffineResponse(np.zeros((9,3)), np.eye(3).ravel())
    theta = energy.AffineResponse(np.zeros((2,3)), np.array([2.,3.]))
    x = np.tile(np.arange(12) * .001, count)
    reference = None
    for batch in (1, 16, 64, 256):
        obj = fem.VolumetricFemEnergy[fem.TetP1Basis, fem.TetCentroidQuadrature](fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, [geometry]*count, [fp]*count, [theta]*count, batch_size=batch)
        actual = obj.evaluate(x).gradient()
        if reference is None:
            reference = actual
        np.testing.assert_array_equal(actual, reference)
        for threads in (1, 2, 4, 8):
            with tbb_thread_limit(threads):
                for name in ('value', 'gradient', 'hessian'):
                    operation = (lambda x: obj.evaluate(x).value) if name == "value" else (lambda x: getattr(obj.evaluate(x), name)())
                    operation(x)
                    samples = []
                    for _ in range(9):
                        start = time.perf_counter()
                        operation(x)
                        samples.append((time.perf_counter()-start)*1000)
                    writer.writerow([count, batch, threads, name, median(samples)])
                    sys.stdout.flush()
