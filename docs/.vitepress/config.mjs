import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'litepgo',
  description: 'Models, formulations, examples, and C++/Python API documentation',
  base: process.env.DOCS_BASE || '/',
  cleanUrls: false,
  markdown: {
    math: true,
    config(md) {
      const fence = md.renderer.rules.fence
      md.renderer.rules.fence = (tokens, index, ...args) => {
        if (tokens[index].info.trim() === 'mermaid') {
          return `<Mermaid graph="${encodeURIComponent(tokens[index].content)}" />`
        }
        return fence(tokens, index, ...args)
      }
    }
  },
  vite: { optimizeDeps: { include: ['mermaid'] } },
  themeConfig: {
    search: { provider: 'local' },
    nav: [
      { text: 'Get started', link: '/guide/build' },
      { text: 'User guide', link: '/guide/architecture' },
      { text: 'Energy tutorial', link: '/tutorials/energy/' },
      { text: 'Torch tutorial', link: '/tutorials/torch/' },
      { text: 'Examples', link: '/examples' },
      { text: 'Benchmarks', link: '/benchmarks/' },
      { text: 'API reference', link: '/reference' },
      { text: 'Development', link: '/development' }
    ],
    sidebar: [
      { text: 'Get started', items: [
        { text: 'Overview', link: '/' },
        { text: 'Build and environment', link: '/guide/build' },
        { text: 'First complete solve', link: '/guide/quickstart' }
      ] },
      { text: 'Energy tutorial', link: '/tutorials/energy/', items: [
        { text: '1. Spring-chain equilibrium', items: [
          { text: 'Python', link: '/tutorials/energy/01_spring_chain_python' },
          { text: 'C++20', link: '/tutorials/energy/01_spring_chain_cpp' }
        ] },
        { text: '2. Spring differences and gradients', items: [
          { text: 'Python', link: '/tutorials/energy/02_spring_response_python' },
          { text: 'C++20', link: '/tutorials/energy/02_spring_response_cpp' }
        ] }
      ] },
      { text: 'Torch tutorial', link: '/tutorials/torch/', collapsed: false, items: [
        { text: '1. Energy and stress', link: '/tutorials/torch/01_density' },
        { text: '2. Native FEM assembly', link: '/tutorials/torch/02_fem' },
        { text: '3. Fixed-pose parameter fitting', link: '/tutorials/torch/03_fixed_pose_fit' },
        { text: '4. Equilibrium differentiation', link: '/tutorials/torch/04_equilibrium_fit' },
        { text: '5. A small neural density', link: '/tutorials/torch/05_neural_density' },
        { text: '6. Fit network weights', link: '/tutorials/torch/06_neural_equilibrium_fit' },
        { text: '7. Design a plastic state', link: '/tutorials/torch/07_plastic_gravity_design' }
      ] },
      { text: 'User guide', items: [
        { text: 'Architecture', link: '/guide/architecture' },
        { text: 'Core concepts and conventions', link: '/guide/concepts' },
        { text: 'Common', link: '/guide/common' },
        { text: 'Mesh and I/O', link: '/guide/mesh/', collapsed: true, items: [
          { text: 'Mesh data and topology', link: '/guide/mesh/data' },
          { text: 'I/O and surface extraction', link: '/guide/mesh/io' }
        ] },
        { text: 'DOF maps and constraints', link: '/guide/dof/', collapsed: true, items: [
          { text: 'DOF maps', link: '/guide/dof/maps' },
          { text: 'Hard constraints', link: '/guide/dof/constraints' },
          { text: 'Surface kinematics', link: '/guide/dof/surface-kinematics' }
        ] },
        { text: 'Linear algebra', link: '/guide/linalg/', collapsed: true, items: [
          { text: 'Factorization lifecycle', link: '/guide/linalg/factorization' },
          { text: 'Backend selection', link: '/guide/linalg/backends' }
        ] },
        { text: 'Energy and response', link: '/guide/energy/', collapsed: true, items: [
          { text: 'Evaluation and composition', link: '/guide/energy/evaluation' },
          { text: 'Parameter transforms', link: '/guide/energy/parameter-transforms' },
          { text: 'Custom energies and responses', link: '/guide/energy/custom' },
          { text: 'Response evaluation', link: '/guide/energy/responses' },
          { text: 'Response Hessians', link: '/guide/energy/response-hessians' },
          { text: 'Batched assembly', link: '/guide/energy/assembly' }
        ] },
        { text: 'Finite elements', link: '/guide/fem/', collapsed: true, items: [
          { text: 'Basis functions', link: '/guide/fem/basis' },
          { text: 'Quadrature', link: '/guide/fem/quadrature' },
          { text: 'Spatial interpolation response', link: '/guide/fem/interpolated-response' },
          { text: 'Coefficient sampling responses', link: '/guide/fem/sampling-responses' },
          { text: 'Spatial responses', link: '/guide/fem/fields' },
          { text: 'Volumetric elasticity', link: '/guide/fem/volume' },
          { text: 'Shells and natural geometry', link: '/guide/fem/shell' },
          { text: 'Hex Hermite DOFs', link: '/guide/fem/hermite' },
          { text: 'Point embedding and boundary sampling', link: '/guide/fem/embedding' },
          { text: 'Force responses', link: '/guide/fem/force-responses' },
          { text: 'Materials written in Python', link: '/guide/fem/python-materials' }
        ] },
        { text: 'Solvers', link: '/guide/solver/', collapsed: true, items: [
          { text: 'Newton method', link: '/guide/solver/newton' },
          { text: 'Gauss–Newton least squares', link: '/guide/solver/gauss-newton' },
          { text: 'Equilibrium sensitivities', link: '/guide/solver/equilibrium' },
          { text: 'Inverse problems', link: '/guide/solver/inverse' }
        ] },
        { text: 'Contact', link: '/guide/contact/', collapsed: true, items: [
          { text: 'Barrier formulation', link: '/guide/contact/barrier' },
          { text: 'Displacement coupling', link: '/guide/contact/coupling' },
          { text: 'Primitive groups and multipliers', link: '/guide/contact/pair-weights' },
          { text: 'Collision safety and diagnostics', link: '/guide/contact/collision-safety' }
        ] },
        { text: 'Spatial queries', link: '/guide/spatial' },
        { text: 'Torch', link: '/guide/torch' }
      ] },
      { text: 'Examples', collapsed: true, items: [
        { text: 'Catalog and run instructions', link: '/examples' },
        { text: 'FEM basics', link: '/examples/fem/', collapsed: true, items: [
          { text: 'Spatial interpolation response', link: '/examples/fem/interpolated-response' },
          { text: 'Coefficient sampling responses', link: '/examples/fem/sampling-response' },
          { text: 'Tet P1 basis and field', link: '/examples/fem/tet-p1-basis' },
          { text: 'Hermite interpolation', link: '/examples/fem/hermite-interpolation' },
          { text: 'Quadrature integration', link: '/examples/fem/quadrature-integration' },
          { text: 'Nonlinear response Hessians', link: '/examples/fem/response-hessian' },
          { text: 'Volumetric material in NumPy', link: '/examples/fem/numpy-neohookean' },
          { text: 'Shell material in NumPy', link: '/examples/fem/numpy-koiter-shell' }
        ] },
        { text: 'Energies and responses', link: '/examples/energy/', collapsed: true, items: [
          { text: 'Custom scalar energies', link: '/examples/energy/custom-energy' },
          { text: 'Mesh energy subclass', link: '/examples/energy/mesh-energy-subclass' },
          { text: 'Energy composition', link: '/examples/energy/energy-composition' },
          { text: 'Mesh response subclass', link: '/examples/energy/mesh-response-subclass' },
          { text: 'Custom vector responses', link: '/examples/energy/custom-response' },
          { text: 'Batched material-force fitting', link: '/examples/energy/mesh-response' }
        ] },
        { text: 'Newton and Gauss–Newton', link: '/examples/solvers/', collapsed: true, items: [
          { text: 'Constrained chain', link: '/examples/solvers/newton-chain' },
          { text: 'TBB thread limits', link: '/examples/solvers/thread-limit' },
          { text: 'Exponential curve fitting', link: '/examples/solvers/gauss-newton-fit' }
        ] },
        { text: 'Mesh optimization', link: '/examples/mesh/', collapsed: true, items: [
          { text: 'Laplacian smoothing and editing', link: '/examples/mesh/laplacian-optimization' },
          { text: 'ARAP local-global deformation', link: '/examples/mesh/arap' }
        ] },
        { text: 'Mesh processing', link: '/examples/mesh-processing/', collapsed: true, items: [
          { text: 'Constrained isotropic remeshing', link: '/examples/mesh-processing/constrained-remesh' }
        ] },
        { text: 'Elastic equilibrium', link: '/examples/equilibrium/', collapsed: true, items: [
          { text: 'Tet P1 corner pull', link: '/examples/equilibrium/tet-p1-corner-pull' },
          { text: 'Hex Q1 corner pull', link: '/examples/equilibrium/hex-q1-corner-pull' },
          { text: 'Hex Hermite corner pull', link: '/examples/equilibrium/hex-hermite-corner-pull' },
          { text: 'Shell saddle pull', link: '/examples/equilibrium/shell-saddle-pull' }
        ] },
        { text: 'Prescribed plastic fields', link: '/examples/plasticity/', collapsed: true, items: [
          { text: 'Tet P1 plastic twist', link: '/examples/plasticity/tet-p1-plastic-twist' },
          { text: 'Hex Q1 plastic twist', link: '/examples/plasticity/hex-q1-plastic-twist' },
          { text: 'Hex Hermite plastic twist', link: '/examples/plasticity/hex-hermite-plastic-twist' },
          { text: 'Differential shell growth', link: '/examples/plasticity/shell-plastic-field' }
        ] },
        { text: 'Contact', link: '/examples/contact/', collapsed: true, items: [
          { text: 'Pressed tet box stack', link: '/examples/contact/tet-p1-box-stack' },
          { text: 'Compression between penalty plates', link: '/examples/contact/tet-p1-plate-compression' },
          { text: 'Area weighting under refinement', link: '/examples/contact/area-weighting' },
          { text: 'Read-only contact snapshot', link: '/examples/contact/contact-snapshot' },
          { text: 'Three primitive groups', link: '/examples/contact/pair-weights' }
        ] },
        { text: 'Spatial queries and embedding', link: '/examples/spatial/', collapsed: true, items: [
          { text: 'Spatial hash queries', link: '/examples/spatial/spatial-query' },
          { text: 'Point embedding', link: '/examples/spatial/point-embedding' }
        ] },
        { text: 'Material and plastic fitting', link: '/examples/fitting/', collapsed: true, items: [
          { text: 'Fixed-pose shell material', link: '/examples/fitting/shell-fixed-pose-material-fit' },
          { text: 'Fixed-pose volume plastic stretches', link: '/examples/fitting/volume-fixed-pose-fp-fit' },
          { text: 'Fixed-pose shell plastic stretches', link: '/examples/fitting/shell-fixed-pose-fp-fit' },
          { text: 'Equilibrium volume material', link: '/examples/fitting/volume-equilibrium-material-fit' },
          { text: 'Equilibrium shell material', link: '/examples/fitting/shell-equilibrium-material-fit' },
          { text: 'Equilibrium volume plastic stretches', link: '/examples/fitting/volume-equilibrium-fp-fit' },
          { text: 'Equilibrium shell plastic stretches', link: '/examples/fitting/shell-equilibrium-fp-fit' }
        ] },
        { text: 'PyTorch materials and fitting', link: '/examples/torch/', collapsed: true, items: [
          { text: 'Learn a neural material', link: '/examples/torch/neural-neohookean' },
          { text: 'Fit network weights through equilibrium', link: '/examples/torch/neural-equilibrium-fit' },
          { text: 'Force responses through equilibrium', link: '/examples/torch/torch-equilibrium-response' },
          { text: 'Constitutive energy in Torch', link: '/examples/torch/torch-neohookean' },
          { text: 'Shell material written in Torch', link: '/examples/torch/torch-shell-material-fit' },
          { text: 'Volume material with L-BFGS', link: '/examples/torch/volume-equilibrium-material-fit-torch' },
          { text: 'Shell material with L-BFGS', link: '/examples/torch/shell-equilibrium-material-fit-torch' },
          { text: 'Volume plastic stretches with L-BFGS', link: '/examples/torch/volume-equilibrium-fp-fit-torch' },
          { text: 'Shell plastic stretches with L-BFGS', link: '/examples/torch/shell-equilibrium-fp-fit-torch' }
        ] }
      ] },
      { text: 'Reference and development', items: [
        { text: 'About the API references', link: '/reference' },
        // Generated HTML must bypass VitePress's client router.
        { text: 'C++ reference', link: '/reference/cpp/index.html', target: '_self' },
        { text: 'Python reference', link: '/reference/python/index.html', target: '_self' },
        { text: 'Benchmarks', link: '/benchmarks/', items: [
          { text: 'Torch material performance', link: '/benchmarks/torch-material' }
        ] },
        { text: 'Development', link: '/development' }
      ] }
    ]
  }
})
