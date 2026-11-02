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
      { text: 'Examples', link: '/examples' },
      { text: 'API reference', link: '/reference' },
      { text: 'Development', link: '/development' }
    ],
    sidebar: [
      { text: 'Get started', items: [
        { text: 'Overview', link: '/' },
        { text: 'Build and environment', link: '/guide/build' },
        { text: 'First complete solve', link: '/guide/quickstart' }
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
          { text: 'Custom energies and responses', link: '/guide/energy/custom' },
          { text: 'Batched assembly', link: '/guide/energy/assembly' }
        ] },
        { text: 'Finite elements', link: '/guide/fem/', collapsed: true, items: [
          { text: 'Fields and quadrature', link: '/guide/fem/fields' },
          { text: 'Volumetric elasticity', link: '/guide/fem/volume' },
          { text: 'Shells and natural geometry', link: '/guide/fem/shell' },
          { text: 'Hex Hermite DOFs', link: '/guide/fem/hermite' },
          { text: 'Point embedding and boundary sampling', link: '/guide/fem/embedding' },
          { text: 'Force responses', link: '/guide/fem/force-responses' }
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
          { text: 'Collision safety and diagnostics', link: '/guide/contact/collision-safety' }
        ] },
        { text: 'Spatial queries', link: '/guide/spatial' },
        { text: 'Torch', link: '/guide/torch' }
      ] },
      { text: 'Examples', collapsed: true, items: [
        { text: 'Catalog and run instructions', link: '/examples' },
        { text: 'Energies and responses', link: '/examples/energy' },
        { text: 'Newton and Gauss–Newton', link: '/examples/solvers' },
        { text: 'Laplacian and ARAP', link: '/examples/mesh' },
        { text: 'Elastic equilibrium', link: '/examples/equilibrium' },
        { text: 'Prescribed plastic fields', link: '/examples/plasticity' },
        { text: 'Contact', link: '/examples/contact' },
        { text: 'Spatial queries and embedding', link: '/examples/spatial' },
        { text: 'Material and plastic fitting', link: '/examples/fitting' },
        { text: 'PyTorch fitting', link: '/examples/torch' }
      ] },
      { text: 'Reference and development', items: [
        { text: 'About the API references', link: '/reference' },
        // Generated HTML must bypass VitePress's client router.
        { text: 'C++ reference', link: '/reference/cpp/index.html', target: '_self' },
        { text: 'Python reference', link: '/reference/python/index.html', target: '_self' },
        { text: 'Development', link: '/development' }
      ] }
    ]
  }
})
