import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'litepgo',
  description: 'litepgo C++ and Python documentation',
  cleanUrls: true,
  themeConfig: {
    search: { provider: 'local' },
    nav: [
      { text: 'Guide', link: '/guide/build' },
      { text: 'API', link: '/api/mesh' },
      { text: 'Examples', link: '/examples' },
      // Generated HTML must bypass VitePress's client router.
      { text: 'C++ reference', link: '/reference/cpp/index.html', target: '_self' },
      { text: 'Python reference', link: '/reference/python/index.html', target: '_self' },
      { text: 'Source', link: 'https://github.com/annajcy/litepgo' }
    ],
    sidebar: [
      { text: 'Guide', items: [
        { text: 'Build and test', link: '/guide/build' },
        { text: 'Development', link: '/guide/development' },
        { text: 'Examples', link: '/examples' }
      ] },
      { text: 'API', items: [
        { text: 'Mesh I/O', link: '/api/mesh' },
        { text: 'FEM', link: '/api/fem' },
        { text: 'Contact', link: '/api/contact' },
        { text: 'Spatial queries and embedding', link: '/api/spatial' },
        { text: 'C++ reference', link: '/reference/cpp/index.html', target: '_self' },
        { text: 'Python reference', link: '/reference/python/index.html', target: '_self' }
      ] }
    ]
  }
})
