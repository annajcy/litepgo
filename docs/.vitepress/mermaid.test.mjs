import assert from 'node:assert/strict'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { createMarkdownRenderer } from 'vitepress'
import config from './config.mjs'

const markdown = await createMarkdownRenderer(
  fileURLToPath(new URL('../', import.meta.url)), config.markdown)

test('multiple Mermaid fences preserve distinct, safely encoded sources', () => {
  const graphs = ['flowchart TB\n  a["a < b"] --> b\n', 'flowchart LR\n  x --> y\n']
  const html = markdown.render(graphs.map(graph => `\`\`\`mermaid\n${graph}\`\`\``).join('\n\n'))
  const sources = [...html.matchAll(/<Mermaid graph="([^"]*)" \/>/g)]
    .map(match => decodeURIComponent(match[1]))
  assert.deepEqual(sources, graphs)
  assert.ok(!html.includes('a < b'))
})

test('ordinary code fences and LaTeX still use the existing renderers', () => {
  const html = markdown.render('```python\nprint(1)\n```\n\n$$\nx^2\n$$\n')
  assert.match(html, /language-python/)
  assert.match(html, /mjx-container/)
  assert.ok(!html.includes('<Mermaid'))
})
