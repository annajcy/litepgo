import assert from 'node:assert/strict'
import test from 'node:test'
import { readFile, readdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { createMarkdownRenderer } from 'vitepress'
import config from './config.mjs'

const directory = new URL('../tutorials/torch/', import.meta.url)
const markdown = await createMarkdownRenderer(fileURLToPath(new URL('../', import.meta.url)), config.markdown)

test('all exported lessons preserve code, math, outputs, and plot links', async () => {
  const lessons = (await readdir(directory)).filter(name => /^\d\d_.*\.md$/.test(name))
  assert.equal(lessons.length, 7)
  for (const name of lessons) {
    const source = await readFile(new URL(name, directory), 'utf8')
    const html = markdown.render(source)
    assert.match(html, /language-python/, name)
    assert.match(html, /mjx-container/, name)
    assert.match(html, /<img /, name)
    assert.match(source, /withBase\('\/tutorial-notebooks\/torch\//, name)
    assert.match(source, /Generated from tutorials\/torch/, name)
    const built = await readFile(new URL(`./dist/tutorials/torch/${name.replace(/\.md$/, '.html')}`, import.meta.url), 'utf8')
    assert.match(built, /Download notebook/, name)
    assert.match(built, /tutorial-notebooks\/torch\//, name)
  }
})

test('each Energy lesson publishes both languages and their notebook downloads', async () => {
  const index = await readFile(new URL('./dist/tutorials/energy/index.html', import.meta.url), 'utf8')
  for (const [lesson, output] of [
    ['01_spring_chain', 'Status Converged'],
    ['02_spring_response', 'Maximum finite difference error']
  ]) {
    for (const language of ['python', 'cpp']) {
      const name = `${lesson}_${language}`
      const other = `${lesson}_${language === 'cpp' ? 'python' : 'cpp'}`
      const source = await readFile(new URL(`../tutorials/energy/${name}.md`, import.meta.url), 'utf8')
      const rendered = markdown.render(source)
      assert.match(rendered, new RegExp(`language-${language}`), name)
      assert.match(rendered, /mjx-container/, name)
      assert.ok(rendered.includes(output), name)
      assert.ok(index.includes(`./${name}.html`), name)
      const built = await readFile(new URL(`./dist/tutorials/energy/${name}.html`, import.meta.url), 'utf8')
      assert.ok(built.includes(`./${other}.html`), name)
      assert.ok(built.includes(`/tutorial-notebooks/energy/${name}.ipynb`), name)
      const downloaded = await readFile(new URL(`./dist/tutorial-notebooks/energy/${name}.ipynb`, import.meta.url))
      const original = await readFile(new URL(`../../tutorials/energy/${name}.ipynb`, import.meta.url))
      assert.deepEqual(downloaded, original, name)
    }
  }
})
