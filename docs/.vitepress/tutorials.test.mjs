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
  assert.equal(lessons.length, 6)
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
