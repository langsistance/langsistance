import test from 'node:test'
import assert from 'node:assert/strict'
import { SHARE_PATH, SHARE_TITLE, buildShareCard } from './share.js'

test('载荷字段固定：只有 title 与 imageUrl', () => {
  // 修复前的 bug：title 拼进 sessionTitle，把用户首问原文带到了群聊卡片上。
  // 这条钉住"不夹带字段"——将来若有人把会话数据塞进载荷，key 集合会变，测试红。
  assert.deepEqual(Object.keys(buildShareCard('x')).sort(), ['imageUrl', 'title'])
})

test('标题恒为品牌文案，与会话无关', () => {
  assert.equal(buildShareCard('x').title, SHARE_TITLE)
  assert.equal(SHARE_TITLE, 'CopiioAI 专利情报，一问即得')
})

test('imageUrl 原样透传调用方传入的配图路径', () => {
  assert.equal(buildShareCard('/assets/share-card.png').imageUrl, '/assets/share-card.png')
})

test('path 指向落地页', () => {
  assert.equal(SHARE_PATH, '/pages/chat/index')
})
