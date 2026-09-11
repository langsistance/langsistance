import Taro from '@tarojs/taro'
// 普通相对路径：产物只依赖 tracked 源码。曾用 `#utils/results`，而 package.json
// 把该前缀映射到 gitignore 的 dist-test/（测试编译产物）——干净检出时它不存在，
// 构建会直接 "Module not found"；而只要跑过一次测试，webpack 就会把**上一次
// 测试编译出的旧副本**打进产物。测试侧如何解析 extensionless 见 tsconfig.test.json。
import { ResultsPayload, pruneResults } from '../utils/results'

/**
 * 结果集的内存态 + 持久化。
 *
 * 纯逻辑（解码/取列/裁剪）在 utils/results.ts，这里只做状态与 I/O。
 * storage 走注入而非直接调 Taro —— 否则配额与淘汰逻辑没法在 node 里测。
 */

/** 与 web 同键名（frontend/nextjs/lib/resultsStore.js:11），便于将来对齐。 */
export const STORAGE_KEY = 'copiioai_results'

/**
 * **落盘那层**保留的集数上限。小程序总量 10MB，20 集 × 约 40KB ≈ 800KB，留足余量。web 是 100。
 */
export const MAX_PERSISTED_SETS = 20
/** index 条目上限。与集数同量级即可。web 是 200。 */
export const MAX_INDEX_ENTRIES = 40
/**
 * **内存那层**保留的集数上限。内存里是**未裁剪**的完整载荷（当前会话要用全部行，
 * 见 mem 的注释），比持久化那份（pruneResults 后 ≤MAX_PERSIST_ROWS 行）大得多——
 * 不设上限就是按会话长度线性增长的泄漏，比这个 feature 当初要治的 base64 泄漏更糟。
 * 故给它自己的、比落盘更小的额度：超了就按 savedAt 丢最旧的，与落盘同一条策略。
 * 小程序的真实上限是**单会话**的集数，够一屏回溯即可。
 */
export const MAX_MEMORY_SETS = 8

export interface StorageAdapter {
  getSync(key: string): any
  setSync(key: string, value: any): void
}

export interface IndexEntry {
  setId: string
  sessionId: string
  queryText: string
  savedAt: number
}

interface StoredShape {
  sets: Record<string, ResultsPayload>
  index: IndexEntry[]
}

function emptyShape(): StoredShape {
  return { sets: {}, index: [] }
}

export interface ResultsStore {
  /**
   * 写内存层。`meta.savedAt` 是这集的**时间戳**，内存淘汰按它排序；
   * 缺省 Date.now()。调用方若在流中途就有准确的 savedAt（比如来自持久化
   * 那一份），传进来可以让两层的淘汰顺序一致。
   */
  put(payload: ResultsPayload, meta?: { savedAt?: number }): void
  get(setId: string): ResultsPayload | null
  load(setId: string): ResultsPayload | null
  persist(payload: ResultsPayload, meta: { sessionId: string; queryText: string; savedAt?: number }): void
}

export function createResultsStore(storage: StorageAdapter): ResultsStore {
  // 当前会话的内存态。全量——持久化那份是裁过的。
  const mem = new Map<string, ResultsPayload>()
  // mem 自己的 savedAt 表：Map 的迭代序是插入序，不等于 savedAt 序（同一集被
  // 重写时插入序还不一定是它最后一次写入的时间），故另存一份时间戳来判断谁最旧。
  const memSavedAt = new Map<string, number>()

  /** 写内存层并淘汰。savedAt 必须与写入同一时刻登记，否则刚进来的那集排不到队尾。 */
  function memWrite(payload: ResultsPayload, savedAt: number) {
    mem.set(payload.setId, payload)
    memSavedAt.set(payload.setId, savedAt)
    evictMemory()
  }

  /**
   * 内存层淘汰：与落盘层**同一策略**（按 savedAt 丢最旧的），只是额度不同。
   * 只看 mem/memSavedAt，不看 storage——storage 是另一层，它有自己的淘汰。
   */
  function evictMemory() {
    while (mem.size > MAX_MEMORY_SETS) {
      let oldestId = ''
      let oldestAt = Infinity
      for (const [id, at] of memSavedAt) {
        if (at < oldestAt) {
          oldestAt = at
          oldestId = id
        }
      }
      if (!oldestId) break
      // 刚写入的那集若成了最旧（savedAt 由调用方给），也照丢——不做特例。
      mem.delete(oldestId)
      memSavedAt.delete(oldestId)
    }
  }

  function read(): StoredShape {
    try {
      const raw = storage.getSync(STORAGE_KEY)
      if (!raw || typeof raw !== 'object') return emptyShape()
      return {
        sets: raw.sets && typeof raw.sets === 'object' ? raw.sets : {},
        index: Array.isArray(raw.index) ? raw.index : [],
      }
    } catch {
      return emptyShape()
    }
  }

  return {
    put(payload, meta) {
      memWrite(payload, meta?.savedAt ?? Date.now())
    },

    get(setId) {
      return mem.get(setId) || null
    },

    load(setId) {
      return read().sets[setId] || null
    },

    persist(payload, meta) {
      const savedAt = meta.savedAt ?? Date.now()
      // 内存先写：即使落盘失败，当前会话也要能用
      memWrite(payload, savedAt)
      try {
        const shape = read()
        shape.sets[payload.setId] = pruneResults(payload)
        shape.index = [
          { setId: payload.setId, sessionId: meta.sessionId, queryText: meta.queryText, savedAt },
          ...shape.index.filter((e) => e.setId !== payload.setId),
        ]
        // 淘汰**按 savedAt**，不是对象键序——web 的 dropOldestSet 用的是
        // Object.keys 顺序，与它自己的 index 插入序并不一致，那个缺陷不照搬。
        const byOldest = Object.entries(shape.sets).sort(
          (a, b) => (findSavedAt(shape, a[0]) - findSavedAt(shape, b[0])),
        )
        for (const [id] of byOldest.slice(0, Math.max(0, byOldest.length - MAX_PERSISTED_SETS))) {
          delete shape.sets[id]
        }
        shape.index = shape.index
          .sort((a, b) => b.savedAt - a.savedAt)
          .slice(0, MAX_INDEX_ENTRIES)
        storage.setSync(STORAGE_KEY, shape)
      } catch {
        // 写满/配额超限：静默放弃。结果集是锦上添花，不能打断对话。
      }
    },
  }
}

function findSavedAt(shape: StoredShape, setId: string): number {
  const entry = shape.index.find((e) => e.setId === setId)
  return entry ? entry.savedAt : 0
}

/** 默认实例：接到 Taro 的同步存储。 */
export const resultsStore = createResultsStore({
  getSync: (key) => Taro.getStorageSync(key),
  setSync: (key, value) => Taro.setStorageSync(key, value),
})
