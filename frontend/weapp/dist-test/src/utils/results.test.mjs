import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeArtifactChunks, pickColumn, metaLine, pruneResults, MAX_PERSIST_ROWS, MAX_PERSIST_ABSTRACT_CHARS, } from './results.js';
/** 造一份 payload，字段名贴近真实导出（result_export.py 的扁平键） */
function payload(overrides = {}) {
    return {
        setId: 'set-1',
        source: 'uspto',
        columns: [
            { key: 'patentTitle', label: '标题', role: 'title' },
            { key: 'patentNumber', label: '专利号', role: 'patent_id' },
            { key: 'assigneeEntityName', label: '申请人', role: 'assignee' },
            { key: 'abstractText', label: '摘要', role: 'abstract' },
            { key: 'downloadUrl', label: '下载链接', role: 'url' },
        ],
        rows: [{ patentTitle: 'A', patentNumber: 'US1', assigneeEntityName: 'X', abstractText: 'a', downloadUrl: 'u' }],
        ...overrides,
    };
}
const b64 = (s) => Buffer.from(s, 'utf8').toString('base64');
test('decodeArtifactChunks 逐 chunk 独立解码——拼字符串再解会在分片边界解错', () => {
    // 分片边界必须**真的**切在一个多字节字符中间，否则这条测试是假的：
    // 若边界落在 ASCII 上，把两块 base64 各自解成字符串再拼接也能得到正确结果，
    // 「拆成字节再合并」与「先拼字符串再解」两种实现无法区分。
    //
    // 这份 payload 的第一个多字节字符是 columns[0].label 的「标」，
    // 占 UTF-8 的 bytes[75..77]（e6 a0 87）。故取 mid = 76：
    // 前半块以 e6 结尾、后半块以 a0 87 开头，「标」被劈成两半。
    // 此时拼字符串再解会得到 U+FFFD 替换符，JSON 解析必失败。
    const text = JSON.stringify(payload());
    const bytes = Buffer.from(text, 'utf8');
    assert.equal(bytes[75], 0xe6, '前置断言：byte 75 应是「标」的首字节，否则 mid 落点已漂移');
    const mid = 76;
    assert.ok(bytes[mid] >= 0x80 && bytes[mid - 1] >= 0x80, '前置断言：mid 必须落在多字节字符内部');
    const chunks = [
        bytes.subarray(0, mid).toString('base64'),
        bytes.subarray(mid).toString('base64'),
    ];
    const out = decodeArtifactChunks(chunks);
    assert.equal(out.setId, 'set-1');
    assert.equal(out.rows.length, 1);
    // 被劈开的那个字符也必须完好还原——只断言 setId/rows 不够，
    // 它们都在被劈位置之前/之后，拼错也可能侥幸通过。
    assert.equal(out.columns[0].label, '标题');
});
test('decodeArtifactChunks 载荷非法时返回 null，不抛', () => {
    assert.equal(decodeArtifactChunks([]), null);
    assert.equal(decodeArtifactChunks([b64('not json')]), null);
    assert.equal(decodeArtifactChunks([b64('{"rows":"x"}')]), null); // rows 非数组
});
test('pickColumn 按 role 取值，无匹配返回空串', () => {
    const p = payload();
    assert.equal(pickColumn(p, 'title', p.rows[0]), 'A');
    assert.equal(pickColumn(p, 'patent_id', p.rows[0]), 'US1');
    assert.equal(pickColumn(p, 'nope', p.rows[0]), '');
});
test('metaLine 只收录有值的 role，顺序固定，用 · 连接', () => {
    const p = payload();
    const withAll = {
        ...p.rows[0],
        applicationNumberText: '17638216',
        publicationNumber: 'US20220294065A1',
    };
    p.columns = [
        ...p.columns,
        { key: 'applicationNumberText', label: '申请号', role: 'application_number' },
        { key: 'publicationNumber', label: '公开号', role: 'publication_number' },
    ];
    // patent_id → application_number → publication_number → assignee → publication_date
    assert.deepEqual(metaLine(p, withAll), ['US1', '17638216', 'US20220294065A1', 'X']);
    // 空值不进 meta
    assert.deepEqual(metaLine(p, { ...p.rows[0], assigneeEntityName: '' }), ['US1']);
});
test('pruneResults 裁行数、截摘要、且绝不误删 url 列', () => {
    const p = payload();
    p.rows = Array.from({ length: 100 }, (_, i) => ({
        patentTitle: `T${i}`, patentNumber: `US${i}`,
        assigneeEntityName: 'X', abstractText: 'a'.repeat(2000), downloadUrl: `u${i}`,
    }));
    const out = pruneResults(p);
    assert.equal(out.rows.length, MAX_PERSIST_ROWS);
    assert.equal(out.rows.length, 40);
    assert.equal(out.rows[0].abstractText.length, MAX_PERSIST_ABSTRACT_CHARS);
    assert.equal(out.rows[0].abstractText.length, 400);
    // url 列必须留着——说明书 tab 要用
    const roles = out.columns.map((c) => c.role);
    assert.ok(roles.includes('url'), 'url 列被裁掉了，说明书 tab 会失效');
    assert.ok(roles.includes('abstract'));
    // 前 40 行按原顺序保留
    assert.equal(out.rows[0].patentTitle, 'T0');
    assert.equal(out.rows[39].patentTitle, 'T39');
});
test('pruneResults 不动 setId/source/columns 的非摘要部分', () => {
    const p = payload();
    const out = pruneResults(p);
    assert.equal(out.setId, 'set-1');
    assert.equal(out.source, 'uspto');
    assert.equal(out.columns.length, p.columns.length);
});
//# sourceMappingURL=results.test.mjs.map