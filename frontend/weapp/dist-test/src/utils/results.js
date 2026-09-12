"use strict";
/**
 * 结果集的纯函数层：解码、按 role 取列、拼 meta、裁剪。
 *
 * 这一层没有 I/O、没有平台 API，所以能真测——而它出的错**不会报错**，
 * 只会安静地显示错数据（少了 url 列就点不开说明书，meta 顺序错了就是
 * 字段对不上号）。故与 store 分开：纯逻辑在此，I/O 在 services/resultsStore.ts。
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.META_ROLES = exports.MAX_PERSIST_ABSTRACT_CHARS = exports.MAX_PERSIST_ROWS = void 0;
exports.decodeArtifactChunks = decodeArtifactChunks;
exports.pickColumn = pickColumn;
exports.metaLine = metaLine;
exports.pruneResults = pruneResults;
/** 每集保留的行数上限。小程序单键 1MB（浏览器无此限制），故比 web 的 50 紧。 */
exports.MAX_PERSIST_ROWS = 40;
/** 摘要截断。它通常是最长字段。web 是 500。 */
exports.MAX_PERSIST_ABSTRACT_CHARS = 400;
/**
 * meta 行的 role 与顺序，对齐 web 的 frontend/nextjs/lib/results.js:34。
 * 只收录**有值**的。
 */
exports.META_ROLES = [
    'patent_id',
    'application_number',
    'publication_number',
    'assignee',
    'publication_date',
];
/**
 * 逐 chunk 独立解 base64 再合并字节。
 *
 * **不能先拼字符串再解**：后端按 256KB 切片（sse_callback.py:11），分片边界
 * 会切断 base64 三元组，拼起来解就是坏的。web 在 chatSession.js:134-161
 * 踩过这个坑，注释还在。
 *
 * 任何一步失败返回 null（对齐 web 的静默跳过，不打断对话）。
 */
function decodeArtifactChunks(chunks) {
    if (!Array.isArray(chunks) || chunks.length === 0)
        return null;
    try {
        // 微信小程序无 Buffer/atob —— 用 base64 字符表手写解码，两处运行时通用
        const BYTES = new Uint8Array(totalBytes(chunks));
        let offset = 0;
        for (const chunk of chunks) {
            const bytes = base64ToBytes(chunk);
            if (!bytes)
                return null;
            BYTES.set(bytes, offset);
            offset += bytes.length;
        }
        const text = utf8Decode(BYTES);
        const parsed = JSON.parse(text);
        if (!parsed || !Array.isArray(parsed.rows))
            return null;
        return {
            setId: String(parsed.setId || ''),
            source: String(parsed.source || 'uspto'),
            columns: Array.isArray(parsed.columns) ? parsed.columns : [],
            rows: parsed.rows,
        };
    }
    catch {
        return null;
    }
}
const B64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
/**
 * 取 base64 字符的 6 bit 值。`=`（以及任何未知字符）一律归 0。
 *
 * **不能直接 indexOf 再使用**：`=` 不在字符表里，indexOf 返回 -1，
 * 而 -1 是全 1 位模式——`(c0 << 18) | ... | -1` 恒为 -1，三个字节全成 0xff。
 * 于是**每一组带 padding 的字符都会解出 [255,255,255]**，subarray 裁完
 * 这些垃圾字节照样留在结果里。0 才是 padding 应得的位模式。
 */
function b64(ch) {
    const i = B64_CHARS.indexOf(ch);
    return i < 0 ? 0 : i;
}
function base64ToBytes(input) {
    const clean = input.replace(/[\r\n\s]/g, '');
    if (!clean)
        return null;
    let len = clean.length;
    while (len % 4 !== 0)
        len++;
    const out = new Uint8Array((len / 4) * 3);
    let o = 0;
    for (let i = 0; i < len; i += 4) {
        const c0 = b64(clean[i] || '=');
        const c1 = b64(clean[i + 1] || '=');
        const c2 = b64(clean[i + 2] || '=');
        const c3 = b64(clean[i + 3] || '=');
        const n = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3;
        out[o++] = (n >> 16) & 0xff;
        out[o++] = (n >> 8) & 0xff;
        out[o++] = n & 0xff;
    }
    const pad = (clean.match(/=+$/) || [''])[0].length;
    return pad ? out.subarray(0, out.length - pad) : out;
}
function totalBytes(chunks) {
    let n = 0;
    for (const c of chunks) {
        const clean = (c || '').replace(/[\r\n\s]/g, '');
        const pad = (clean.match(/=+$/) || [''])[0].length;
        n += (clean.length / 4) * 3 - pad;
    }
    return n;
}
/** UTF-8 解码。优先用 TextDecoder，缺失时走手写回退（小程序基础库较全，一般走前者）。 */
function utf8Decode(bytes) {
    // eslint-disable-next-line no-undef
    if (typeof TextDecoder !== 'undefined') {
        // eslint-disable-next-line no-undef
        return new TextDecoder('utf-8').decode(bytes);
    }
    let out = '';
    let i = 0;
    while (i < bytes.length) {
        const b = bytes[i];
        if (b < 0x80) {
            out += String.fromCharCode(b);
            i += 1;
        }
        else if (b < 0xe0) {
            out += String.fromCharCode(((b & 0x1f) << 6) | (bytes[i + 1] & 0x3f));
            i += 2;
        }
        else if (b < 0xf0) {
            out += String.fromCharCode(((b & 0x0f) << 12) | ((bytes[i + 1] & 0x3f) << 6) | (bytes[i + 2] & 0x3f));
            i += 3;
        }
        else {
            const cp = ((b & 0x07) << 18) |
                ((bytes[i + 1] & 0x3f) << 12) |
                ((bytes[i + 2] & 0x3f) << 6) |
                (bytes[i + 3] & 0x3f);
            const s = cp - 0x10000;
            out += String.fromCharCode(0xd800 + (s >> 10), 0xdc00 + (s & 0x3ff));
            i += 4;
        }
    }
    return out;
}
/** 按 role 取该行对应列的值；无此 role 或值为空返回 ''。 */
function pickColumn(payload, role, row) {
    const col = payload.columns.find((c) => c.role === role);
    if (!col)
        return '';
    return String(row[col.key] ?? '');
}
/** 拼 meta 行：只收录 META_ROLES 中**有值**的项，顺序固定，用 ' · ' 连接。 */
function metaLine(payload, row) {
    const out = [];
    for (const role of exports.META_ROLES) {
        const v = pickColumn(payload, role, row);
        if (v)
            out.push(v);
    }
    return out;
}
/**
 * 裁剪成可持久化的副本：行数、摘要长度、以及列。
 *
 * `url` 列**必须保留**——说明书 tab 从它取 PDF 直链，裁掉就点不开了。
 * 其余 role 为 text 的列（后端会塞一个行级 source 列）也保留，否则
 * 行级 source 丢失会导致详情接口传错 source。
 */
function pruneResults(payload) {
    const abstractCol = payload.columns.find((c) => c.role === 'abstract');
    const columns = payload.columns.filter((c) => c.role !== 'text' || c.key === 'source');
    const rows = payload.rows.slice(0, exports.MAX_PERSIST_ROWS).map((row) => {
        const next = { ...row };
        if (abstractCol) {
            const v = next[abstractCol.key];
            if (typeof v === 'string' && v.length > exports.MAX_PERSIST_ABSTRACT_CHARS) {
                next[abstractCol.key] = v.slice(0, exports.MAX_PERSIST_ABSTRACT_CHARS);
            }
        }
        return next;
    });
    return {
        setId: payload.setId,
        source: payload.source,
        columns,
        rows,
    };
}
//# sourceMappingURL=results.js.map