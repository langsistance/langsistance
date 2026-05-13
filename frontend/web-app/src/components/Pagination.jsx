function getPageNums(current, total) {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1)
  if (current <= 4) return [1, 2, 3, 4, 5, '…', total]
  if (current >= total - 3) return [1, '…', total - 4, total - 3, total - 2, total - 1, total]
  return [1, '…', current - 1, current, current + 1, '…', total]
}

export default function Pagination({ page, totalPages, onChange }) {
  if (totalPages === 0) return null
  const pages = getPageNums(page, totalPages)
  return (
    <div className="pagination-container">
      <button
        className={`page-btn${page === 1 ? ' disabled' : ''}`}
        onClick={() => page > 1 && onChange(page - 1)}
        disabled={page === 1}
      >‹</button>
      {pages.map((p, i) =>
        p === '…'
          ? <span key={`e${i}`} className="page-ellipsis">…</span>
          : <button
              key={p}
              className={`page-btn${p === page ? ' active' : ''}`}
              onClick={() => p !== page && onChange(p)}
            >{p}</button>
      )}
      <button
        className={`page-btn${page === totalPages ? ' disabled' : ''}`}
        onClick={() => page < totalPages && onChange(page + 1)}
        disabled={page === totalPages}
      >›</button>
    </div>
  )
}
