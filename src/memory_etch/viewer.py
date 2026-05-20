#!/usr/bin/env python3
"""Standalone HTTP viewer for Memory Etch.

Provides a web UI at http://127.0.0.1:9120 to browse facts, search,
view timelines, and inspect relations.

Usage:
    python -m memory_etch.viewer [--port 9120] [--db /path/to/memory.db]
"""

import argparse
import http.server
import json
import logging
import os
import sqlite3
import sys
import urllib.parse
from pathlib import Path

from .store import EtchStore

logger = logging.getLogger(__name__)

# ---- HTML Template (single-file SPA with mint design) ----

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Etch Viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:         #050505;
    --fg:         #ffffff;
    --surface:    #0d0d0d;
    --surface2:   #111111;
    --line:       #262626;
    --line-soft:  #1c1c1c;
    --mint:       #70ffd6;
    --mint-dim:   #2fe8b6;
    --mint-glow:  rgba(112,255,214,0.08);
    --muted:      #9b9b9b;
    --muted2:     #666666;
    --red:        #f85149;
    --yellow:     #d29922;
    --green:      #3fb950;
    --font-sans:  'Space Grotesk', system-ui, sans-serif;
    --font-mono:  'DM Mono', 'Courier New', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--fg);
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.5;
    min-height: 100dvh;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    border-bottom: 1px solid var(--line);
  }
  .header h1 {
    font-family: var(--font-sans);
    font-size: 15px;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
    color: var(--fg);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .mint-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--mint);
    flex-shrink: 0;
  }
  .db-path {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted2);
    max-width: 50%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .toolbar {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 10px 20px;
    border-bottom: 1px solid var(--line-soft);
    flex-wrap: wrap;
  }
  .toolbar input[type=text] {
    flex: 1;
    min-width: 180px;
    background: var(--surface);
    border: 1px solid var(--line);
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: 13px;
    padding: 6px 10px;
    outline: none;
    transition: border-color .15s;
  }
  .toolbar input[type=text]:focus { border-color: var(--mint); }
  .toolbar input[type=text]::placeholder { color: var(--muted2); }
  .toolbar select {
    background: var(--surface);
    border: 1px solid var(--line);
    color: var(--muted);
    font-family: var(--font-sans);
    font-size: 12px;
    padding: 6px 8px;
    outline: none;
    cursor: pointer;
    appearance: none;
    padding-right: 20px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='4'%3E%3Cpath d='M0 0l4 4 4-4z' fill='%239b9b9b'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 6px center;
  }
  .toolbar button {
    background: transparent;
    border: 1px solid var(--line);
    color: var(--muted);
    font-family: var(--font-sans);
    font-size: 12px;
    padding: 6px 14px;
    cursor: pointer;
    transition: all .15s;
  }
  .toolbar button:hover { border-color: var(--mint); color: var(--mint); }
  .toolbar button:active { transform: scale(.97); }
  .btn-primary {
    background: var(--mint) !important;
    color: #050505 !important;
    border-color: var(--mint) !important;
    font-weight: 600;
  }
  .btn-primary:hover { background: var(--mint-dim) !important; border-color: var(--mint-dim) !important; }

  .stats {
    display: flex;
    gap: 0;
    padding: 8px 20px;
    border-bottom: 1px solid var(--line-soft);
    font-family: var(--font-mono);
    font-size: 12px;
    flex-wrap: wrap;
  }
  .stat-item { padding: 0 16px; border-right: 1px solid var(--line-soft); }
  .stat-item:first-child { padding-left: 0; }
  .stat-item:last-child { border-right: none; }
  .stat-item .num { color: var(--fg); font-weight: 500; }
  .stat-item .label { color: var(--muted2); margin-left: 6px; }

  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    color: var(--muted2);
    text-transform: uppercase;
    letter-spacing: .08em;
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--line);
    white-space: nowrap;
    user-select: none;
  }
  td { padding: 7px 12px; border-bottom: 1px solid var(--line-soft); color: var(--fg); vertical-align: middle; }
  tr { cursor: pointer; }
  tr:hover td { background: var(--surface); }

  .trust-bar-wrap { display: inline-flex; align-items: center; gap: 6px; }
  .trust-bar { width: 44px; height: 3px; background: var(--line); border-radius: 2px; overflow: hidden; flex-shrink: 0; }
  .trust-fill { height: 100%; border-radius: 2px; transition: width .3s ease; }
  .trust-label { font-family: var(--font-mono); font-size: 10px; color: var(--muted); }

  .tag-wrap { display: inline-flex; flex-wrap: wrap; gap: 0 4px; }
  .tag { font-family: var(--font-mono); font-size: 10px; color: var(--muted); }
  .tag::before { content: '#'; color: var(--muted2); }

  .cat-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 10px;
    border: 1px solid var(--line);
    color: var(--muted);
    padding: 0 6px;
    line-height: 18px;
    letter-spacing: .03em;
  }
  .cat-badge.mint { border-color: var(--mint); color: var(--mint); }
  .cat-badge.green { border-color: var(--green); color: var(--green); }
  .cat-badge.red { border-color: var(--red); color: var(--red); }

  .pagination { display: flex; justify-content: center; align-items: center; gap: 4px; padding: 16px 0; }
  .pagination button {
    background: transparent; border: 1px solid var(--line); color: var(--muted);
    font-family: var(--font-sans); font-size: 12px; padding: 4px 12px;
    cursor: pointer; transition: all .15s;
  }
  .pagination button:hover:not(:disabled) { border-color: var(--mint); color: var(--mint); }
  .pagination button:disabled { opacity: .25; cursor: default; }
  .pagination button:active:not(:disabled) { transform: scale(.96); }
  .page-info { font-family: var(--font-mono); font-size: 11px; color: var(--muted2); padding: 0 8px; }

  #overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.5); z-index: 99; }
  #overlay.show { display: block; }
  .detail-panel {
    position: fixed; top: 0; right: 0; width: 460px; max-width: 100vw; height: 100dvh;
    background: var(--surface); border-left: 1px solid var(--line); overflow-y: auto;
    padding: 24px 20px 40px; transform: translateX(100%); transition: transform .2s ease; z-index: 100;
  }
  .detail-panel.open { transform: translateX(0); }
  .detail-panel .close { float: right; cursor: pointer; font-size: 20px; color: var(--muted2); line-height: 1; }
  .detail-panel .close:hover { color: var(--fg); }
  .detail-panel h2 { font-size: 16px; font-weight: 600; color: var(--fg); margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid var(--line); }
  .detail-panel .field { display: flex; gap: 8px; margin-bottom: 5px; font-size: 13px; line-height: 1.6; }
  .detail-panel .field label { font-family: var(--font-mono); font-size: 10px; color: var(--muted2); text-transform: uppercase; letter-spacing: .06em; min-width: 60px; flex-shrink: 0; padding-top: 1px; }
  .detail-panel .field .val { color: var(--fg); word-break: break-word; font-size: 13px; }
  .detail-panel .section-title { font-family: var(--font-mono); font-size: 10px; color: var(--muted2); text-transform: uppercase; letter-spacing: .08em; margin-top: 16px; margin-bottom: 8px; padding-top: 12px; border-top: 1px solid var(--line-soft); }
  .detail-panel .rel-row { padding: 5px 0; font-size: 12px; border-bottom: 1px solid var(--line-soft); font-family: var(--font-mono); }
  .detail-panel .rel-row:last-child { border-bottom: none; }
  .detail-panel .rel-row .rel-type { display: inline-block; font-size: 9px; padding: 1px 5px; border: 1px solid var(--line); text-transform: uppercase; letter-spacing: .05em; margin-right: 4px; }
  .detail-panel .rel-row .rel-type.err { border-color: var(--red); color: var(--red); }
  .detail-panel .rel-row .rel-type.ok { border-color: var(--green); color: var(--green); }
  .detail-panel .rel-row .rel-type.warn { border-color: var(--yellow); color: var(--yellow); }
  .detail-panel .tl-arrow { color: var(--muted2); margin-right: 4px; }

  .empty { text-align: center; color: var(--muted2); font-family: var(--font-mono); font-size: 12px; padding: 40px 20px; }

  @media (max-width: 640px) {
    .header { flex-direction: column; align-items: flex-start; gap: 4px; padding: 10px 14px; }
    .db-path { max-width: 100%; }
    .toolbar { padding: 8px 14px; }
    .toolbar input[type=text] { min-width: 120px; }
    .stats { padding: 6px 14px; }
    .stat-item { padding: 0 10px; }
    .stat-item:first-child { padding-left: 0; }
    th, td { padding: 5px 8px; font-size: 12px; }
    .detail-panel { width: 100vw; padding: 16px; }
  }
</style>
</head>
<body>

<div class="header">
  <h1><span class="mint-dot"></span>Etch Viewer</h1>
  <span class="db-path" id="db-label"></span>
</div>

<div class="toolbar">
  <select id="cat-filter" onchange="loadFacts()">
    <option value="">All categories</option>
  </select>
  <select id="proj-filter" onchange="loadFacts()">
    <option value="">All projects</option>
  </select>
  <input type="text" id="search-q" placeholder="Search facts..." onkeydown="if(event.key==='Enter')loadSearch()">
  <button class="btn-primary" onclick="loadSearch()">Search</button>
  <button onclick="loadFacts()">Clear</button>
</div>

<div class="stats" id="stats"></div>

<div class="table-wrap">
  <table><thead><tr>
    <th>ID</th><th>Content</th><th>Tags</th><th>Trust</th><th>Updated</th>
  </tr></thead><tbody id="facts-tbody"></tbody></table>
</div>

<div class="pagination" id="pagination"></div>

<div id="overlay" onclick="closeDetail()"></div>
<div class="detail-panel" id="detail-panel">
  <span class="close" onclick="closeDetail()">&times;</span>
  <div id="detail-content"></div>
</div>

<script>
const API = '';
let currentPage = 0;
const LIMIT = 50;

const CAT_COLORS = { project: 'mint', user_pref: 'green', tech: 'mint', tool: 'green', general: '' };

(async () => {
  try {
    const r = await fetch('/api/db');
    const d = await r.json();
    document.getElementById('db-label').textContent = d.path;
  } catch(e) {}
  const r = await fetch('/api/projects');
  const projs = await r.json();
  const sel = document.getElementById('proj-filter');
  (projs.projects||[]).forEach(p => {
    const opt = document.createElement('option');
    opt.value = p; opt.textContent = p; sel.appendChild(opt);
  });
  loadFacts();
  loadStats();
})();

async function apiGet(path) {
  const r = await fetch(API + path);
  return r.json();
}

async function loadStats() {
  const s = await apiGet('/api/stats');
  document.getElementById('stats').innerHTML = `
    <span class="stat-item"><span class="num">${s.fact_count}</span><span class="label">Facts</span></span>
    <span class="stat-item"><span class="num">${s.session_count}</span><span class="label">Sessions</span></span>
    <span class="stat-item"><span class="num">${s.relation_count}</span><span class="label">Relations</span></span>
    <span class="stat-item"><span class="num">${s.extraction_count}</span><span class="label">Extractions</span></span>
    <span class="stat-item"><span class="num">${s.active_sessions}</span><span class="label">Active</span></span>
  `;
}

async function loadFacts(page) {
  currentPage = page !== undefined ? page : currentPage;
  const cat = document.getElementById('cat-filter').value;
  const proj = document.getElementById('proj-filter').value;
  let url = `/api/facts?limit=${LIMIT}&offset=${currentPage * LIMIT}`;
  if (cat) url += `&category=${cat}`;
  if (proj) url += `&project=${proj}`;
  const data = await apiGet(url);
  renderTable(data.facts || []);
  renderPagination(data.count || 0);
}

async function loadSearch() {
  const q = document.getElementById('search-q').value.trim();
  if (!q) return loadFacts(0);
  currentPage = 0;
  const data = await apiGet(`/api/search?q=${encodeURIComponent(q)}`);
  renderTable(data.results || []);
  document.getElementById('pagination').innerHTML = '';
}

function renderTable(facts) {
  const tbody = document.getElementById('facts-tbody');
  if (!facts.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No facts found</td></tr>';
    return;
  }
  tbody.innerHTML = facts.map(f => {
    const trust = Math.round(f.trust_score * 100);
    const color = trust > 70 ? 'var(--green)' : trust > 40 ? 'var(--yellow)' : 'var(--red)';
    const tags = (f.tags||'').split(',').filter(Boolean).map(t => `<span class="tag">${esc(t.trim())}</span>`).join('');
    const updated = (f.updated_at||'').slice(0,16).replace('T',' ');
    return `<tr onclick="openDetail(${f.fact_id})">
      <td style="font-family:var(--font-mono);font-size:11px;color:var(--muted2)">${f.fact_id}</td>
      <td style="max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(f.content||'').slice(0,120)}${(f.content||'').length>120?'...':''}</td>
      <td><span class="tag-wrap">${tags}</span></td>
      <td><span class="trust-bar-wrap"><span class="trust-bar"><span class="trust-fill" style="width:${trust}%;background:${color}"></span></span><span class="trust-label">${trust}%</span></span></td>
      <td style="font-family:var(--font-mono);font-size:11px;color:var(--muted)">${updated}</td>
    </tr>`;
  }).join('');
}

function renderPagination(total) {
  const pages = Math.ceil(total / LIMIT);
  const el = document.getElementById('pagination');
  if (pages <= 1) { el.innerHTML = ''; return; }
  let h = `<button onclick="loadFacts(${currentPage-1})" ${currentPage<=0?'disabled':''}>←</button>`;
  h += `<span class="page-info">${currentPage+1} / ${pages}</span>`;
  h += `<button onclick="loadFacts(${currentPage+1})" ${currentPage>=pages-1?'disabled':''}>→</button>`;
  el.innerHTML = h;
}

async function openDetail(factId) {
  try {
    const fact = await apiGet(`/api/facts/${factId}`);
    const rels = await apiGet(`/api/relations/${factId}`);
    const tl = await apiGet(`/api/timeline/${factId}?before=3&after=3`);
    renderDetail(fact, rels.relations||[], tl);
    document.getElementById('detail-panel').classList.add('open');
    document.getElementById('overlay').classList.add('show');
  } catch(e) { console.error(e); }
}

function renderDetail(fact, relations, tl) {
  const tags = (fact.tags||'').split(',').filter(Boolean).map(t => `<span class="tag">${esc(t.trim())}</span>`).join(' ');
  const relHtml = relations.map(r => {
    const cls = r.relation_type === 'conflicts_with' ? 'err' : r.relation_type === 'compatible' ? 'ok' : 'warn';
    return `<div class="rel-row"><span class="rel-type ${cls}">${esc(r.relation_type)}</span> → #${r.other_fact_id} <span style="color:var(--muted2)">(${Math.round(r.confidence*100)}%, ${esc(r.judged_by||'auto')})</span></div>`;
  }).join('') || '<div class="empty" style="padding:10px">No relations</div>';

  const tlHtml = () => {
    let h = '<div class="section-title">Timeline</div>';
    if (tl.before && tl.before.length) {
      tl.before.slice().reverse().forEach(f => { h += `<div class="rel-row" style="color:var(--muted)"><span class="tl-arrow">←</span>${esc(f.content||'').slice(0,80)}</div>`; });
    }
    h += `<div class="rel-row" style="color:var(--mint);font-weight:600">${esc(fact.content||'').slice(0,100)}</div>`;
    if (tl.after && tl.after.length) {
      tl.after.forEach(f => { h += `<div class="rel-row"><span class="tl-arrow">→</span>${esc(f.content||'').slice(0,80)}</div>`; });
    }
    if (!(tl.before||[]).length && !(tl.after||[]).length) h += '<div class="empty" style="padding:10px">No session timeline</div>';
    return h;
  };

  const catCls = CAT_COLORS[fact.category] || '';
  document.getElementById('detail-content').innerHTML = `
    <h2>#${fact.fact_id} — ${esc(fact.content||'').slice(0,60)}</h2>
    <div class="field"><label>Category</label><div class="val"><span class="cat-badge ${catCls}">${esc(fact.category||'-')}</span></div></div>
    <div class="field"><label>Trust</label><div class="val" style="font-family:var(--font-mono)">${Math.round(fact.trust_score*100)}%</div></div>
    <div class="field"><label>Tags</label><div class="val">${tags}</div></div>
    <div class="field"><label>Project</label><div class="val" style="font-family:var(--font-mono);color:var(--muted)">${esc(fact.project||'-')}</div></div>
    <div class="field"><label>Topic</label><div class="val" style="font-family:var(--font-mono);color:var(--muted)">${esc(fact.topic_key||'-')}</div></div>
    <div class="field"><label>Revisions</label><div class="val" style="font-family:var(--font-mono)">${fact.revision_count||0}</div></div>
    <div class="field"><label>Created</label><div class="val" style="font-family:var(--font-mono);font-size:12px;color:var(--muted)">${fact.created_at||'-'}</div></div>
    <div class="field"><label>Updated</label><div class="val" style="font-family:var(--font-mono);font-size:12px;color:var(--muted)">${fact.updated_at||'-'}</div></div>
    <div class="field"><label>Session</label><div class="val" style="font-family:var(--font-mono);font-size:12px;color:var(--muted2)">${esc(fact.session_id||'-')}</div></div>
    <div class="section-title">Relations (${relations.length})</div>
    ${relHtml}
    ${tlHtml()}
  `;
}

function closeDetail() {
  document.getElementById('detail-panel').classList.remove('open');
  document.getElementById('overlay').classList.remove('show');
}

function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
</script>
</body>
</html>"""

# ---- HTTP Request Handler ----


class ViewerHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that serves the HTML page + REST API."""

    def __init__(self, *args, db_path=None, **kwargs):
        self._db_path = db_path
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "" or path == "/":
                self._serve_html()
            elif path.startswith("/api/"):
                self._serve_api(path, qs)
            else:
                self._json_error(404, "Not found")
        except Exception as e:
            self._json_error(500, str(e))

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def _serve_api(self, path: str, qs: dict):
        db: sqlite3.Connection = self.server._db

        if path == "/api/db":
            self._json({"path": str(self.server._db_path)})

        elif path == "/api/stats":
            def _safe_count(tbl: str, where: str = "", params: tuple = ()) -> int:
                try:
                    return db.execute(f"SELECT COUNT(*) FROM {tbl} {where}", params).fetchone()[0]
                except Exception:
                    return 0
            facts = _safe_count("facts", "WHERE (deleted IS NULL OR deleted = 0)")
            sessions = _safe_count("sessions")
            relations = _safe_count("fact_relations")
            extractions = _safe_count("extractions")
            active = _safe_count("sessions", "WHERE status='active'")
            self._json({
                "fact_count": facts, "session_count": sessions, "relation_count": relations,
                "extraction_count": extractions, "active_sessions": active,
            })

        elif path == "/api/projects":
            try:
                rows = db.execute(
                    "SELECT DISTINCT project FROM facts WHERE project != '' AND (deleted IS NULL OR deleted = 0) ORDER BY project"
                ).fetchall()
                self._json({"projects": [r[0] for r in rows]})
            except Exception:
                self._json({"projects": []})

        elif path == "/api/facts":
            limit = min(int(qs.get("limit", ["50"])[0]), 200)
            offset = int(qs.get("offset", ["0"])[0])
            where = ["1=1 AND (deleted IS NULL OR deleted = 0)"]
            params = []
            cat = qs.get("category", [None])[0]
            if cat:
                where.append("category = ?")
                params.append(cat)
            proj = qs.get("project", [None])[0]
            if proj:
                where.append("project = ?")
                params.append(proj)
            w = " AND ".join(where)
            total = db.execute(f"SELECT COUNT(*) FROM facts WHERE {w}", params).fetchone()[0]
            rows = db.execute(
                "SELECT fact_id, content, category, tags, trust_score, project, "
                "created_at, updated_at, session_id, topic_key, revision_count, importance "
                f"FROM facts WHERE {w} ORDER BY trust_score DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            self._json({"facts": [dict(r) for r in rows], "count": total})

        elif path.startswith("/api/facts/"):
            try:
                fid = int(path.split("/")[-1])
            except (ValueError, IndexError):
                self._json_error(400, "Invalid fact_id")
                return
            row = db.execute(
                "SELECT fact_id, content, category, tags, trust_score, retrieval_count, "
                "helpful_count, created_at, updated_at, session_id, topic_key, revision_count, project, importance "
                "FROM facts WHERE fact_id = ?", (fid,)
            ).fetchone()
            if row is None:
                self._json_error(404, f"Fact #{fid} not found")
                return
            d = dict(row)
            self._json(d)

        elif path == "/api/search":
            query = qs.get("q", [""])[0]
            if not query:
                self._json({"results": [], "count": 0})
                return
            limit = min(int(qs.get("limit", ["50"])[0]), 200)
            try:
                rows = db.execute(
                    """SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score, f.created_at, f.updated_at
                        FROM facts f
                        JOIN facts_fts fts ON fts.rowid = f.fact_id
                        WHERE (f.deleted IS NULL OR f.deleted = 0) AND facts_fts MATCH ?
                        ORDER BY fts.rank
                        LIMIT ?""",
                    (query, limit),
                ).fetchall()
                self._json({"results": [dict(r) for r in rows], "count": len(rows)})
            except Exception:
                self._json({"results": [], "count": 0, "error": "Search unavailable"})

        elif path.startswith("/api/relations/"):
            try:
                fid = int(path.split("/")[-1])
            except (ValueError, IndexError):
                self._json_error(400, "Invalid fact_id")
                return
            tables = [r["name"] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if "fact_relations" not in tables:
                self._json({"relations": [], "count": 0})
                return
            rows = db.execute(
                """SELECT r.relation_id,
                          CASE WHEN r.fact_id_a = ? THEN r.fact_id_b ELSE r.fact_id_a END AS other_fact_id,
                          r.relation_type, r.confidence, r.judged_by, r.created_at
                   FROM fact_relations r
                   WHERE r.fact_id_a = ? OR r.fact_id_b = ?
                   ORDER BY r.created_at DESC""",
                (fid, fid, fid),
            ).fetchall()
            self._json({"relations": [dict(r) for r in rows], "count": len(rows)})

        elif path.startswith("/api/timeline/"):
            try:
                fid = int(path.split("/")[-1])
            except (ValueError, IndexError):
                self._json_error(400, "Invalid fact_id")
                return
            before = int(qs.get("before", ["5"])[0])
            after = int(qs.get("after", ["5"])[0])
            anchor = db.execute(
                "SELECT fact_id, content, session_id FROM facts WHERE fact_id = ?", (fid,)
            ).fetchone()
            if anchor is None:
                self._json_error(404, f"Fact #{fid} not found")
                return
            session_id = anchor["session_id"] or ""
            b4, aft = [], []
            if session_id:
                try:
                    b4 = db.execute(
                        "SELECT fact_id, content, category, tags, trust_score, created_at FROM facts "
                        "WHERE fact_id < ? AND session_id = ? ORDER BY fact_id DESC LIMIT ?",
                        (fid, session_id, before),
                    ).fetchall()
                    aft = db.execute(
                        "SELECT fact_id, content, category, tags, trust_score, created_at FROM facts "
                        "WHERE fact_id > ? AND session_id = ? ORDER BY fact_id ASC LIMIT ?",
                        (fid, session_id, after),
                    ).fetchall()
                except Exception:
                    pass
            self._json({
                "fact": {"fact_id": anchor["fact_id"], "content": anchor["content"], "session_id": session_id},
                "before": [dict(r) for r in b4],
                "after": [dict(r) for r in aft],
            })
        else:
            self._json_error(404, "Unknown API endpoint")

    def _json(self, data: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode("utf-8"))

    def _json_error(self, code: int, msg: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"error": msg}).encode("utf-8"))

    def log_message(self, fmt, *args):
        logger = logging.getLogger(__name__)
        logger.info(
            "%s - %s", self.log_date_time_string(), fmt % args if args else str(fmt)
        )


# ---- Server ----


class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    """Threaded HTTP server with reference to the DB connection."""
    allow_reuse_address = True
    daemon_threads = True


def find_db_path() -> str:
    """Auto-detect the etch database path."""
    env = os.environ.get("MEMORY_ETCH_DB")
    if env:
        return env
    default = Path.home() / ".etch" / "memory.db"
    if default.exists():
        return str(default)
    return str(default)


def create_viewer_server(
    db_path: str,
    host: str = "127.0.0.1",
    port: int = 9120,
) -> ThreadedHTTPServer:
    """Create a configured viewer server."""
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    server = ThreadedHTTPServer((host, port), ViewerHandler)
    server._db = conn
    server._db_path = db_path
    return server


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Memory Etch Viewer")
    ap.add_argument("--port", "-p", type=int, default=9120, help="Port (default: 9120)")
    ap.add_argument("--db", "-d", default=None, help="Path to etch memory.db")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    args = ap.parse_args()

    db_path = args.db or find_db_path()
    if not Path(db_path).exists():
        logger.error("Database not found at %s", db_path)
        logger.error("Specify path with --db or set MEMORY_ETCH_DB env var")
        sys.exit(1)

    server = create_viewer_server(db_path, args.host, args.port)

    logger.info("Memory Etch Viewer at http://%s:%s", args.host, args.port)
    logger.info("Database: %s", db_path)
    logger.info("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server._db.close()
        server.server_close()


if __name__ == "__main__":
    main()
