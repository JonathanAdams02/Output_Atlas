#!/usr/bin/env python3
"""
build.py — generate a self-contained atlas.html for the Research Atlas.

Pipeline: fetch OpenAlex → embed → UMAP 3D → HDBSCAN → TF-IDF labels → write HTML

The output is a single atlas.html: plain HTML/CSS/JS with Three.js bundled inline.
No server needed — just double-click to open.
"""

import json, re, time, urllib.parse, urllib.request
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer

# ── CONFIG ────────────────────────────────────────────────────────────────────
DIMS   = 3
MAILTO = "jonathan.adams@kuleuven.be"

AUTHOR_IDS     = ["A5052009397","A5073952660","A5046990645","A5062466055","A5009995930","A5004036435","A5037642793","A5087545626","a5004439858","a5029582591","a5118991313","a5075193529"]##Jan,matthieu, doga, maartem, louise, chi-hao, francois laurent, thomas, filip, marta bono, robbe decloedt, laurent mertens  # your PIs' OpenAlex author IDs

INSTITUTION_ID = None

OUT_PATH = Path(__file__).resolve().parent / "atlas.html"

EMBED_MODEL      = "all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE = 15   # now actually wired into HDBSCAN below — lower = more, smaller clusters
RANDOM_STATE     = 42

# NOTE: this is the single source of truth for cluster colors — it gets
# injected into the HTML template at build time (see write_atlas()).
# Previously there was a *second*, hardcoded copy of this list inside the
# JS template that actually controlled rendering, so editing this list did
# nothing. That's fixed now: the JS reads __CLUSTER_COLORS__ instead.
CLUSTER_COLORS = [
    '#c2613f', '#c0902e', '#5a9e57', '#2f9ea0',
    '#3f74c0', '#7b6bcf', '#b057a8', '#b8506b',
    '#d97b4f', '#8a9e3f', '#3f9e7a', '#4f8fc0',
    '#9a5fc0', '#c05f8f', '#6b7b3f', '#3f6b9e',
]

# Filled during normalize() — only the display names of the PIs
PI_AUTHOR_NAMES = set()

# ── FETCH THREE.JS LIBS ───────────────────────────────────────────────────────
def fetch_js_libraries():
    """Download Three.js and OrbitControls to embed directly in the HTML."""
    print("Fetching Three.js libraries for bundling…")
    three_url  = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
    orbit_url  = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"
    with urllib.request.urlopen(three_url) as r:
        three_js = r.read().decode("utf-8")
    with urllib.request.urlopen(orbit_url) as r:
        orbit_js = r.read().decode("utf-8")
    print("  Three.js libraries fetched.")
    return three_js, orbit_js

# ── HTML TEMPLATE ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research Atlas 3D</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#fff;font-family:'IBM Plex Sans',system-ui,sans-serif;color:#1c1c1c;overflow:hidden}

/* ── layout ── */
#app{display:flex;height:100vh;width:100%}

/* ── sidebar ── */
#sidebar{width:312px;flex:none;height:100%;border-right:1px solid #ececea;display:flex;flex-direction:column;background:#fcfcfb}
#sb-head{padding:24px 22px 18px;border-bottom:1px solid #ececea}
#sb-lab{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:#9a9a93;margin-bottom:8px}
#sb-title{font-size:20px;font-weight:600;letter-spacing:-.01em}
#sb-title span{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#b9b9b2;font-weight:400;vertical-align:middle;margin-left:4px}
#sb-desc{font-size:12.5px;color:#76766f;margin-top:5px;line-height:1.45}
#sb-body{flex:1;overflow-y:auto;padding:20px 22px 22px;display:flex;flex-direction:column;gap:24px}
#sb-foot{padding:14px 22px;border-top:1px solid #ececea;display:flex;justify-content:space-between;align-items:center;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#9a9a93}
#sb-foot #count-vis{color:#3a3a36}#sb-foot #count-sep{color:#b6b6ae}

/* ── sidebar controls ── */
.ctrl-label{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:#9a9a93;display:block;margin-bottom:8px}
#search{width:100%;padding:9px 11px;border:1px solid #e0e0db;border-radius:8px;font-family:inherit;font-size:13px;color:#1c1c1c;background:#fff;outline:none}
#search:focus{border-color:#9a9a93}
.year-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.year-row .ctrl-label{margin-bottom:0}
#year-display{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#3a3a36}
input[type=range]{accent-color:#2d2d2d;width:100%;height:3px;display:block}
input[type=range]+input[type=range]{margin-top:6px}
.topic-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
#reset-btn{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#9a9a93;background:transparent;border:none;cursor:pointer;padding:2px 0}
#reset-btn:hover{color:#1c1c1c}
#cluster-list{display:flex;flex-direction:column;gap:2px}
.cluster-btn{display:flex;align-items:center;gap:10px;width:100%;padding:6px 8px;border:none;background:transparent;cursor:pointer;text-align:left;border-radius:7px;font-family:inherit;transition:opacity .15s}
.cluster-btn:hover{background:#f1f1ee}
.cluster-btn .dot{width:10px;height:10px;border-radius:50%;flex:none;border:1px solid rgba(0,0,0,.08)}
.cluster-btn .name{flex:1;font-size:13px;color:#262622}
.cluster-btn .cnt{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#a8a8a1}
select{width:100%;padding:9px 11px;border:1px solid #e0e0db;border-radius:8px;font-family:inherit;font-size:13px;color:#1c1c1c;background:#fff;outline:none;cursor:pointer}

/* ── main canvas area ── */
#main{position:relative;flex:1;height:100%;background:#fff;overflow:hidden;cursor:grab}
#main.dragging{cursor:grabbing}
#canvas-wrap{position:absolute;inset:0}
#hint{position:absolute;top:18px;left:20px;font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.06em;color:#bcbcb4;pointer-events:none;user-select:none}

/* ── zoom buttons ── */
#zoom-btns{position:absolute;right:18px;bottom:18px;display:flex;flex-direction:column;gap:6px}
.zoom-btn{width:34px;height:34px;border:1px solid #e6e6e1;background:#fff;border-radius:8px;font-size:18px;color:#3a3a36;cursor:pointer;line-height:1;display:flex;align-items:center;justify-content:center}
.zoom-btn:hover{background:#f4f4f1}

/* ── tooltip ── */
#tooltip{position:absolute;pointer-events:none;background:#1c1c1c;color:#fff;padding:8px 11px;border-radius:8px;max-width:260px;box-shadow:0 6px 20px rgba(0,0,0,.18);z-index:5;display:none}
#tooltip.visible{display:block}
#tt-cluster{font-family:'IBM Plex Mono',monospace;font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:#b9b9b2;display:flex;align-items:center;gap:7px;margin-bottom:4px}
#tt-dot{width:7px;height:7px;border-radius:50%;flex:none}
#tt-title{font-size:12.5px;line-height:1.35;font-weight:500}

/* ── detail panel ── */
#detail{position:absolute;top:0;right:0;height:100%;width:374px;background:#fff;border-left:1px solid #ececea;box-shadow:-10px 0 30px rgba(0,0,0,.05);display:flex;flex-direction:column;z-index:6;transform:translateX(100%);transition:transform .2s ease}
#detail.open{transform:translateX(0)}
#det-head{padding:22px 26px 0;display:flex;justify-content:space-between;align-items:center}
#det-cluster{display:flex;align-items:center;gap:9px}
#det-dot{width:11px;height:11px;border-radius:50%}
#det-clname{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:#76766f}
#close-btn{width:30px;height:30px;border:1px solid #eaeae5;background:#fff;border-radius:7px;font-size:16px;color:#76766f;cursor:pointer;line-height:1}
#close-btn:hover{background:#f4f4f1}
#det-body{padding:16px 26px 26px;overflow-y:auto}
#det-title{font-size:19px;font-weight:600;line-height:1.3;letter-spacing:-.01em;margin-bottom:12px;text-wrap:pretty}
#det-meta{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:#76766f;line-height:1.6;margin-bottom:20px}
#det-abs-label{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#b0b0a8;margin-bottom:8px}
#det-abstract{font-size:14px;line-height:1.65;color:#34342f;margin-bottom:24px;text-wrap:pretty}
#det-link{display:inline-flex;align-items:center;gap:8px;padding:10px 16px;background:#1c1c1c;color:#fff;text-decoration:none;border-radius:8px;font-size:13px;font-weight:500}
#det-link:hover{background:#333}

::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:#deded9;border-radius:5px}
::-webkit-scrollbar-track{background:transparent}
</style>
<script>__THREE_JS__</script>
<script>__ORBIT_JS__</script>
</head>
<body>
<div id="app">

  <!-- SIDEBAR -->
  <aside id="sidebar">
    <div id="sb-head">
      <div id="sb-lab">Neuropsychiatry Lab</div>
      <div id="sb-title">Research Atlas <span>3D</span></div>
      <div id="sb-desc">A semantic map of publications positioned by abstract content.</div>
    </div>
    <div id="sb-body">
      <div>
        <label class="ctrl-label" for="search">Search</label>
        <input id="search" type="text" placeholder="title, abstract, author…">
      </div>
      <div>
        <div class="year-row">
          <label class="ctrl-label">Year</label>
          <span id="year-display"></span>
        </div>
        <input id="yr-min" type="range">
        <input id="yr-max" type="range">
      </div>
      <div>
        <div class="topic-head">
          <label class="ctrl-label" style="margin-bottom:0">Topics</label>
          <button id="reset-btn">reset all</button>
        </div>
        <div id="cluster-list"></div>
      </div>
      <div>
              <label class="ctrl-label" for="author-select">Author</label>
              <select id="author-select"><option value="">All authors</option></select>
            </div>
            <div>
              <label class="ctrl-label">Auto-rotate</label>
              <div style="display:flex;gap:6px">
                <button id="rot-on"  style="flex:1;padding:8px;border:1px solid #e0e0db;border-radius:8px;font-family:'IBM Plex Mono',monospace;font-size:11px;cursor:pointer;background:#1c1c1c;color:#fff">On</button>
                <button id="rot-off" style="flex:1;padding:8px;border:1px solid #e0e0db;border-radius:8px;font-family:'IBM Plex Mono',monospace;font-size:11px;cursor:pointer;background:#fff;color:#9a9a93">Off</button>
              </div>
            </div>
          </div>
    <div id="sb-foot">
      <span><span id="count-vis">0</span><span id="count-sep"> / </span><span id="count-tot">0</span> shown</span>
      <span>v · 2026.06</span>
    </div>
  </aside>

  <!-- CANVAS -->
  <main id="main">
    <div id="canvas-wrap"></div>
    <div id="hint">drag to orbit · scroll to zoom · click a point for details</div>
    <div id="zoom-btns">
      <button class="zoom-btn" id="btn-in">+</button>
      <button class="zoom-btn" id="btn-out">−</button>
      <button class="zoom-btn" id="btn-fit" style="font-size:9.5px;font-family:'IBM Plex Mono',monospace">FIT</button>
    </div>
    <div id="tooltip">
      <div id="tt-cluster"><span id="tt-dot"></span><span id="tt-label"></span></div>
      <div id="tt-title"></div>
    </div>
  </main>

  <!-- DETAIL PANEL -->
  <div id="detail">
    <div id="det-head">
      <div id="det-cluster"><span id="det-dot"></span><span id="det-clname"></span></div>
      <button id="close-btn">×</button>
    </div>
    <div id="det-body">
      <div id="det-title"></div>
      <div id="det-meta"></div>
      <div id="det-abs-label">Abstract</div>
      <div id="det-abstract"></div>
      <a id="det-link" href="#" target="_blank">View publication →</a>
    </div>
  </div>
</div>

<script>
// ── DATA ──────────────────────────────────────────────────────────────────────
const PAPERS_RAW = __PAPERS_DATA__;
const PI_AUTHORS = __PI_AUTHORS__;

const COLORS = __CLUSTER_COLORS__;
const NOISE_COLOR = '#9a9a93';

function hexRgb(h){
  return [
    parseInt(h.slice(1,3),16)/255,
    parseInt(h.slice(3,5),16)/255,
    parseInt(h.slice(5,7),16)/255
  ];
}

// Build cluster map (noise / -1 excluded from map, handled separately)
const clusterMap = {};
PAPERS_RAW.forEach(p => {
  const id = p.cluster;
  if (id === -1) return;
  if (!clusterMap[id]) clusterMap[id] = {
    id, name: p.clusterLabel || 'Cluster ' + id,
    color: COLORS[id % COLORS.length],
    count: 0, cx: 0, cy: 0, cz: 0, n: 0
  };
  clusterMap[id].count++;
  clusterMap[id].cx += p.x || 0;
  clusterMap[id].cy += p.y || 0;
  clusterMap[id].cz += p.z || 0;
  clusterMap[id].n++;
});
Object.values(clusterMap).forEach(c => {
  if (c.n > 0) { c.cx /= c.n; c.cy /= c.n; c.cz /= c.n; }
});
const clusters = Object.values(clusterMap).sort((a, b) => a.id - b.id);
const papers   = PAPERS_RAW.map((p, i) => ({...p, _id: i, cluster: p.cluster ?? -1}));

// Year range
const allYears = papers.map(p => p.year).filter(Boolean);
const yearMin  = Math.min(...allYears), yearMax = Math.max(...allYears);

// ── STATE ─────────────────────────────────────────────────────────────────────
const state = {
  q: '', author: '',
  yMin: yearMin, yMax: yearMax,
  clusters: Object.fromEntries(clusters.map(c => [c.id, true])),
  hoverId: null,
  selected: null
};

function active(p) {
  // noise points (cluster === -1) are fully interactive — no early exit
  if (p.year < state.yMin || p.year > state.yMax) return false;
  if (state.author && !(p.authors || []).includes(state.author)) return false;
  // for clustered points, respect the cluster toggle
  if (p.cluster !== -1 && !state.clusters[p.cluster]) return false;
  if (state.q) {
    const q = state.q.toLowerCase();
    if (!(p.title.toLowerCase().includes(q) ||
          (p.abstract || '').toLowerCase().includes(q) ||
          (p.authors || []).join(' ').toLowerCase().includes(q))) return false;
  }
  return true;
}

// ── SIDEBAR INIT ──────────────────────────────────────────────────────────────
const yrMinEl   = document.getElementById('yr-min');
const yrMaxEl   = document.getElementById('yr-max');
const yrDisplay = document.getElementById('year-display');

[yrMinEl, yrMaxEl].forEach(el => { el.min = yearMin; el.max = yearMax; });
yrMinEl.value = yearMin;
yrMaxEl.value = yearMax;

function updateYearDisplay() {
  yrDisplay.textContent = state.yMin + ' – ' + state.yMax;
}
updateYearDisplay();

function updateCounts() {
  const vis = papers.filter(p => active(p)).length;
  const tot = papers.length;   // includes noise points
  document.getElementById('count-vis').textContent = vis;
  document.getElementById('count-tot').textContent = tot;
}

// Author select — PI names only, injected from build.py
const authorSel = document.getElementById('author-select');
PI_AUTHORS.forEach(a => {
  const o = document.createElement('option');
  o.value = a; o.textContent = a;
  authorSel.appendChild(o);
});

// Cluster buttons
const clusterList = document.getElementById('cluster-list');
clusters.forEach(c => {
  const btn = document.createElement('button');
  btn.className = 'cluster-btn';
  btn.dataset.id = c.id;
  btn.innerHTML = `<span class="dot" style="background:${c.color}"></span><span class="name">${c.name}</span><span class="cnt">${c.count}</span>`;
  clusterList.appendChild(btn);
});

// ── TOOLTIP ───────────────────────────────────────────────────────────────────
const tooltip = document.getElementById('tooltip');

function showTooltip(p, sx, sy) {
  const c = clusterMap[p.cluster];
  document.getElementById('tt-dot').style.background = c ? c.color : NOISE_COLOR;
  document.getElementById('tt-label').textContent    = (c ? c.name : 'Unclustered') + ' · ' + p.year;
  document.getElementById('tt-title').textContent    = p.title;
  tooltip.style.left      = sx + 'px';
  tooltip.style.top       = (sy - 16) + 'px';
  tooltip.style.transform = 'translate(-50%,-100%)';
  tooltip.classList.add('visible');
}
function hideTooltip() { tooltip.classList.remove('visible'); }

// ── DETAIL PANEL ──────────────────────────────────────────────────────────────
const detailEl = document.getElementById('detail');

function showDetail(p) {
  state.selected = p;
  const c = clusterMap[p.cluster];
  document.getElementById('det-dot').style.background = c ? c.color : NOISE_COLOR;
  document.getElementById('det-clname').textContent   = c ? c.name : 'Unclustered';
  document.getElementById('det-title').textContent    = p.title;
  document.getElementById('det-meta').textContent     = (p.authors || []).join(', ') + ' · ' + p.year;
  document.getElementById('det-abstract').textContent = p.abstract || '';
  document.getElementById('det-link').href            = p.url || '#';
  detailEl.classList.add('open');
  updatePoints();
}

document.getElementById('close-btn').addEventListener('click', () => {
  state.selected = null;
  detailEl.classList.remove('open');
  updatePoints();
});

// ── THREE.JS BOOTSTRAP ────────────────────────────────────────────────────────
(function bootstrap() {
  const THREE = window.THREE;
  const wrap   = document.getElementById('canvas-wrap');
  const mainEl = document.getElementById('main');
  let W = mainEl.clientWidth, H = mainEl.clientHeight;

  const scene  = new THREE.Scene();
  scene.background = new THREE.Color('#ffffff');

  const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 1000);
  camera.position.set(2, 1.5, 17);

  const renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  renderer.setSize(W, H);
  wrap.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping   = true;
  controls.dampingFactor   = 0.08;
  controls.rotateSpeed     = 0.8;
  controls.autoRotate      = true;
  controls.autoRotateSpeed = 0.5;

  // ── Geometry ──────────────────────────────────────────────────────────────
  const n   = papers.length;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  const sz  = new Float32Array(n);
  const al  = new Float32Array(n);

  papers.forEach((p, i) => {
    pos[i * 3]     = p.x || 0;
    pos[i * 3 + 1] = p.y || 0;
    pos[i * 3 + 2] = p.z || 0;
  });

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('aColor',   new THREE.BufferAttribute(col, 3));
  geom.setAttribute('aSize',    new THREE.BufferAttribute(sz,  1));
  geom.setAttribute('aAlpha',   new THREE.BufferAttribute(al,  1));

  const mat = new THREE.ShaderMaterial({
    uniforms: {uScale: {value: 108 * renderer.getPixelRatio()}},
    vertexShader: `
      attribute float aSize;
      attribute float aAlpha;
      attribute vec3  aColor;
      varying vec3  vColor;
      varying float vAlpha;
      uniform float uScale;
      void main() {
        vColor = aColor; vAlpha = aAlpha;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = aSize * (uScale / -mv.z);
        gl_Position  = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3  vColor;
      varying float vAlpha;
      void main() {
        vec2  c = gl_PointCoord - 0.5;
        float d = length(c);
        float a = smoothstep(0.5, 0.4, d);
        if (a <= 0.0) discard;
        gl_FragColor = vec4(vColor, vAlpha * a);
      }`,
    transparent: true,
    depthTest:   true,
    depthWrite:  false
  });

  const pts = new THREE.Points(geom, mat);
  scene.add(pts);

  // ── Cluster labels ────────────────────────────────────────────────────────
  const labelEls = {};
  clusters.forEach(c => {
    const d = document.createElement('div');
    d.style.cssText = [
      'position:absolute',
      'transform:translate(-50%,-50%)',
      'pointer-events:none',
      'white-space:nowrap',
      'background:rgba(255,255,255,.88)',
      'border:1px solid rgba(0,0,0,.07)',
      'border-radius:11px',
      'padding:3px 9px 3px 7px',
      "font-family:'IBM Plex Mono',monospace",
      'font-size:11.5px',
      'color:#2a2a26',
      'display:none',
      'align-items:center',
      'gap:7px',
      'z-index:3'
    ].join(';');
    d.innerHTML = `<span style="width:7px;height:7px;border-radius:50%;background:${c.color};flex:none;display:inline-block"></span>${c.name}`;
    wrap.appendChild(d);
    labelEls[c.id] = d;
  });

  // ── Raycasting ────────────────────────────────────────────────────────────
  const ray = new THREE.Raycaster();
  ray.params.Points.threshold = 0.28;

  function ndc(e) {
    const r = renderer.domElement.getBoundingClientRect();
    return {
      x:  ((e.clientX - r.left) / r.width)  * 2 - 1,
      y: -((e.clientY - r.top)  / r.height) * 2 + 1,
      sx: e.clientX - r.left,
      sy: e.clientY - r.top
    };
  }

  function pickPaper(e) {
    const m = ndc(e);
    ray.setFromCamera({x: m.x, y: m.y}, camera);
    const hits = ray.intersectObject(pts);
    for (const h of hits) {
      const p = papers[h.index];
      if (active(p)) return {paper: p, sx: m.sx, sy: m.sy};
    }
    return {paper: null};
  }

  let downXY = null;
  renderer.domElement.addEventListener('pointermove', e => {
    const r = pickPaper(e);
    mainEl.style.cursor = r.paper ? 'pointer' : 'grab';
    if (r.paper) { showTooltip(r.paper, r.sx, r.sy); state.hoverId = r.paper._id; }
    else         { hideTooltip(); state.hoverId = null; }
    updatePoints();
  });
  renderer.domElement.addEventListener('pointerdown', e => {
    downXY = {x: e.clientX, y: e.clientY};
  });
  renderer.domElement.addEventListener('pointerup', e => {
    if (!downXY) return;
    const moved = Math.hypot(e.clientX - downXY.x, e.clientY - downXY.y);
    downXY = null;
    if (moved > 5) return;
    const r = pickPaper(e);
    if (r.paper) showDetail(r.paper);
  });
  renderer.domElement.addEventListener('pointerleave', () => {
    hideTooltip(); state.hoverId = null; updatePoints();
  });

  // ── Zoom buttons ──────────────────────────────────────────────────────────
  function zoomBy(f) {
    const dir = camera.position.clone().sub(controls.target).multiplyScalar(1 / f);
    camera.position.copy(controls.target.clone().add(dir));
    controls.update();
  }
  document.getElementById('btn-in').onclick  = () => zoomBy(1.3);
  document.getElementById('btn-out').onclick = () => zoomBy(1 / 1.3);
  document.getElementById('btn-fit').onclick = () => {
      controls.target.set(0, 0, 0);
      camera.position.set(2, 1.5, 17);
      controls.update();
    };
    function setRotation(on) {
      controls.autoRotate = on;
      document.getElementById('rot-on').style.background  = on ? '#1c1c1c' : '#fff';
      document.getElementById('rot-on').style.color       = on ? '#fff'    : '#9a9a93';
      document.getElementById('rot-off').style.background = on ? '#fff'    : '#1c1c1c';
      document.getElementById('rot-off').style.color      = on ? '#9a9a93' : '#fff';
    }
    document.getElementById('rot-on').onclick  = () => setRotation(true);
    document.getElementById('rot-off').onclick = () => setRotation(false);

  // ── Resize ────────────────────────────────────────────────────────────────
  new ResizeObserver(() => {
    W = mainEl.clientWidth; H = mainEl.clientHeight;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
    mat.uniforms.uScale.value = 108 * renderer.getPixelRatio();
  }).observe(mainEl);

  // ── Sidebar wiring ────────────────────────────────────────────────────────
  document.getElementById('search').addEventListener('input', e => {
    state.q = e.target.value; updatePoints();
  });
  yrMinEl.addEventListener('input', e => {
    state.yMin = Math.min(+e.target.value, state.yMax);
    yrMinEl.value = state.yMin; updateYearDisplay(); updatePoints();
  });
  yrMaxEl.addEventListener('input', e => {
    state.yMax = Math.max(+e.target.value, state.yMin);
    yrMaxEl.value = state.yMax; updateYearDisplay(); updatePoints();
  });
  authorSel.addEventListener('change', e => {
    state.author = e.target.value; updatePoints();
  });
  document.getElementById('reset-btn').addEventListener('click', () => {
    state.q = ''; state.author = ''; state.yMin = yearMin; state.yMax = yearMax;
    document.getElementById('search').value = '';
    authorSel.value = '';
    yrMinEl.value = yearMin; yrMaxEl.value = yearMax; updateYearDisplay();
    clusters.forEach(c => { state.clusters[c.id] = true; });
    document.querySelectorAll('.cluster-btn').forEach(b => b.style.opacity = 1);
    updatePoints();
  });
  document.querySelectorAll('.cluster-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = +btn.dataset.id;
      state.clusters[id] = !state.clusters[id];
      btn.style.opacity = state.clusters[id] ? 1 : 0.42;
      updatePoints();
    });
  });

  // ── Update point colours / sizes ──────────────────────────────────────────
  const noiseRgb = hexRgb(NOISE_COLOR);

  function updatePoints() {
    updateCounts();
    const base = 1.0;
    for (let i = 0; i < n; i++) {
      const p       = papers[i];
      const act     = active(p);
      const sel     = state.selected && state.selected._id === p._id;
      const hov     = state.hoverId === p._id;
      const isNoise = p.cluster === -1;

      let r, g, b, a, s;

      if (!act) {
        // inactive / filtered out — nearly invisible
        r = 0.62; g = 0.62; b = 0.59; a = 0.09; s = base * 0.7;
      } else if (isNoise) {
        // active noise point — visible grey, slightly smaller than clustered
        r = noiseRgb[0]; g = noiseRgb[1]; b = noiseRgb[2]; a = 0.55; s = base * 0.85;
      } else {
        // active clustered point
        const c   = clusterMap[p.cluster];
        const rgb = c ? hexRgb(c.color) : noiseRgb;
        r = rgb[0]; g = rgb[1]; b = rgb[2]; a = 0.95; s = base;
      }

      // hover / selected — slightly smaller pop for noise to stay visually distinct
      if ((sel || hov) && act) {
        a = 1.0;
        s = base * (isNoise ? 1.5 : 1.9);
      }

      col[i*3] = r; col[i*3+1] = g; col[i*3+2] = b;
      al[i] = a; sz[i] = s;
    }
    geom.attributes.aColor.needsUpdate = true;
    geom.attributes.aAlpha.needsUpdate = true;
    geom.attributes.aSize.needsUpdate  = true;
  }

  updatePoints();

  // ── Animation loop ────────────────────────────────────────────────────────
  (function loop() {
    requestAnimationFrame(loop);
    controls.update();

    clusters.forEach(c => {
      const el = labelEls[c.id];
      if (!el) return;
      if (!state.clusters[c.id]) { el.style.display = 'none'; return; }
      const v = new THREE.Vector3(c.cx, c.cy, c.cz).project(camera);
      if (v.z > 1) { el.style.display = 'none'; return; }
      el.style.display = 'flex';
      el.style.left = ((v.x * 0.5 + 0.5) * W) + 'px';
      el.style.top  = ((-v.y * 0.5 + 0.5) * H) + 'px';
    });

    renderer.render(scene, camera);
  })();
})();
</script>
</body>
</html>
"""

# ── FETCH ─────────────────────────────────────────────────────────────────────
def openalex_filter():
    parts = []
    if AUTHOR_IDS:
        parts.append("authorships.author.id:" + "|".join(a.upper() for a in AUTHOR_IDS))
    if INSTITUTION_ID:
        parts.append("authorships.institutions.id:" + INSTITUTION_ID)
    parts.append("has_abstract:true")
    return ",".join(parts)

def fetch_openalex():
    base = "https://api.openalex.org/works"
    cursor, rows = "*", []
    while cursor:
        filt = openalex_filter()
        filter_str = urllib.parse.quote(filt, safe="|,:")
        cursor_str = urllib.parse.quote(cursor, safe="*")
        mailto_str = urllib.parse.quote(MAILTO)
        qs  = f"filter={filter_str}&per-page=200&cursor={cursor_str}&mailto={mailto_str}"
        url = f"{base}?{qs}"
        print(f"  → {url}")          # print so you can verify the URL looks right
        with urllib.request.urlopen(url) as r:
            data = json.load(r)
        rows.extend(data["results"])
        cursor = data["meta"].get("next_cursor")
        time.sleep(0.2)
        print(f"  fetched {len(rows)} works…")
    return rows

def abstract_from_inverted(inv):
    if not inv: return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))

def normalize(works):
    """Parse works and collect PI display names as a side-effect."""
    pi_ids_upper = {a.upper() for a in AUTHOR_IDS}
    out = []
    for w in works:
        ab = abstract_from_inverted(w.get("abstract_inverted_index"))
        if not ab or len(ab) < 60:
            continue
        for a in w.get("authorships", []):
            aid = (a["author"].get("id") or "").split("/")[-1].upper()
            if aid in pi_ids_upper:
                PI_AUTHOR_NAMES.add(a["author"]["display_name"])

        if not out:  # print first paper's raw keywords to inspect
            print("keywords sample:", w.get("keywords", [])[:3])
            print("topics sample:",   w.get("topics",   [])[:3])

        out.append({
            "id":       w["id"].split("/")[-1],
            "title":    w.get("title") or "(untitled)",
            "year":     w.get("publication_year"),
            "authors":  [auth["author"]["display_name"] for auth in w.get("authorships", [])[:8]],
            "abstract": ab,
            "url":      w.get("doi") or w["id"],
            "keywords": [k.get("keyword") or k.get("display_name") or "" for k in w.get("keywords", []) if k],
            "mesh":     [m["descriptor_name"] for m in w.get("mesh", [])
                         if m.get("descriptor_name") and m.get("is_major_topic")],
            "topics":   [t["display_name"] for t in w.get("topics", [])[:3] if t.get("display_name")],
        })
    return out

# ── EMBED / CLUSTER ───────────────────────────────────────────────────────────
def build(papers):
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]

    print("Embedding…")
    model = SentenceTransformer(EMBED_MODEL)
    vecs  = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    print("Reducing to 10D for clustering…")
    vecs_cluster = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE
    ).fit_transform(vecs)

    print(f"Reducing to {DIMS}D for visualisation…")
    coords = umap.UMAP(
        n_components=DIMS,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE
    ).fit_transform(vecs)

    print(f"Clustering with HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE})…")
    labels = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,  # lower = more, smaller clusters
        min_samples=2,                      # reduces noise points
        cluster_selection_method="leaf"     # finer-grained clusters
    ).fit_predict(vecs_cluster)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  {n_clusters} clusters, {n_noise} noise points")
    if n_clusters > len(CLUSTER_COLORS):
        print(f"  NOTE: {n_clusters} clusters but only {len(CLUSTER_COLORS)} colors defined — "
              f"colors will repeat (cluster id % {len(CLUSTER_COLORS)}). Add more to CLUSTER_COLORS if you want every cluster visually distinct.")

    names = label_clusters(texts, labels, papers)

    # Normalise coords to roughly [-5, 5]
    coords -= coords.mean(axis=0)
    coords /= (np.abs(coords).max() + 1e-9) / 5.0

    for i, p in enumerate(papers):
        c = int(labels[i])
        p["cluster"]      = c
        p["clusterLabel"] = names.get(c, "Unclustered")
        p["x"] = round(float(coords[i][0]), 4)
        p["y"] = round(float(coords[i][1]), 4)
        if DIMS == 3:
            p["z"] = round(float(coords[i][2]), 4)

    return papers

def label_clusters(texts, labels, papers):
    groups = defaultdict(list)
    for i, (t, c) in enumerate(zip(texts, labels)):
        if c != -1:
            groups[c].append(i)

    BLACKLIST = {
        "van", "de", "den", "der", "het", "een", "ter",
        "humans", "human", "adult", "adults", "female", "females",
        "male", "males", "aged", "child", "children", "animal",
        "medicine", "psychology", "disease", "neuroscience",
        "health", "healthcare", "science", "clinical", "pathology",
        "psychiatry", "diagnosis", "prognosis", "treatment",
        "patients", "research", "study", "methods", "results", "suicide"
    }

    FUNCTION_WORDS = {
        "with", "without", "following", "after", "before", "during",
        "using", "among", "between", "within", "across", "versus",
        "into", "from", "than", "that", "this", "these", "those",
        "which", "when", "where", "while", "variant", "primary",
        "secondary", "major", "minor",
    }

    def is_clean(term):
        words = term.lower().split()
        return (
            all(w.isalpha() for w in words)
            and not any(w in BLACKLIST for w in words)
            and not any(w in FUNCTION_WORDS for w in words)
            and len(term) > 4
        )

    # ── pass 1: count MeSH per cluster and globally ──
    mesh_per_cluster  = {}
    total_mesh_counts = defaultdict(float)

    for c, indices in groups.items():
        counts = defaultdict(float)
        for i in indices:
            for term in papers[i].get("mesh", []):
                if term and is_clean(term):
                    counts[term.lower()] += 1.0
                    total_mesh_counts[term.lower()] += 1.0
        mesh_per_cluster[c] = counts

    # ── pass 2: score and label ──
    names = {-1: "Unclustered"}
    for c, indices in groups.items():
        counts = mesh_per_cluster[c]

        # c-TF-IDF: freq in cluster / freq across all clusters
        scored = {
            kw: freq / (total_mesh_counts[kw] ** 0.7 + 1)
            for kw, freq in counts.items()
        }
        ranked = sorted(scored, key=lambda k: -scored[k])[:2]

        if ranked:
            names[c] = " · ".join(k.title() for k in ranked)
            print(f"  cluster {c} (n={len(indices)}): {names[c]}  [mesh]")
            continue

        # ── fallback 1: discriminative keywords + title bigrams ──
        kw_counts   = defaultdict(float)
        total_kw    = defaultdict(float)
        for i in indices:
            for kw in papers[i].get("keywords", []):
                if kw and is_clean(kw):
                    kw_counts[kw.lower()] += 1.0
                    total_kw[kw.lower()]  += 1.0
            title_words = papers[i].get("title", "").lower().split()
            for j in range(len(title_words) - 1):
                bg = title_words[j] + " " + title_words[j+1]
                if is_clean(bg) and all(len(w) > 3 for w in bg.split()):
                    kw_counts[bg] += 0.4
                    total_kw[bg]  += 0.4

        # build global totals for this fallback
        all_kw_total = defaultdict(float)
        for c2, idx2 in groups.items():
            for i in idx2:
                for kw in papers[i].get("keywords", []):
                    if kw and is_clean(kw):
                        all_kw_total[kw.lower()] += 1.0

        scored_kw = {
            kw: freq / (all_kw_total[kw] ** 0.7 + 1)
            for kw, freq in kw_counts.items()
        }
        ranked_kw = sorted(scored_kw, key=lambda k: -scored_kw[k])[:2]

        if ranked_kw:
            names[c] = " · ".join(k.title() for k in ranked_kw)
            print(f"  cluster {c} (n={len(indices)}): {names[c]}  [keywords]")
            continue

        # ── fallback 2: TF-IDF on abstracts ──
        cluster_texts = [texts[i] for i in indices]
        tfidf  = TfidfVectorizer(stop_words="english", max_features=2000,
                                 ngram_range=(2, 3), sublinear_tf=True)
        X      = tfidf.fit_transform(cluster_texts)
        scores = np.asarray(X.mean(axis=0)).ravel()
        terms  = np.array(tfidf.get_feature_names_out())
        ranked_tf = [t for t in terms[scores.argsort()[::-1]]
                     if is_clean(t)][:2]
        names[c] = " · ".join(t.title() for t in ranked_tf) if ranked_tf else f"Cluster {c}"
        print(f"  cluster {c} (n={len(indices)}): {names[c]}  [tfidf]")

    return names

# ── WRITE ─────────────────────────────────────────────────────────────────────
def write_atlas(papers, three_js, orbit_js):
    html = HTML
    html = html.replace("__THREE_JS__",       three_js)
    html = html.replace("__ORBIT_JS__",       orbit_js)
    html = html.replace("__PAPERS_DATA__",    json.dumps(papers, ensure_ascii=False))
    html = html.replace("__PI_AUTHORS__",     json.dumps(sorted(PI_AUTHOR_NAMES), ensure_ascii=False))
    html = html.replace("__CLUSTER_COLORS__", json.dumps(CLUSTER_COLORS, ensure_ascii=False))
    OUT_PATH.write_text(html, encoding="utf-8")
    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"\n✓ Wrote {OUT_PATH}")
    print(f"  {len(papers)} papers · {DIMS}D · {size_mb:.1f} MB")
    print(f"  {len(PI_AUTHOR_NAMES)} PI authors in dropdown: {sorted(PI_AUTHOR_NAMES)}")
    print("  Open atlas.html directly in any browser — no server needed.")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    three_js, orbit_js = fetch_js_libraries()

    print("\nFetching from OpenAlex…")
    raw_works = fetch_openalex()
    papers    = normalize(raw_works)
    print(f"{len(papers)} papers with usable abstracts.")
    print(f"{len(PI_AUTHOR_NAMES)} PI author names collected: {sorted(PI_AUTHOR_NAMES)}")

    papers = build(papers)
    write_atlas(papers, three_js, orbit_js)