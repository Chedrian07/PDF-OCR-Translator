// 런타임 단위 테스트용 최소 DOM — jsdom 없이 실제 프런트 모듈(js/*.js)을 Node에서
// 돌리기 위한 것이다. 트리 조작(append/insert/remove), 텍스트, 속성·dataset·classList,
// 이벤트, 포커스, 단순 선택자(태그·#id·.class·[attr]·[attr="v"]·후손 결합자, 쉼표 목록)만
// 흉내 낸다. HTML 파싱은 하지 않는다 — innerHTML은 문자열로만 보관한다.
//
// 사용: const dom = installFakeDom(t); → globalThis.document/window/requestAnimationFrame을
// 테스트 동안 바꿔 끼우고 t.after에서 되돌린다.

class FakeClassList {
  constructor(node) { this.node = node; }
  get set() { return this.node._classes; }
  add(...names) { for (const n of names) if (n) this.set.add(n); }
  remove(...names) { for (const n of names) this.set.delete(n); }
  contains(name) { return this.set.has(name); }
  toggle(name, force) {
    const on = force === undefined ? !this.set.has(name) : !!force;
    if (on) this.set.add(name); else this.set.delete(name);
    return on;
  }
  get length() { return this.set.size; }
  toString() { return [...this.set].join(' '); }
}

class FakeNode {
  constructor(doc, nodeType) {
    this.ownerDocument = doc;
    this.nodeType = nodeType;
    this.childNodes = [];
    this.parentNode = null;
  }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  get nextSibling() {
    if (!this.parentNode) return null;
    const list = this.parentNode.childNodes;
    return list[list.indexOf(this) + 1] || null;
  }
  get previousSibling() {
    if (!this.parentNode) return null;
    const list = this.parentNode.childNodes;
    return list[list.indexOf(this) - 1] || null;
  }
  get parentElement() { return this.parentNode && this.parentNode.nodeType === 1 ? this.parentNode : null; }
  get isConnected() {
    let node = this;
    while (node) {
      if (node === this.ownerDocument.documentElement) return true;
      node = node.parentNode;
    }
    return false;
  }
  _adopt(child) {
    if (child.nodeType === 11) { // DocumentFragment — 자식만 옮긴다
      const kids = [...child.childNodes];
      child.childNodes = [];
      for (const k of kids) k.parentNode = null;
      return kids;
    }
    if (child.parentNode) child.parentNode.removeChild(child);
    return [child];
  }
  appendChild(child) {
    for (const k of this._adopt(child)) { k.parentNode = this; this.childNodes.push(k); }
    return child;
  }
  append(...nodes) {
    for (const n of nodes) this.appendChild(typeof n === 'string' ? this.ownerDocument.createTextNode(n) : n);
  }
  prepend(...nodes) {
    const first = this.firstChild;
    for (const n of nodes) {
      const node = typeof n === 'string' ? this.ownerDocument.createTextNode(n) : n;
      if (first) this.insertBefore(node, first); else this.appendChild(node);
    }
  }
  insertBefore(child, ref) {
    if (!ref) return this.appendChild(child);
    const kids = this._adopt(child);
    const at = this.childNodes.indexOf(ref);
    if (at < 0) throw new Error('insertBefore: ref is not a child');
    for (const k of kids) k.parentNode = this;
    this.childNodes.splice(at, 0, ...kids);
    return child;
  }
  removeChild(child) {
    const at = this.childNodes.indexOf(child);
    if (at < 0) throw new Error('removeChild: not a child');
    this.childNodes.splice(at, 1);
    child.parentNode = null;
    // 브라우저의 focus fixup: 포커스된 노드가 (이동을 포함해) 트리에서 빠지면 body로 간다.
    const doc = this.ownerDocument;
    if (doc && doc.activeElement && child.contains(doc.activeElement)) doc.activeElement = doc.body;
    return child;
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  replaceWith(...nodes) {
    const parent = this.parentNode;
    if (!parent) return;
    for (const n of nodes) parent.insertBefore(typeof n === 'string' ? this.ownerDocument.createTextNode(n) : n, this);
    parent.removeChild(this);
  }
  contains(node) {
    while (node) {
      if (node === this) return true;
      node = node.parentNode;
    }
    return false;
  }
  get textContent() {
    if (this.nodeType === 3) return this.data;
    return this.childNodes.map((c) => c.textContent).join('');
  }
  _clearChildren() {
    const doc = this.ownerDocument;
    const focused = doc && doc.activeElement;
    for (const c of this.childNodes) c.parentNode = null;
    const dropped = this.childNodes;
    this.childNodes = [];
    if (focused && dropped.some((c) => c.contains(focused))) doc.activeElement = doc.body;
  }
  set textContent(value) {
    if (this.nodeType === 3) { this.data = String(value); return; }
    this._clearChildren();
    this._html = undefined;
    const text = value == null ? '' : String(value);
    if (text) this.appendChild(this.ownerDocument.createTextNode(text));
  }
}

class FakeText extends FakeNode {
  constructor(doc, data) { super(doc, 3); this.data = String(data); }
  get nodeValue() { return this.data; }
}

class FakeFragment extends FakeNode {
  constructor(doc) { super(doc, 11); }
  querySelector(sel) { return FakeElement.prototype.querySelector.call(this, sel); }
  querySelectorAll(sel) { return FakeElement.prototype.querySelectorAll.call(this, sel); }
}

function parseCompound(text) {
  const out = { tag: null, id: null, classes: [], attrs: [] };
  const re = /([a-zA-Z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w:-]+)(?:([~^$*|]?=)\s*(?:"([^"]*)"|'([^']*)'|([^\]\s]+)))?\]/g;
  let m;
  let consumed = '';
  while ((m = re.exec(text)) !== null) {
    consumed += m[0];
    if (m[1]) out.tag = m[1].toUpperCase();
    else if (m[2]) out.id = m[2];
    else if (m[3]) out.classes.push(m[3]);
    else out.attrs.push({ name: m[4], op: m[5] || null, value: m[6] ?? m[7] ?? m[8] ?? null });
  }
  if (consumed.length !== text.length) throw new Error(`fake-dom: unsupported selector "${text}"`);
  return out;
}

function splitTopLevel(selector, sep) {
  const parts = [];
  let depth = 0;
  let quote = null;
  let cur = '';
  for (const ch of selector) {
    if (quote) { cur += ch; if (ch === quote) quote = null; continue; }
    if (ch === '"' || ch === "'") { quote = ch; cur += ch; continue; }
    if (ch === '[') depth += 1;
    if (ch === ']') depth -= 1;
    if (depth === 0 && sep.test(ch)) { if (cur.trim()) parts.push(cur.trim()); cur = ''; continue; }
    cur += ch;
  }
  if (cur.trim()) parts.push(cur.trim());
  return parts;
}

function matchesCompound(node, c) {
  if (!node || node.nodeType !== 1) return false;
  if (c.tag && node.tagName !== c.tag) return false;
  if (c.id && node.getAttribute('id') !== c.id) return false;
  for (const cls of c.classes) if (!node.classList.contains(cls)) return false;
  for (const a of c.attrs) {
    const v = node.getAttribute(a.name);
    if (v == null) return false;
    if (a.op === '=' && v !== a.value) return false;
  }
  return true;
}

function matchesSelector(node, selector) {
  return splitTopLevel(selector, /,/).some((alt) => {
    const chain = splitTopLevel(alt, /\s/).map(parseCompound);
    if (!matchesCompound(node, chain[chain.length - 1])) return false;
    let anc = node.parentNode;
    for (let i = chain.length - 2; i >= 0; i -= 1) {
      while (anc && !matchesCompound(anc, chain[i])) anc = anc.parentNode;
      if (!anc) return false;
      anc = anc.parentNode;
    }
    return true;
  });
}

function* descendants(root) {
  for (const child of root.childNodes) {
    if (child.nodeType === 1) {
      yield child;
      yield* descendants(child);
    }
  }
}

class FakeElement extends FakeNode {
  constructor(doc, tag) {
    super(doc, 1);
    this.tagName = String(tag).toUpperCase();
    this.localName = String(tag).toLowerCase();
    this._attrs = new Map();
    this._classes = new Set();
    this.classList = new FakeClassList(this);
    this.dataset = {};
    this.style = { setProperty(k, v) { this[k] = v; } };
    this.listeners = new Map();
    this.hidden = false;
    this.disabled = false;
    this.tabIndex = -1;
    this.scrollTop = 0;
    this.scrollLeft = 0;
    this.clientHeight = 0;
    this.clientWidth = 0;
    this.scrollHeight = 0;
    this.offsetHeight = 0;
    this.offsetParent = null;
    if (this.tagName === 'TEMPLATE') this.content = new FakeFragment(doc);
  }
  get children() { return this.childNodes.filter((n) => n.nodeType === 1); }
  get firstElementChild() { return this.children[0] || null; }
  get lastElementChild() { const k = this.children; return k[k.length - 1] || null; }
  get childElementCount() { return this.children.length; }
  get className() { return [...this._classes].join(' '); }
  set className(value) { this._classes = new Set(String(value || '').split(/\s+/).filter(Boolean)); }
  get id() { return this.getAttribute('id') || ''; }
  set id(value) { this.setAttribute('id', value); }
  get innerHTML() { return this._html !== undefined ? this._html : ''; }
  set innerHTML(value) {
    this._clearChildren();
    this._html = String(value);
  }
  setAttribute(name, value) {
    const v = String(value);
    if (name === 'class') { this.className = v; return; }
    if (name.startsWith('data-')) {
      const key = name.slice(5).replace(/-([a-z])/g, (_, ch) => ch.toUpperCase());
      this.dataset[key] = v;
    }
    if (name === 'hidden') this.hidden = true;
    if (name === 'disabled') this.disabled = true;
    if (name === 'tabindex') this.tabIndex = Number(v);
    this._attrs.set(name, v);
  }
  getAttribute(name) {
    if (name === 'class') return this._classes.size ? this.className : null;
    if (name.startsWith('data-')) {
      const key = name.slice(5).replace(/-([a-z])/g, (_, ch) => ch.toUpperCase());
      return key in this.dataset ? String(this.dataset[key]) : null;
    }
    if (name === 'hidden') return this.hidden ? '' : null;
    if (name === 'disabled') return this.disabled ? '' : null;
    return this._attrs.has(name) ? this._attrs.get(name) : null;
  }
  hasAttribute(name) { return this.getAttribute(name) != null; }
  removeAttribute(name) {
    if (name === 'class') { this._classes = new Set(); return; }
    if (name.startsWith('data-')) {
      const key = name.slice(5).replace(/-([a-z])/g, (_, ch) => ch.toUpperCase());
      delete this.dataset[key];
    }
    if (name === 'hidden') this.hidden = false;
    if (name === 'disabled') this.disabled = false;
    this._attrs.delete(name);
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  removeEventListener(type, fn) {
    const list = this.listeners.get(type) || [];
    const at = list.indexOf(fn);
    if (at >= 0) list.splice(at, 1);
  }
  dispatch(type, init = {}) {
    const ev = { type, target: this, currentTarget: this, defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; }, stopPropagation() {}, ...init };
    for (const fn of [...(this.listeners.get(type) || [])]) fn.call(this, ev);
    return ev;
  }
  click() { if (!this.disabled) this.dispatch('click'); }
  focus() { this.ownerDocument.activeElement = this; }
  blur() { if (this.ownerDocument.activeElement === this) this.ownerDocument.activeElement = this.ownerDocument.body; }
  matches(selector) { return matchesSelector(this, selector); }
  closest(selector) {
    let node = this;
    while (node && node.nodeType === 1) {
      if (matchesSelector(node, selector)) return node;
      node = node.parentNode;
    }
    return null;
  }
  querySelector(selector) {
    for (const node of descendants(this)) if (matchesSelector(node, selector)) return node;
    return null;
  }
  querySelectorAll(selector) {
    const out = [];
    for (const node of descendants(this)) if (matchesSelector(node, selector)) out.push(node);
    return out;
  }
  getBoundingClientRect() { return { top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0 }; }
  getClientRects() { return []; }
  scrollTo(opts) { if (opts && typeof opts.top === 'number') this.scrollTop = opts.top; }
  scrollIntoView() {}
}

export function createFakeDocument() {
  const doc = {
    activeElement: null,
    readyState: 'complete',
    hidden: false,
    visibilityState: 'visible',
    listeners: new Map(),
    createElement(tag) { return new FakeElement(doc, tag); },
    createTextNode(text) { return new FakeText(doc, text); },
    createDocumentFragment() { return new FakeFragment(doc); },
    getElementById(id) { return doc.documentElement.querySelector(`#${id}`); },
    querySelector(sel) { return doc.documentElement.querySelector(sel); },
    querySelectorAll(sel) { return doc.documentElement.querySelectorAll(sel); },
    addEventListener(type, fn) {
      if (!doc.listeners.has(type)) doc.listeners.set(type, []);
      doc.listeners.get(type).push(fn);
    },
    removeEventListener(type, fn) {
      const list = doc.listeners.get(type) || [];
      const at = list.indexOf(fn);
      if (at >= 0) list.splice(at, 1);
    },
    dispatch(type, init = {}) {
      for (const fn of [...(doc.listeners.get(type) || [])]) fn({ type, ...init });
    },
  };
  doc.documentElement = new FakeElement(doc, 'html');
  doc.body = new FakeElement(doc, 'body');
  doc.documentElement.appendChild(doc.body);
  doc.activeElement = doc.body;
  return doc;
}

// 테스트 동안 전역 document/window/rAF를 가짜로 바꾼다. 반환값은 가짜 document.
export function installFakeDom(t) {
  const doc = createFakeDocument();
  const saved = {};
  for (const key of ['document', 'window', 'requestAnimationFrame', 'cancelAnimationFrame']) {
    saved[key] = Object.getOwnPropertyDescriptor(globalThis, key);
  }
  const frames = [];
  globalThis.document = doc;
  globalThis.window = globalThis;
  globalThis.requestAnimationFrame = (fn) => { frames.push(fn); return frames.length; };
  globalThis.cancelAnimationFrame = () => {};
  doc.runFrames = () => { while (frames.length) frames.shift()(0); };
  t.after(() => {
    for (const [key, desc] of Object.entries(saved)) {
      if (desc) Object.defineProperty(globalThis, key, desc);
      else delete globalThis[key];
    }
  });
  return doc;
}

// 테스트용 요소 하나를 body에 붙여 돌려준다 (el.* 슬롯 채우기용).
export function mount(doc, tag = 'div', id) {
  const node = doc.createElement(tag);
  if (id) node.setAttribute('id', id);
  doc.body.appendChild(node);
  return node;
}
