// Hand
// ====
// A lint over the checked defs, outside the kernel: it reads the terms the
// checker returns (every node Ann-wrapped with its type) and changes none.
// A handle (File, Socket, Listener, Window, Audio) is its fd in a word, and
// affine means at most once, so one a def binds and never uses stays open
// until the process ends (UPSTREAM U19). hand_drops names each such binder.
// Not named: a binder whose name starts with _, one in a scope that ends
// in IO.die (the exit closes it), and a handle a generic def holds as a
// type variable.

import * as Bend from "./bend.ts";

type Name = Bend.Name;
type LTerm = Bend.LTerm;
type HTerm = Bend.HTerm;

export type Drop = { def: Name; k: Name; ty: Name; s?: Bend.Span };

const HANDLES = new Set(["File", "Socket", "Listener", "Window", "Audio"]);

const HOLE: HTerm = Bend.Var("_", -1);

// the handle a value of type A holds: A, or one in its type arguments or
// its constructors' fields (none in a function or an IO action)
function held(book: Bend.Book, A: HTerm, seen: Set<Name>): Name | null {
  const t = Bend.term_wnf(book, A);
  if (t.$ === "Ref") {
    return HANDLES.has(t.k) ? t.k : null;
  }
  if (t.$ !== "ADT" || t.k === "IO" || seen.has(t.k)) {
    return null;
  }
  seen.add(t.k);
  const tld = book.tlds[t.k];
  const xs = t.x.map((x) => x.$ === "Lam" ? (x.f as Bend.HBody)(HOLE) : x);
  for (const c of tld?.$ === "ADT" ? tld.c : []) {
    let T = Bend.term_wnf(book, c.T);
    for (; T.$ === "All"; T = Bend.term_wnf(book, T.B(HOLE))) {
      xs.push(T.A);
    }
  }
  return xs.reduce((h: Name | null, x) => h ?? held(book, x, seen), null);
}

// whether level i is used in t (a type annotation is no use)
function used(t: unknown, i: number): boolean {
  if (typeof t !== "object" || t === null) {
    return false;
  }
  const n = t as { $?: string; i?: number; x?: unknown };
  if (n.$ === "Var") {
    return n.i === i;
  }
  if (n.$ === "Ann") {
    return used(n.x, i);
  }
  return Object.entries(t).some(([f, v]) => f !== "s" && used(v, i));
}

// whether a scope ends in IO.die, whose exit closes every handle
function dies(t: LTerm): boolean {
  for (;;) {
    if (t.$ === "Ann") {
      t = t.x;
    } else if (t.$ === "Lam" || t.$ === "Let") {
      t = t.f;
    } else {
      while (t.$ === "App") {
        t = t.f.$ === "Ann" ? t.f.x : t.f;
      }
      return t.$ === "Ref" && t.k === "IO.die";
    }
  }
}

function type_of(T: LTerm): HTerm {
  return Bend.term_force(Bend.term_higher(T));
}

function walk(book: Bend.Book, def: Name, t: unknown, out: Drop[]): void {
  if (typeof t !== "object" || t === null) {
    return;
  }
  const n = t as LTerm;
  const drop = (k: Name, q: Bend.Quant | undefined, A: HTerm, i: number,
    f: LTerm, s?: Bend.Span): void => {
    const h = q?.$ === "None" || k.startsWith("_") ? null
      : held(book, A, new Set());
    if (h !== null && !used(f, i) && !dies(f)) {
      out.push({ def, k, ty: h, s });
    }
  };
  if (n.$ === "Ann" && n.x.$ === "Lam") {
    const all = Bend.term_wnf(book, type_of(n.T));
    if (all.$ === "All") {
      drop(n.x.k, all.q, all.A, n.x.i, n.x.f, n.x.s ?? n.s);
    }
  } else if (n.$ === "Let") {
    n.k.forEach((k, j) => {
      const v = n.v[j];
      if (v.$ === "Ann") {
        drop(k, n.q[j], type_of(v.T), n.i[j], n.f, n.s);
      }
    });
  }
  if (n.$ !== "Var") {
    for (const [f, v] of Object.entries(t)) {
      if (f !== "s" && f !== "T") {
        walk(book, def, v, out);
      }
    }
  }
}

// the handles each of defs binds and never uses
export function hand_drops(book: Bend.Book, defs: Name[]): Drop[] {
  const out: Drop[] = [];
  for (const k of defs) {
    const tld = book.tlds[k];
    if (tld?.$ === "Def" && tld.e !== undefined) {
      walk(book, k, tld.e, out);
    }
  }
  return out;
}
