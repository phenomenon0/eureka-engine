// Export: the checked book as core terms, for an independent re-checker
// (kernel/). bend.ts is not touched: this walks the Book that book_valid
// has already accepted and prints every family and every definition, in
// the order the checker filled them, as s-expressions over de Bruijn
// indices. Nothing here is trusted: the kernel re-checks all of it.
//
// FORMAT (one item per line; # starts a comment line)
//
//   file  ::= "(bend-core 1)" item*
//   item  ::= (adt NAME PN FLAGS SIG (ctr NAME FN TYPE)*)
//           | (def NAME N X FLAGS TYPE BODY)
//   FLAGS ::= (flags FLAG*)   FLAG ::= law | base | unsafe | foreign
//   BODY  ::= none | TERM     (none: a bodiless native, foreign or law)
//   NAME  ::= a "string" with \" and \\ escapes
//   Q     ::= 0 | 1 | 2       (erased -x, affine x, reusable +x)
//
//   TERM ::= (var I)             de Bruijn index, 0 = innermost binder
//          | (ref NAME)          a definition or a family (bare)
//          | (kind TERM)         Kind(g): Type = (kind (q 1)), Data (q 2)
//          | quant | (q Q)       the sort of quantities and its literals
//          | (min TERM TERM)     a <&> b
//          | (all Q TERM TERM)   @q x:A -> B (binds in B)
//          | (lam M TERM)        x => f; M is _ or + (a + binder mark)
//          | (app TERM TERM)
//          | (adt NAME (NAME*) TERM*)   D<ps>, the peeled constructors first
//          | (ctr NAME TERM*)    C{fields} (parameters come from the goal)
//          | (mat NAME TERM TERM)       \{C: h; m}
//          | efq | rfl
//          | (eql TERM TERM TERM)       {a == b : T}
//          | (rwt TERM TERM TERM)       %e : P; f
//          | (let Q TERM TERM)   q x = v; f (binds in f); a parallel let
//                                is nested, each value in the outer scope
//          | (ann TERM TERM)     {x : T}
//          | (hole NAME)         ?name
//
// N is the def's column count (its case-tree arity), X its count of
// leading ~ (template) parameters. A family's SIG is its parameter
// telescope tipped at its kind; a constructor's TYPE is the family's
// parameters then its FN fields, tipped at the family at its parameters.
// A law is emitted where its def fills it (its last place in the order).

import * as Bend from "./bend.ts";

type Q = Bend.Quant;

function q_str(q: Q): string {
  return q.$ === "None" ? "0" : q.$ === "Lone" ? "1" : "2";
}

function name(k: string): string {
  return "\"" + k.replace(/\\/g, "\\\\").replace(/"/g, "\\\"") + "\"";
}

// lower an HOAS term: a binder is opened on a marker variable holding its
// level, and a marker at level l under depth d prints as index d - 1 - l
function term(tm: Bend.HTerm, d: number): string {
  const t = Bend.term_force(tm);
  switch (t.$) {
    case "Var": {
      if (t.i < 0 || t.i >= d) {
        throw new Error("export: a free variable " + t.k);
      }
      return "(var " + String(d - 1 - t.i) + ")";
    }
    case "Ref": return "(ref " + name(t.k) + ")";
    case "Typ": return "(kind " + term(t.g, d) + ")";
    case "Qnt": return "quant";
    case "Qua": return "(q " + q_str(t.q) + ")";
    case "Min": return "(min " + term(t.a, d) + " " + term(t.b, d) + ")";
    case "All": {
      const B = t.B(Bend.Var(t.k, d));
      return "(all " + q_str(t.q) + " " + term(t.A, d) + " " + term(B, d + 1) + ")";
    }
    case "Lam": {
      const f = t.f(Bend.Var(t.k, d));
      return "(lam " + (t.q?.$ === "Many" ? "+" : "_") + " " + term(f, d + 1) + ")";
    }
    case "App": return "(app " + term(t.f, d) + " " + term(t.x, d) + ")";
    case "ADT": {
      const xs = t.x.map((x) => " " + term(x, d)).join("");
      return "(adt " + name(t.k) + " (" + t.r.map(name).join(" ") + ")" + xs + ")";
    }
    case "Ctr": return "(ctr " + name(t.k) + t.x.map((x) => " " + term(x, d)).join("") + ")";
    // a string or nat literal leaves as the constructor chain it stands for;
    // a nat past NAT_LITERAL_MAX as U32.to_nat of its word, as the compiler
    // reads it, so no chain outgrows the stack
    case "Lit": return term(Bend.term_higher(typeof t.v === "number"
      && t.v > Bend.NAT_LITERAL_MAX
      ? Bend.App(Bend.Ref("U32.to_nat"), Bend.u32_to_term(t.v))
      : Bend.lit_full(t)), d);
    case "Mat": return "(mat " + name(t.k) + " " + term(t.h, d) + " " + term(t.m, d) + ")";
    case "Efq": return "efq";
    case "Eql": return "(eql " + term(t.a, d) + " " + term(t.b, d) + " " + term(t.T, d) + ")";
    case "Rfl": return "rfl";
    case "Rwt": return "(rwt " + term(t.e, d) + " " + term(t.p, d) + " " + term(t.f, d) + ")";
    case "Let": {
      const n  = t.k.length;
      const xs = t.k.map((k, j): Bend.HTerm => Bend.Var(k, d + j));
      let out  = term(t.f(xs), d + n);
      for (let j = n - 1; j >= 0; j--) {
        // value j sits under j let binders, but its levels stay below d
        out = "(let " + q_str(t.q[j]) + " " + term(t.v[j], d + j) + " " + out + ")";
      }
      return out;
    }
    case "Ann": return "(ann " + term(t.x, d) + " " + term(t.T, d) + ")";
    case "Hol": return "(hole " + name(t.k) + ")";
    case "Sub": throw new Error("export: a substitution node survived flattening");
  }
}

export function book_export(book: Bend.Book): string {
  const last = new Map<string, number>();
  const seen = new Map<string, number>();
  book.order.forEach((k, i) => {
    last.set(k, i);
    seen.set(k, (seen.get(k) ?? 0) + 1);
  });
  const out: string[] = ["(bend-core 1)"];
  book.order.forEach((k, i) => {
    if (last.get(k) !== i) {
      return;
    }
    const tld = book.tlds[k];
    if (tld.$ === "ADT") {
      const cs = tld.c.map((c) => " (ctr " + name(c.k) + " " + String(c.n) + " " + term(c.T, 0) + ")");
      out.push("(adt " + name(k) + " " + String(tld.n) + " (flags" + (tld.b ? " base" : "") + ") "
        + term(tld.T, 0) + cs.join("") + ")");
      return;
    }
    const fl: string[] = [];
    if ((seen.get(k) ?? 0) > 1 || (tld.v === null && !tld.i)) fl.push("law");
    if (tld.b) fl.push("base");
    if (tld.u) fl.push("unsafe");
    if (tld.i) fl.push("foreign");
    const body = tld.v === null ? "none" : term(tld.v, 0);
    out.push("(def " + name(k) + " " + String(tld.n) + " " + String(tld.x ?? 0)
      + " (flags" + fl.map((f) => " " + f).join("") + ") " + term(tld.T, 0) + " " + body + ")");
  });
  return out.join("\n") + "\n";
}
