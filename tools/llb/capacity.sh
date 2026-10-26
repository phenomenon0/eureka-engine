# Measured capacity policy. Sourced by llb; table columns: alias required-free-MiB evidence.
# Thresholds apply to the measured prompt range, on sm_86; other GPUs remain unverified.
# A threshold is a conservative tested level, not the exact allocation minimum.
capacity_free() {
  dev=${CUDA_VISIBLE_DEVICES:-0}; dev=${dev%%,*}
  CAPACITY_FREE=$(nvidia-smi -i "$dev" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  num "$CAPACITY_FREE" || die "cannot read free GPU memory; run llb doctor"
}
capacity_need() {
  CAPACITY_NEED=$(awk -v a="$1" '$1==a {print $2; exit}' "$conf/capacity.tsv")
  case $CAPACITY_NEED in ''|*[!0-9]*) return 1 ;; esac
}
capacity_probe() {
  probe -q "$@"
  capacity_probe_rc=$?
  case $capacity_probe_rc in
    0) return 0 ;;
    3) die "llb-probe is missing; cannot validate this GPU; re-extract the release bundle" ;;
    *) if [ -n "${PR-}" ]; then printf '%s\n' "$PR" >&2; exit "$capacity_probe_rc"; fi
       die "GPU capability check failed; run llb doctor" ;;
  esac
}
capacity_check() {
  capacity_need "$m_alias" || die "$NAME: no tested free-memory level completed all three prompts as RUN; see capacity.tsv and its evidence"
  capacity_probe
  capacity_free
  [ "$CAPACITY_FREE" -ge "$CAPACITY_NEED" ] || die "$NAME: lowest tested level with RUN at all three prompts is $CAPACITY_NEED MiB free; $CAPACITY_FREE MiB is free. Lower levels are not established as RUN (see capacity.tsv and its evidence)"
}
capacity_pick() {
  capacity_probe
  capacity_free
  CAPACITY_PICK=
  # Table order is largest model first; skip missing models and unsupported cells.
  for candidate in $(awk '!/^#/ && NF {print $1}' "$conf/capacity.tsv"); do
    capacity_need "$candidate" || continue
    [ "$CAPACITY_FREE" -ge "$CAPACITY_NEED" ] || continue
    row "$candidate" || continue
    tok; find_alias
    [ "$NFOUND" = 1 ] || continue
    CAPACITY_PICK=$candidate
    say "selected $candidate: largest installed model allowed by the measured table ($CAPACITY_FREE MiB free; tested level $CAPACITY_NEED MiB)"
    return 0
  done
  die "no installed model has RUN at all three prompts at a tested level fitting $CAPACITY_FREE MiB free; see llb models and capacity.tsv"
}
capacity_doctor() {
  capacity_free
  can= cannot=
  for candidate in $(aliases); do
    if capacity_need "$candidate"; then
      if [ "$CAPACITY_FREE" -ge "$CAPACITY_NEED" ]; then can="${can}${can:+, }$candidate"
      else cannot="${cannot}${cannot:+; }$candidate (tested RUN at all three prompts: $CAPACITY_NEED MiB free)"; fi
    else cannot="${cannot}${cannot:+; }$candidate (no tested free-memory level completed all three prompts as RUN)"; fi
  done
  printf 'this card can run: %s; not: %s\n' "${can:-none}" "${cannot:-none}"
  printf 'Capacity evidence: RTX 3090 with ballast, prompts up to 3011 tokens; other physical cards NOT RUN.\n'
  printf 'Thresholds are conservative tested levels, not exact memory minima; receipt references are in capacity.tsv.\n'
}
