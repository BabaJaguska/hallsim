/* The baked lever page. index.json describes the page; one JSON per
   variant and slider setting holds the series, fetched as the sliders
   move. The row layouts come baked, every shaded period in under its
   name; the setting decides which stay. */
(async function () {
  "use strict";
  const index = await (await fetch("index.json")).json();
  const app = document.getElementById("app");
  const cache = new Map();
  const refs = new Map();
  const state = {
    variant: index.variants[0].slug,
    sev: index.presets[index.reference].slice(),
    cols: columns(),
  };
  let pending = 0;

  function columns() {
    const wide = index.columns[0];
    const narrow = index.columns[index.columns.length - 1];
    return window.innerWidth > 600 ? wide : narrow;
  }
  function clean(v) {
    return Math.round(Number(v) * 1e6) / 1e6 + 0;
  }
  function key(sev) {
    return sev.map((v) => clean(v).toFixed(2)).join("_");
  }
  function fmt(lever, v) {
    return (lever.signed && v >= 0 ? "+" : "") + Number(v).toFixed(2);
  }
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const c of children || []) {
      node.appendChild(
        typeof c === "string" ? document.createTextNode(c) : c
      );
    }
    return node;
  }
  function severities() {
    const out = {};
    index.levers.forEach((lv, i) => {
      out[lv.handle] = state.sev[i];
    });
    return out;
  }
  function shadeOn(sh, sev) {
    if (!sh.lever) return true;
    const s = sev[sh.lever] || 0;
    if (sh.when === "positive") return s > 0;
    if (sh.when === "negative") return s < 0;
    return s !== 0;
  }
  function shadeName(sh, sev) {
    const below = (sev[sh.lever] || 0) < 0;
    return sh.negative_label && below ? sh.negative_label : sh.label;
  }

  const brand = el("div", { class: "brand" }, [
    el("h1", { id: "brand", text: index.title }),
  ]);
  if (index.tagline) {
    brand.appendChild(el("div", { class: "tag", text: index.tagline }));
  }
  app.appendChild(el("div", { class: "top" }, [brand]));

  const side = el("div", { class: "side" });
  const chips = el("div", { class: "chips" });
  const chipOf = {};
  for (const name of Object.keys(index.presets)) {
    const chip = el("button", {
      class: "chip",
      text: name,
      onclick: () => setSeverities(index.presets[name]),
    });
    chipOf[name] = chip;
    chips.appendChild(chip);
  }
  side.appendChild(chips);
  const sliders = [];
  const values = [];
  index.levers.forEach((lv, i) => {
    const val = el("span", { class: "val" });
    const title = lv.handle + (lv.description ? ": " + lv.description : "");
    const h3 = el("h3", { title: title }, [lv.name, val]);
    const input = el("input", {
      type: "range",
      min: lv.lo,
      max: lv.hi,
      step: index.step,
      value: state.sev[i],
    });
    input.addEventListener("input", () => {
      val.textContent = fmt(lv, input.value);
    });
    input.addEventListener("change", () => {
      state.sev[i] = clean(input.value);
      render();
    });
    const marks = el("div", { class: "marks" });
    for (let k = 0; k < 5; k++) {
      const m = lv.lo + ((lv.hi - lv.lo) * k) / 4;
      marks.appendChild(el("span", { text: String(Math.round(m * 100) / 100) }));
    }
    side.appendChild(el("div", { class: "card" }, [h3, input, marks]));
    sliders.push(input);
    values.push(val);
  });
  const segOf = {};
  if (index.variants.length > 1) {
    const group = el("div", { class: "seg-group" });
    for (const v of index.variants) {
      const seg = el("button", { class: "seg" }, [
        el("span", { class: "seg-top", text: v.top }),
        el("span", { class: "seg-sub", text: v.sub }),
      ]);
      seg.addEventListener("click", () => {
        state.variant = v.slug;
        render();
      });
      segOf[v.slug] = seg;
      group.appendChild(seg);
    }
    side.appendChild(
      el("div", { class: "card" }, [
        el("h3", { text: index.variant_title }),
        group,
      ])
    );
  }
  if (index.about.length) {
    side.appendChild(
      el("div", { class: "about" }, index.about.map((p) => el("p", { text: p })))
    );
  }
  if (index.logos.length) {
    side.appendChild(
      el(
        "div",
        { class: "logos" },
        index.logos.map(([f, alt]) => el("img", { src: "assets/" + f, alt: alt }))
      )
    );
  }

  const legend = el("div", { class: "legend" });
  const main = el("div", { class: "main" }, [legend]);
  const panelOf = {};
  for (const row of index.rows) {
    const h2 = el("h2", {}, [row.title]);
    if (row.deposit) {
      h2.appendChild(el("span", { class: "dep", text: "  " + row.deposit }));
    }
    const panel = el("div", { class: "panel" });
    panelOf[row.name] = panel;
    main.appendChild(el("div", { class: "row" }, [h2, panel]));
  }
  app.appendChild(el("div", { class: "wrap" }, [side, main]));

  function load(url, store, id) {
    if (!store.has(id)) {
      store.set(
        id,
        fetch(url).then((r) => {
          if (!r.ok) throw new Error(url + " " + r.status);
          return r.json();
        })
      );
    }
    return store.get(id);
  }
  function referenceOf(slug) {
    return load("reference/" + slug + ".json", refs, slug);
  }
  function dataOf(slug, k) {
    return load("data/" + slug + "/" + k + ".json", cache, slug + "/" + k);
  }

  function axes(j) {
    return j === 0 ? {} : { xaxis: "x" + (j + 1), yaxis: "y" + (j + 1) };
  }
  function line(t, y, style, hover, j) {
    const tr = Object.assign(
      { type: "scatter", mode: "lines", x: t, y: y, line: style },
      axes(j)
    );
    if (hover) tr.hovertemplate = hover;
    else tr.hoverinfo = "skip";
    return tr;
  }
  function band(t, lo, hi, fill, j) {
    const edge = (y, tonext) =>
      Object.assign(
        {
          type: "scatter",
          mode: "lines",
          x: t,
          y: y,
          line: { width: 0 },
          fillcolor: fill,
          hoverinfo: "skip",
        },
        tonext ? { fill: "tonexty" } : {},
        axes(j)
      );
    return [edge(hi, false), edge(lo, true)];
  }
  function traces(row, ref, cur) {
    const t = ref.t;
    const s = index.style;
    const current = { color: row.color, width: s.curve_width };
    const out = [];
    row.panels.forEach((_, j) => {
      const pop = cur.population[row.name];
      if (pop) {
        const c = ref.population[row.name][j];
        const p = pop[j];
        out.push(...band(t, c.lo, c.hi, s.control_band, j));
        out.push(...band(t, p.lo, p.hi, row.band, j));
        out.push(line(t, c.mean, s.reference_line, null, j));
        out.push(line(t, p.mean, current, s.hover, j));
      } else {
        const r = ref.series[row.name][j];
        out.push(line(t, r, s.reference_line, null, j));
        out.push(line(t, cur.series[row.name][j], current, s.hover, j));
      }
    });
    return out;
  }
  function layoutFor(row, sev) {
    const base = index.layouts[state.variant][String(state.cols)][row.name];
    const layout = JSON.parse(JSON.stringify(base));
    const on = new Set(
      index.shades[state.variant]
        .filter((sh) => shadeOn(sh, sev))
        .map((sh) => sh.label)
    );
    layout.shapes = (layout.shapes || []).filter((sh) => on.has(sh.name));
    return layout;
  }
  function renderLegend(sev) {
    const items = [
      el("span", {}, [el("span", { class: "ln" }), "current setting"]),
      el("span", {}, [
        el("span", { class: "ln dotted" }),
        index.reference_label,
      ]),
    ];
    for (const sh of index.shades[state.variant]) {
      if (!shadeOn(sh, sev)) continue;
      items.push(
        el("span", {}, [
          el("span", { class: "sw", style: "background:" + sh.color }),
          shadeName(sh, sev),
        ])
      );
    }
    legend.replaceChildren(...items);
  }
  async function render() {
    const k = key(state.sev);
    const slug = state.variant;
    const id = ++pending;
    const sev = severities();
    index.levers.forEach((lv, i) => {
      values[i].textContent = fmt(lv, state.sev[i]);
      sliders[i].value = state.sev[i];
    });
    for (const [name, chip] of Object.entries(chipOf)) {
      const selected = key(index.presets[name]) === k;
      chip.className = selected ? "chip selected" : "chip";
    }
    for (const [s, seg] of Object.entries(segOf)) {
      seg.className = s === slug ? "seg selected" : "seg";
    }
    renderLegend(sev);
    let ref;
    let cur;
    try {
      [ref, cur] = await Promise.all([referenceOf(slug), dataOf(slug, k)]);
    } catch (e) {
      console.error(e);
      return;
    }
    if (id !== pending) return;
    for (const row of index.rows) {
      Plotly.react(
        panelOf[row.name],
        traces(row, ref, cur),
        layoutFor(row, sev),
        { displayModeBar: false, responsive: true }
      );
    }
  }
  function setSeverities(vals) {
    state.sev = vals.map(clean);
    render();
  }
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      const c = columns();
      if (c !== state.cols) {
        state.cols = c;
        render();
      }
    }, 150);
  });
  render();
})();
