/**
 * ComfyDL rich-widget foundation (custom canvas, Legacy + Nodes 2.0).
 *
 * Public helpers:
 *   createCanvasRichWidget(node, spec) -> non-serialized custom widget
 *   redrawRichWidget(node, widget)     -> repaint both renderers
 *   isVueRenderer()                    -> current renderer probe
 *
 * spec:
 *   name/type       unique widget identity;
 *   minHeight       Legacy minimum and Vue lower bound;
 *   contentHeight() desired content height in CSS pixels;
 *   draw(ctx, box)  draw inside {node, width, y, height};
 *   mouse(event, pos, node, box) optional pointer/mouse handler.
 *
 * Persistence deliberately stays outside the widget: both serialize flags are
 * false, so the widget cannot shift schema-backed widgets_values. Small UI
 * state belongs in node.properties; execution payloads can be restored from
 * onExecuted/onConfigure. The sizing shields follow the measured obvpm
 * contract, not an assumed official API: Legacy computedHeight leaves the
 * bottom resize zone free, while Nodes 2.0 must not receive that stale
 * graph-unit value. width always defers to the node.
 *
 * First instance: CdlDatasetPreview renders its existing ui.text / saved
 * canvas_text as a scrollable canvas report. show_text.js keeps a hidden
 * serialized [head, canvas_text] state channel and remains the visible DOM
 * fallback when addCustomWidget is unavailable or mounting fails.
 */
import { app } from "/scripts/app.js";

const RESIZE_ZONE = 15;
const PREVIEW_PAD = 8;
const PREVIEW_LINE = 15;
const PREVIEW_MIN_HEIGHT = 48;
const PREVIEW_PLACEHOLDER = "(run the workflow to preview the dataset)";
const PREVIEW_PROPERTY = "cdl_preview_text";
const RESTORE_PROPERTY = "cdl_restore_widgets_values";
const HIST_PROPERTY = "cdl_histogram";
const HIST_HEIGHT = 170;
const HIST_PAD = 8;
let executedFallbackBound = false;

export function isVueRenderer() {
  return typeof LiteGraph !== "undefined" && !!LiteGraph.vueNodesMode;
}

function widgetValueStore() {
  // Nodes 2.0 keeps per-widget state (value/options) in a reactive pinia
  // store; the raw LiteGraph widget object is not observed there. This
  // mirrors the frontend's own syncPromotedComboHostOptions access path.
  try {
    const root = document.querySelector("#vue-app");
    const pinia = root?.__vue_app__?.config?.globalProperties?.$pinia;
    return pinia?._s?.get("widgetValue") ?? null;
  } catch {
    return null;
  }
}

function syncComboOptionsToVueStore(node, widget, values) {
  const store = widgetValueStore();
  if (!store || typeof store.getWidget !== "function") return false;
  try {
    const graphId = app.rootGraph?.id ?? app.graph?.rootGraph?.id;
    if (graphId == null) return false;
    const id = `${graphId}:${encodeURIComponent(String(node.id))}`
      + `:${encodeURIComponent(widget.name)}`;
    const record = store.getWidget(id);
    if (!record) return false;
    // Assigning a fresh options object (not mutating the old one) is what
    // makes the Vue renderer rebuild the dropdown; verified against the
    // WidgetSelectDefault component, which re-reads it on every open.
    record.options = { ...record.options, values };
    const current = String(record.value ?? "");
    if (current !== "" && !values.includes(current)) {
      record.value = values[0];
      widget.value = values[0];
    }
    return true;
  } catch (error) {
    console.warn("[ComfyDL.RichWidget] combo option sync failed;"
      + " the dropdown may keep its stale option list", error);
    return false;
  }
}

function ensureHistogramHoverHook(node, widget, state) {
  const nodeEl = document.querySelector(`[data-node-id="${node.id}"]`);
  const canvas = nodeEl?.querySelector("canvas") ?? null;
  if (!canvas || state.domEl === canvas) return;
  // Vue may re-create the node DOM (theme switch, renderer toggle); the
  // previous listeners died with the old element, so re-hook on the new one.
  state.domEl = canvas;
  const toWidgetPos = (event) => {
    const rect = canvas.getBoundingClientRect();
    // Page zoom is a CSS transform: getBoundingClientRect is scaled while
    // the widget box stays in layout pixels — compensate both axes.
    const scaleX = (canvas.offsetWidth || rect.width) / rect.width;
    const scaleY = (canvas.offsetHeight || rect.height) / rect.height;
    return [
      (event.clientX - rect.left) * scaleX,
      (event.clientY - rect.top) * scaleY,
    ];
  };
  // Nodes 2.0 renders the widget as a DOM canvas and never forwards
  // button-less pointer moves to widget.mouse; bridge the real DOM events
  // into the same handler so hover tracking works. Legacy rendering has no
  // per-node DOM canvas, so this stays a no-op there.
  canvas.addEventListener("pointermove", (event) => {
    widget.mouse?.({ type: "pointermove" }, toWidgetPos(event), node);
  });
  canvas.addEventListener("pointerleave", () => {
    widget.mouse?.({ type: "pointermove" }, [-1e4, -1e4], node);
  });
}

export function redrawRichWidget(node, widget) {
  node.graph?.setDirtyCanvas(true, true);
  // Nodes 2.0 paints custom widgets through WidgetLegacy's own canvas.
  widget?.triggerDraw?.();
}

function normalizedCanvasBox(ctx, width, height) {
  const canvas = ctx?.canvas;
  if (!isVueRenderer()) return [width, height];
  const wrapper = canvas?.parentElement;
  if (!wrapper || !wrapper.style?.minHeight) return [width, height];
  const layoutWidth = canvas.offsetWidth || wrapper.clientWidth;
  if (!(layoutWidth > 0) || !(width > 0)) return [width, height];
  if (Math.abs(layoutWidth - width) < 1) return [width, height];
  // WidgetLegacy allocates a zoom-dependent backing store. Match its pixel
  // ratio but draw at the CSS layout width so hit-testing and paint agree.
  const ratio = canvas.width / width || 2;
  canvas.width = Math.round(layoutWidth * ratio);
  if (height > 0) canvas.height = Math.round((height + 2) * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return [layoutWidth, height];
}

export function createCanvasRichWidget(node, spec) {
  const minHeight = Math.max(1, Number(spec.minHeight) || PREVIEW_MIN_HEIGHT);
  let allocatedHeight;

  function legacyHeight(widget, widgetY, fallback) {
    if (isVueRenderer()) return fallback;
    const nodeHeight = node.size?.[1];
    const visible = node.widgets?.filter((w) => !w.hidden);
    const isLast = !!visible && visible[visible.length - 1] === widget;
    if (nodeHeight == null || widgetY == null || !isLast) return fallback;
    return Math.max(minHeight, nodeHeight - widgetY);
  }

  const widget = {
    name: spec.name,
    type: spec.type,
    value: "",
    serialize: false,
    options: { serialize: false },
    cdlRichWidget: true,

    computeLayoutSize() {
      const wanted = Math.max(minHeight, Math.ceil(spec.contentHeight?.() || minHeight));
      if (isVueRenderer()) {
        return { minHeight: wanted, maxHeight: wanted, minWidth: 0 };
      }
      return { minHeight, maxHeight: 100000, minWidth: 0 };
    },

    draw(ctx, drawNode, width, y, height) {
      [width, height] = normalizedCanvasBox(ctx, width, height);
      const boxHeight = legacyHeight(this, y, height);
      this.cdlBox = { x: 0, y, width: width || drawNode.size[0], height: boxHeight };
      spec.draw?.call(this, ctx, {
        node: drawNode,
        width: this.cdlBox.width,
        y,
        height: boxHeight,
      });
    },

    mouse(event, pos, mouseNode) {
      if (!spec.mouse) return false;
      return !!spec.mouse.call(this, event, pos, mouseNode, this.cdlBox);
    },
  };

  node.addCustomWidget(widget);

  Object.defineProperty(widget, "computedHeight", {
    configurable: true,
    get() {
      if (isVueRenderer() || allocatedHeight == null) return undefined;
      return Math.max(0, legacyHeight(this, this.y, allocatedHeight) - RESIZE_ZONE);
    },
    set(value) {
      allocatedHeight = value;
    },
  });
  Object.defineProperty(widget, "width", {
    configurable: true,
    get: () => undefined,
    set: () => {},
  });
  return widget;
}

function previewText(value) {
  const text = Array.isArray(value) ? value.join("\n") : String(value ?? "");
  return text || PREVIEW_PLACEHOLDER;
}

function mountPreview(node) {
  if (node.__cdlRichPreview) return node.__cdlRichPreview;
  if (typeof node.addCustomWidget !== "function") return null;

  const state = {
    lines: [PREVIEW_PLACEHOLDER],
    scroll: 0,
    maxScroll: 0,
    drag: null,
    box: null,
  };

  const widget = createCanvasRichWidget(node, {
    name: "cdl_rich_preview",
    type: "cdl_rich_preview",
    minHeight: PREVIEW_MIN_HEIGHT,
    contentHeight: () => state.lines.length * PREVIEW_LINE + PREVIEW_PAD * 2,

    draw(ctx, box) {
      const pad = PREVIEW_PAD;
      const width = Math.max(0, box.width - pad * 2);
      const height = Math.max(0, box.height - 4);
      state.box = { x: pad, y: box.y, width, height };
      const room = Math.max(0, Math.floor((height - pad * 2) / PREVIEW_LINE));
      state.maxScroll = Math.max(0, state.lines.length - room);
      state.scroll = Math.max(0, Math.min(state.scroll, state.maxScroll));

      ctx.save();
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(pad, box.y, width, height, 6);
      else ctx.rect(pad, box.y, width, height);
      ctx.fillStyle = "rgba(0, 0, 0, 0.18)";
      ctx.fill();
      ctx.clip();
      ctx.font = "12px ui-monospace, Consolas, monospace";
      ctx.textBaseline = "top";
      ctx.fillStyle = LiteGraph?.NODE_TEXT_COLOR ?? "#ddd";
      for (let row = 0; row < room; row += 1) {
        const line = state.lines[state.scroll + row];
        if (line === undefined) break;
        ctx.fillText(line, pad * 2, box.y + pad + row * PREVIEW_LINE);
      }
      if (state.maxScroll > 0 && room > 0) {
        const trackX = pad + width - 4;
        const trackH = Math.max(0, height - pad * 2);
        const thumbH = Math.max(12, trackH * (room / state.lines.length));
        const travel = Math.max(0, trackH - thumbH);
        const thumbY = box.y + pad + travel * (state.scroll / state.maxScroll);
        ctx.fillStyle = "rgba(255, 255, 255, 0.22)";
        ctx.fillRect(trackX, box.y + pad, 2, trackH);
        ctx.fillStyle = "rgba(26, 188, 156, 0.8)";
        ctx.fillRect(trackX - 1, thumbY, 4, thumbH);
      }
      ctx.restore();
    },

    mouse(event, pos, _node) {
      const type = event.type;
      const box = state.box;
      if (!box) return false;
      const px = pos[0];
      const py = pos[1];
      const inside = px >= box.x && px <= box.x + box.width
        && py >= box.y && py <= box.y + box.height - RESIZE_ZONE;

      if (type === "pointerdown" || type === "mousedown") {
        if (!inside || state.maxScroll <= 0) return false;
        state.drag = { startY: py, startScroll: state.scroll };
        return true;
      }
      if (!state.drag) return false;
      if (type === "pointermove" || type === "mousemove") {
        const delta = (state.drag.startY - py) / PREVIEW_LINE;
        state.scroll = Math.max(0, Math.min(
          state.maxScroll,
          Math.round(state.drag.startScroll + delta),
        ));
        redrawRichWidget(node, widget);
        return true;
      }
      if (type === "pointerup" || type === "mouseup") {
        state.drag = null;
        return true;
      }
      return false;
    },
  });

  node.__cdlRichPreview = { widget, state };
  return node.__cdlRichPreview;
}

function showPreview(node, value) {
  const preview = mountPreview(node);
  if (!preview) return;
  // previewText already normalizes arrays to one string.
  const text = previewText(value);
  // Idempotence: the onExecuted path and the direct `executed` event fallback
  // can both deliver the same payload. Re-applying identical text would
  // needlessly reset the scroll position, so unchanged text is a no-op.
  if (preview.state.lines.length > 0
    && preview.state.lines.join("\n") === text
    && node.properties?.[PREVIEW_PROPERTY] === text) {
    return;
  }
  preview.state.lines = text.split("\n");
  preview.state.scroll = 0;
  preview.state.drag = null;
  node.properties ??= {};
  node.properties[PREVIEW_PROPERTY] = text;
  redrawRichWidget(node, preview.widget);
}

function findRichNode(rawId) {
  if (rawId == null) return null;
  // The raw event carries node ids as strings; Vue-mode getNodeById only
  // resolves numbers. Mirror the frontend's normalization tolerance.
  const graph = app.graph;
  let node = graph?.getNodeById?.(rawId);
  if (!node && typeof rawId === "string" && /^\d+$/.test(rawId)) {
    node = graph?.getNodeById?.(Number(rawId));
  }
  if (!node && Array.isArray(graph?._nodes)) {
    node = graph._nodes.find((n) => String(n.id) === String(rawId));
  }
  return node;
}

function updatePreviewFromEvent(event) {
  // addEventListener hands over the CustomEvent itself — the payload lives
  // on event.detail (the official dispatch destructures the same way).
  const detail = event?.detail;
  const output = detail?.output;
  if (!output) return;
  const node = findRichNode(detail.display_node ?? detail.node);
  if (!node) return;
  try {
    if (output.text) {
      showPreview(node, output.text);
    } else if (output.cdl_histogram) {
      const payload = Array.isArray(output.cdl_histogram)
        ? output.cdl_histogram[0] : output.cdl_histogram;
      showHistogram(node, payload);
    }
  } catch (error) {
    console.warn("[ComfyDL.RichWidget] event update failed", error);
  }
}

function mountHistogram(node) {
  if (node.__cdlRichHistogram) return node.__cdlRichHistogram;
  if (typeof node.addCustomWidget !== "function") return null;

  const state = { series: null, feature: null, hover: null, box: null,
    domEl: null };

  function histogramPlot() {
    const box = state.box;
    return {
      x: box.x + HIST_PAD,
      y: box.y + 26,
      w: box.width - HIST_PAD * 2,
      h: Math.max(10, box.height - 26 - 18 - 4),
    };
  }

  const fmtValue = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    if (Math.abs(n) >= 1000 || (Math.abs(n) < 0.001 && n !== 0)) {
      return n.toPrecision(4);
    }
    return String(Math.round(n * 10000) / 10000);
  };

  const widget = createCanvasRichWidget(node, {
    name: "cdl_rich_histogram",
    type: "cdl_rich_histogram",
    minHeight: HIST_HEIGHT,
    contentHeight: () => HIST_HEIGHT,

    draw(ctx, box) {
      ensureHistogramHoverHook(node, this, state);
      const width = Math.max(0, box.width - HIST_PAD * 2);
      const height = Math.max(0, box.height - 4);
      state.box = { x: HIST_PAD, y: box.y, width, height };
      ctx.save();
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(HIST_PAD, box.y, width, height, 6);
      else ctx.rect(HIST_PAD, box.y, width, height);
      ctx.fillStyle = "rgba(0, 0, 0, 0.18)";
      ctx.fill();
      ctx.clip();
      const p = state.series && state.feature
        ? state.series[state.feature] : null;
      if (!p || !Array.isArray(p.counts) || !p.counts.length) {
        ctx.font = "12px ui-monospace, Consolas, monospace";
        ctx.textBaseline = "top";
        ctx.fillStyle = LiteGraph?.NODE_TEXT_COLOR ?? "#ddd";
        const hint = p && p.n_total === 0
          ? `(column "${state.feature}" has 0 samples)`
          : p
            ? `(column "${state.feature}" has no finite values`
              + `${p.n_dropped ? `, ${p.n_dropped} NaN/Inf dropped` : ""})`
            : "(run the workflow to plot the histogram)";
        ctx.fillText(hint, HIST_PAD * 2, box.y + HIST_PAD * 2);
        ctx.restore();
        return;
      }

      const counts = p.counts;
      const edges = Array.isArray(p.edges) ? p.edges : [];
      const plot = histogramPlot();
      const peak = Math.max(1, ...counts);
      const binW = plot.w / counts.length;
      for (let i = 0; i < counts.length; i += 1) {
        const barH = (counts[i] / peak) * plot.h;
        if (barH <= 0) continue;
        ctx.fillStyle = state.hover === i
          ? "rgba(26, 188, 154, 0.95)"
          : "rgba(26, 188, 154, 0.55)";
        ctx.fillRect(plot.x + i * binW + 0.5, plot.y + plot.h - barH,
          Math.max(1, binW - 1), barH);
      }

      ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
      ctx.beginPath();
      ctx.moveTo(plot.x, plot.y + plot.h + 0.5);
      ctx.lineTo(plot.x + plot.w, plot.y + plot.h + 0.5);
      ctx.stroke();
      ctx.font = "10px ui-monospace, Consolas, monospace";
      ctx.textBaseline = "top";
      ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
      ctx.fillText(fmtValue(p.min), plot.x, plot.y + plot.h + 4);
      ctx.textAlign = "right";
      ctx.fillText(fmtValue(p.max), plot.x + plot.w, plot.y + plot.h + 4);
      ctx.textAlign = "left";

      const dropped = p.n_dropped ? ` \u00b7 dropped ${p.n_dropped}` : "";
      const sampled = p.sampled ? " \u00b7 sampled" : "";
      ctx.font = "12px ui-monospace, Consolas, monospace";
      ctx.fillStyle = LiteGraph?.NODE_TEXT_COLOR ?? "#ddd";
      ctx.fillText(`${state.feature}  n=${p.n_used}${sampled}${dropped}`
        + `  mean=${fmtValue(p.mean)}  std=${fmtValue(p.std)}`,
        HIST_PAD * 2, box.y + 6);

      if (state.hover != null && state.hover >= 0
        && state.hover < counts.length) {
        const i = state.hover;
        const cx = plot.x + i * binW + binW / 2;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
        ctx.beginPath();
        ctx.moveTo(cx, plot.y);
        ctx.lineTo(cx, plot.y + plot.h);
        ctx.stroke();
        const lo = edges[i] ?? p.min;
        const hi = edges[i + 1] ?? p.max;
        const text = `[${fmtValue(lo)} ~ ${fmtValue(hi)})  count=${counts[i]}`;
        ctx.font = "11px ui-monospace, Consolas, monospace";
        const textW = ctx.measureText(text).width;
        const tx = Math.max(plot.x,
          Math.min(cx - textW / 2, plot.x + plot.w - textW));
        ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
        ctx.fillRect(tx - 4, plot.y + 2, textW + 8, 16);
        ctx.fillStyle = "#1abc9c";
        ctx.fillText(text, tx, plot.y + 5);
      }
      ctx.restore();
    },

    mouse(event, pos) {
      const box = state.box;
      if (!box) return false;
      const inside = pos[0] >= box.x && pos[0] <= box.x + box.width
        && pos[1] >= box.y && pos[1] <= box.y + box.height;
      if (event.type !== "pointermove" && event.type !== "mousemove") {
        return false;
      }
      const p = state.series && state.feature
        ? state.series[state.feature] : null;
      if (!inside || !p || !Array.isArray(p.counts) || !p.counts.length) {
        if (state.hover != null) {
          state.hover = null;
          redrawRichWidget(node, widget);
        }
        return false;
      }
      const plot = histogramPlot();
      const rel = (pos[0] - plot.x) / Math.max(1, plot.w);
      const index = Math.max(0, Math.min(p.counts.length - 1,
        Math.floor(rel * p.counts.length)));
      if (state.hover !== index) {
        state.hover = index;
        redrawRichWidget(node, widget);
      }
      return true;
    },
  });

  node.__cdlRichHistogram = { widget, state };
  return node.__cdlRichHistogram;
}

function refreshHistogramCombo(node, payload) {
  const widget = node.widgets?.find((w) => w.name === "feature");
  if (!widget) return;
  const series = payload && payload.series ? payload.series : null;
  if (!series) return;
  const values = ["*"].concat(Object.keys(series));
  widget.options ??= {};
  widget.options.values = values;
  // Legacy canvas renderer reads the raw widget; Nodes 2.0 needs the store.
  syncComboOptionsToVueStore(node, widget, values);
  const current = String(widget.value ?? "");
  if (current !== "*" && !values.includes(current)) {
    widget.value = values.includes(String(payload.feature))
      ? String(payload.feature) : "*";
  }
}

function normalizeHistogramPayload(payload) {
  // Accept the multi-column payload ({series: {name: entry}}) and the
  // pre-0.82g single-column one ({counts: [...]}) so restored workflows
  // saved by older builds still render.
  if (!payload || typeof payload !== "object") return null;
  if (payload.series && typeof payload.series === "object") {
    const series = {};
    const names = Array.isArray(payload.feature_names)
      ? payload.feature_names.map(String) : [];
    for (const name of names) {
      if (payload.series[name]) series[name] = payload.series[name];
    }
    for (const key of Object.keys(payload.series)) {
      if (!series[key]) series[key] = payload.series[key];
    }
    if (!Object.keys(series).length) return null;
    return { ...payload, series };
  }
  if (Array.isArray(payload.counts)) {
    const entry = { ...payload };
    const name = String(payload.feature ?? "*");
    delete entry.feature;
    delete entry.feature_names;
    return {
      feature: name,
      feature_names: [name],
      series: { [name]: entry },
    };
  }
  return null;
}

function showHistogram(node, payload) {
  const hist = mountHistogram(node);
  if (!hist) return;
  const body = normalizeHistogramPayload(payload);
  if (!body) return;
  hist.state.series = body.series;
  hist.state.hover = null;
  // Show the column the widget currently asks for (instant combo switch),
  // falling back to the execution-time selection, then the first column.
  const widget = node.widgets?.find((w) => w.name === "feature");
  const wanted = String(widget?.value ?? "");
  const keys = Object.keys(body.series);
  hist.state.feature = body.series[wanted]
    ? wanted
    : (body.series[body.feature] ? body.feature : keys[0]);
  node.properties ??= {};
  node.properties[HIST_PROPERTY] = body;
  refreshHistogramCombo(node, body);
  redrawRichWidget(node, hist.widget);
}

function wireHistogramNode(nodeType) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = onNodeCreated?.apply(this, arguments);
    try {
      mountHistogram(this);
      // Both renderers invoke widget.callback when the combo value changes
      // (Vue does it through createWidgetUpdateHandler). The payload
      // already carries every column, so a selection change can redraw
      // instantly from the cached series without re-running the workflow.
      const feature = this.widgets?.find((w) => w.name === "feature");
      if (feature && !feature.__cdlHistFeatureHook) {
        feature.__cdlHistFeatureHook = true;
        const original = feature.callback;
        feature.callback = (value) => {
          const callbackResult = original?.call(feature, value);
          const saved = this.properties?.[HIST_PROPERTY];
          if (saved) {
            try {
              showHistogram(this, saved);
            } catch (error) {
              console.warn("[ComfyDL.RichWidget] histogram switch failed",
                error);
            }
          }
          return callbackResult;
        };
      }
    } catch (error) {
      console.warn("[ComfyDL.RichWidget] histogram canvas unavailable;"
        + " the node still works without the chart", error);
    }
    return result;
  };

  const onExecuted = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (message) {
    const result = onExecuted?.apply(this, arguments);
    const payload = message?.cdl_histogram;
    if (payload) {
      try {
        showHistogram(this, Array.isArray(payload) ? payload[0] : payload);
      } catch (error) {
        console.warn("[ComfyDL.RichWidget] histogram update failed", error);
      }
    }
    return result;
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const result = onConfigure?.apply(this, arguments);
    setTimeout(() => {
      try {
        const saved = this.properties?.[HIST_PROPERTY];
        if (saved) showHistogram(this, saved);
      } catch (error) {
        console.warn("[ComfyDL.RichWidget] histogram restore failed", error);
      }
    }, 0);
    return result;
  };
}

function stampPreviewRestoreValues(nodes) {
  for (const node of nodes || []) {
    if (node.type === "CdlDatasetPreview" && Array.isArray(node.widgets_values)) {
      node.properties ??= {};
      // loadGraphData clones the workflow before this hook. Keep the raw
      // values on that clone only; loadedGraphNode deletes this temporary
      // property after repairing the migrated widget values.
      node.properties[RESTORE_PROPERTY] = [...node.widgets_values];
    }
  }
}

app.registerExtension({
  name: "ComfyDL.RichWidget",

  beforeConfigureGraph(graphData) {
    stampPreviewRestoreValues(graphData?.nodes);
    for (const subgraph of graphData?.definitions?.subgraphs || []) {
      stampPreviewRestoreValues(subgraph.nodes);
    }
  },

  setup() {
    // Belt and braces: node.onExecuted is the official delivery path, but
    // some page states / frontend builds do not reach it (observed once on a
    // long-lived real session). The raw `executed` event carries the same
    // payload; showPreview is idempotent, so double delivery is harmless.
    // app.api can still be undefined when setup() runs — optional chaining
    // would silently skip the subscription, so bind now or retry briefly.
    if (executedFallbackBound || typeof app.api?.addEventListener !== "function") {
      if (!executedFallbackBound) {
        const retry = setInterval(() => {
          if (typeof app.api?.addEventListener === "function") {
            clearInterval(retry);
            app.api.addEventListener("executed", updatePreviewFromEvent);
            executedFallbackBound = true;
          }
        }, 200);
        setTimeout(() => clearInterval(retry), 10000);
      }
      return;
    }
    app.api.addEventListener("executed", updatePreviewFromEvent);
    executedFallbackBound = true;
  },

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "CdlDatasetHistogram") {
      wireHistogramNode(nodeType);
      return;
    }
    if (nodeData.name !== "CdlDatasetPreview") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      try {
        mountPreview(this);
      } catch (error) {
        console.warn("[ComfyDL.RichWidget] canvas preview unavailable; using ShowText", error);
      }
      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = onExecuted?.apply(this, arguments);
      if (message && message.text) {
        try {
          showPreview(this, message.text);
        } catch (error) {
          console.warn("[ComfyDL.RichWidget] preview update failed", error);
        }
      }
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = onConfigure?.apply(this, arguments);
      setTimeout(() => {
        try {
          const raw = this.properties?.[RESTORE_PROPERTY];
          if (this.properties) delete this.properties[RESTORE_PROPERTY];
          const head = this.widgets?.find((w) => w.name === "head");
          if (head && raw?.length) head.value = raw[0];
          const saved = this.properties?.[PREVIEW_PROPERTY]
            ?? raw?.slice(1)
            ?? info?.widgets_values?.[1]
            ?? this.widgets_values?.[1];
          if (saved) showPreview(this, saved);
        } catch (error) {
          console.warn("[ComfyDL.RichWidget] preview restore failed", error);
        }
      }, 0);
      return result;
    };
  },

  loadedGraphNode(node) {
    if (node.comfyClass === "CdlDatasetHistogram") {
      setTimeout(() => {
        try {
          const saved = node.properties?.[HIST_PROPERTY];
          if (saved) showHistogram(node, saved);
        } catch (error) {
          console.warn("[ComfyDL.RichWidget] histogram restore failed", error);
        }
      }, 0);
      return;
    }
    if (node.comfyClass !== "CdlDatasetPreview") return;
    setTimeout(() => {
      try {
        const raw = node.properties?.[RESTORE_PROPERTY];
        if (node.properties) delete node.properties[RESTORE_PROPERTY];
        const head = node.widgets?.find((w) => w.name === "head");
        if (head && raw?.length) head.value = raw[0];
        const state = node.widgets?.find((w) => w.cdlShowText);
        const stateText = raw?.length > 1 ? raw.slice(1) : state?.value;
        if (state && raw?.length > 1) state.value = previewText(raw.slice(1));
        const saved = node.properties?.[PREVIEW_PROPERTY] ?? stateText;
        if (saved) showPreview(node, saved);
      } catch (error) {
        console.warn("[ComfyDL.RichWidget] loaded-workflow restore failed", error);
      }
    }, 0);
  },
});
