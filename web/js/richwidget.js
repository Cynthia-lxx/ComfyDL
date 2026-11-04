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

export function isVueRenderer() {
  return typeof LiteGraph !== "undefined" && !!LiteGraph.vueNodesMode;
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
  preview.state.lines = previewText(value).split("\n");
  preview.state.scroll = 0;
  preview.state.drag = null;
  node.properties ??= {};
  node.properties[PREVIEW_PROPERTY] = preview.state.lines.join("\n");
  redrawRichWidget(node, preview.widget);
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

  beforeRegisterNodeDef(nodeType, nodeData) {
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
