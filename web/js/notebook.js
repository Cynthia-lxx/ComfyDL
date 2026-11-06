/**
 * ComfyDL Notebook (notebook.js) - cell UI for CdlNotebookCell.
 *
 * Contract with the backend (comfydl/nodes/notebook.py):
 *   - ui payload = {"cdl_notebook_log": [stdout], "cdl_notebook_result": [repr]}
 *     (final channel; the same cdl_notebook_log key also arrives streamed
 *     through raw `executed` WS events while the cell runs)
 *   - outputs: result (*) / plot (IMAGE) / stdout (STRING)
 *
 * Behaviour:
 *   - a code textarea (DOM widget) bound to the serialized `code` value;
 *   - a richwidget canvas showing the streaming log + the result line, with
 *     a permanent amber banner reminding that cell code executes WITHOUT a
 *     sandbox and the feature is gated behind the comfydl.notebook switch
 *     (default off);
 *   - onExecuted / executed-event / onConfigure restore, idempotent.
 *
 * Containment: only the CdlNotebookCell prototype is patched. Reduced-motion:
 * no animation of any kind.
 */
import { app } from "/scripts/app.js";
import { createCanvasRichWidget } from "./richwidget.js";

const LOG_PROPERTY = "cdl_notebook_log";
const BANNER_HEIGHT = 26;
const LOG_MIN_HEIGHT = 96;
const LOG_LINE = 15;
const MAX_LINES = 400;

const BANNER_TEXT = "\u26a0 exec without sandbox \u2014 gated by the"
  + " comfydl.notebook switch (off by default)";

function mountNotebookOutput(node) {
  if (node.__cdlNotebookOutput) return node.__cdlNotebookOutput;
  if (typeof node.addCustomWidget !== "function") return null;

  const state = { lines: ["(run the cell to see its output)"], box: null };

  const widget = createCanvasRichWidget(node, {
    name: "cdl_notebook_output",
    type: "cdl_notebook_output",
    minHeight: LOG_MIN_HEIGHT + BANNER_HEIGHT,
    contentHeight: () => BANNER_HEIGHT
      + Math.max(LOG_MIN_HEIGHT,
        state.lines.length * LOG_LINE + 16),

    draw(ctx, box) {
      const pad = 8;
      const width = Math.max(0, box.width - pad * 2);
      state.box = { x: pad, y: box.y, width, height: box.height };

      ctx.save();
      // Amber banner: permanent Tier-2 warning, always the first strip.
      ctx.fillStyle = "rgba(249, 115, 22, 0.16)";
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(pad, box.y + 2, width, BANNER_HEIGHT - 4, 5);
      else ctx.rect(pad, box.y + 2, width, BANNER_HEIGHT - 4);
      ctx.fill();
      ctx.font = "11px ui-monospace, Consolas, monospace";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#f59e0b";
      ctx.fillText(BANNER_TEXT, pad + 8, box.y + BANNER_HEIGHT / 2);

      // Log area.
      const logY = box.y + BANNER_HEIGHT + 4;
      const logH = Math.max(0, box.height - BANNER_HEIGHT - 8);
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(pad, logY, width, logH, 6);
      else ctx.rect(pad, logY, width, logH);
      ctx.fillStyle = "rgba(0, 0, 0, 0.22)";
      ctx.fill();
      ctx.clip();

      ctx.font = "12px ui-monospace, Consolas, monospace";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#d1d5db";
      const room = Math.max(0, Math.floor((logH - 10) / LOG_LINE));
      const tail = state.lines.slice(-room);
      for (let row = 0; row < tail.length; row += 1) {
        ctx.fillText(tail[row], pad + 8, logY + 6 + row * LOG_LINE);
      }
      ctx.restore();
    },
  });

  node.__cdlNotebookOutput = { widget, state };
  return node.__cdlNotebookOutput;
}

function extractLog(output) {
  // onExecuted hands over the output dict itself on current builds; older
  // ones (and the raw executed event) nest it under message.output. Accept
  // both shapes — verbose-verified: top-level on this build, but the event
  // path and future updates must not depend on the layout.
  const log = output?.cdl_notebook_log;
  const repr = output?.cdl_notebook_result;
  return {
    text: Array.isArray(log) ? log[0] : log,
    result: Array.isArray(repr) ? repr[0] : repr,
  };
}

function setLog(node, text, resultRepr) {
  const output = mountNotebookOutput(node);
  if (!output) return;
  const lines = String(text ?? "").replace(/\r/g, "").split("\n");
  if (lines.length > MAX_LINES) {
    lines.splice(0, lines.length - MAX_LINES,
      `(... ${lines.length - MAX_LINES} earlier lines trimmed ...)`);
  }
  if (lines.length === 0) lines.push("[no output]");
  // Echo the result socket's value and type on the console: `result` is a
  // wildcard, and feeding e.g. a number into a STRING target (SaveText)
  // fails at that node — seeing "[Result] 90" here makes the mismatch
  // obvious at the source instead.
  if (resultRepr !== undefined && resultRepr !== null) {
    lines.push(`[Result] ${resultRepr}`);
  }
  output.state.lines = lines;
  node.graph?.setDirtyCanvas(true, true);
  output.widget?.triggerDraw?.();
}

function wireNotebookNode(nodeType) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = onNodeCreated?.apply(this, arguments);
    try {
      mountNotebookOutput(this);
    } catch (error) {
      console.warn("[ComfyDL.Notebook] cell output widget unavailable;"
        + " the node still works without the console view", error);
    }
    return result;
  };

  const onExecuted = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (message) {
    const result = onExecuted?.apply(this, arguments);
    const { text, result: resultRepr } = extractLog(message?.output ?? message);
    if (text) {
      try {
        setLog(this, text, resultRepr);
      } catch (error) {
        console.warn("[ComfyDL.Notebook] log update failed", error);
      }
    }
    return result;
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function (info) {
    const result = onConfigure?.apply(this, arguments);
    try {
      const saved = this.properties?.[LOG_PROPERTY];
      if (saved) setLog(this, saved);
    } catch (error) {
      console.warn("[ComfyDL.Notebook] log restore failed", error);
    }
    return result;
  };
}

function isNotebookNode(node) {
  // comfyClass is the LiteGraph-side name; some graph states (Vue-rendered
  // nodes restored from storage) only carry `type`. Accept both.
  return (node?.comfyClass === "CdlNotebookCell"
    || node?.type === "CdlNotebookCell");
}

function findNotebookNode(rawId) {
  const wanted = String(rawId);
  const nodes = app.graph?._nodes ?? [];
  for (const node of nodes) {
    if (isNotebookNode(node) && String(node.id) === wanted) {
      return node;
    }
  }
  return null;
}

let streamedFallbackBound = false;

function onStreamedEvent(event) {
  const detail = event?.detail;
  const { text, result: resultRepr } = extractLog(detail?.output);
  if (!text) return;
  const node = findNotebookNode(detail?.display_node ?? detail?.node);
  if (!node) return;
  try {
    setLog(node, text, resultRepr);
  } catch (error) {
    console.warn("[ComfyDL.Notebook] streamed log update failed", error);
  }
}

app.registerExtension({
  name: "ComfyDL.Notebook",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === "CdlNotebookCell") {
      wireNotebookNode(nodeType);
    }
  },
  setup() {
    // Belt and braces: stream stdout even when onExecuted does not fire
    // per-chunk. Same payload shape, idempotent setter — double delivery
    // is harmless (mirrors richwidget's executed-event fallback).
    const bind = () => {
      if (streamedFallbackBound || typeof app.api?.addEventListener !== "function") {
        return false;
      }
      app.api.addEventListener("executed", onStreamedEvent);
      streamedFallbackBound = true;
      return true;
    };
    if (!bind()) {
      // Retry indefinitely at low frequency: app.api can late-bind long
      // after setup on restored sessions, and a permanently unbound
      // fallback is exactly how "console never refreshes" reports happen.
      const retry = setInterval(() => {
        if (bind()) clearInterval(retry);
      }, 2000);
    }
  },
});
