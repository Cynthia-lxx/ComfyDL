/**
 * ComfyDL ShowText (show_text.js) - canvas-inline preview for CdlDatasetPreview.
 *
 * Contract with the backend (comfydl/nodes/data_io_writers.py):
 *   - widgets_values = [head, canvas_text]  (written back on execution;
 *     index 0 is the head INT widget, index 1 the display text)
 *   - ui payload = {"text": [canvas_text]}  (live channel, same text)
 *
 * Behaviour (pythongosssss ShowText pattern, trimmed):
 *   - onExecuted: rebuild the display widgets from the ui payload;
 *   - onConfigure: restore from widgets_values[1:] (saved-workflow load);
 *   - display widgets are read-only, monospace, dimmed, and grow with the
 *     text (bounded) so the table is readable inside the node body.
 *
 * Containment: only the CdlDatasetPreview prototype is patched; every other
 * node is untouched. Reduced-motion: no animation of any kind.
 *
 * In: extension module (served from /extensions/comfydl/js/show_text.js).
 * Out: registered node behaviour patch.
 */
import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const MAX_WIDGET_LINES = 50;

function isDisplayWidget(w) {
  return !!w && !!w.cdlShowText;
}

function removeDisplayWidgets(node) {
  // splice from the end so earlier indices stay valid
  for (let i = node.widgets.length - 1; i >= 1; i -= 1) {
    if (isDisplayWidget(node.widgets[i])) node.widgets.splice(i, 1);
  }
}

function populate(node, text) {
  const richActive = !!node.__cdlRichPreview?.widget;
  // text arrives as the ui payload's list (one entry: the whole clipped
  // table) or as a saved string — normalize to one string.
  const value = Array.isArray(text) ? text.join("\n") : String(text ?? "");
  removeDisplayWidgets(node);

  const w = ComfyWidgets.STRING(
    node, "cdl_show_text", ["STRING", { multiline: true }], app
  ).widget;
  w.cdlShowText = true;
  if (richActive) {
    // Keep [head, canvas_text] schema-backed even though the canvas is the
    // visible renderer. Both hidden marks are required by the two renderers;
    // Nodes 2.0 also needs zero layout and a real display:none on the DOM row.
    w.hidden = true;
    w.options.hidden = true;
    w.computeSize = () => [0, -4];
    w.computeLayoutSize = () => ({
      minHeight: 0, maxHeight: 0, minWidth: 0, maxWidth: 0,
    });
  }
  // Read-only must survive the frontend re-creating the DOM element
  // (Nodes 2.0 re-renders widgets): apply property AND attribute, kill the
  // tab stop, and re-assert one frame later in case the element is swapped
  // after creation.  populate() re-runs on every execution, so the display
  // heals itself even if an edit slips through between runs — the backend
  // write-back is the source of truth.
  const apply = () => {
    const el = w.inputEl;
    if (!el) return;
    el.readOnly = true;
    el.setAttribute("readonly", "");
    el.tabIndex = -1;
    el.style.opacity = "0.6";
    el.style.fontFamily = "monospace";
    if (richActive) {
      el.style.display = "none";
    } else {
      const lines = Math.min(value.split("\n").length + 1, MAX_WIDGET_LINES);
      el.style.height = Math.min(28 + lines * 14, 760) + "px";
    }
  };
  apply();
  requestAnimationFrame(apply);
  w.value = value;
  if (richActive) {
    // A non-serialized widget before a serialized one is saved as a null
    // placeholder on some frontend versions. Keep the rich renderer last so
    // the persisted contract remains exactly [head, canvas_text].
    const rich = node.__cdlRichPreview.widget;
    const index = node.widgets.indexOf(rich);
    if (index >= 0 && index !== node.widgets.length - 1) {
      node.widgets.splice(index, 1);
      node.widgets.push(rich);
    }
    rich.triggerDraw?.();
  }
  app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "ComfyDL.ShowText",
  beforeRegisterNodeDef(nodeType) {
    if (nodeType.comfyClass !== "CdlDatasetPreview") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      // Create the serialized state slot BEFORE workflow values are mapped.
      // Otherwise the legacy canvas_text value can be assigned to head.
      if (this.__cdlRichPreview?.widget) populate(this, "");
      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      if (message && message.text) populate(this, message.text);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      onConfigure?.apply(this, arguments);
      // Use the raw snapshot: before the hidden state widget is recreated,
      // LiteGraph can otherwise misassign the legacy text slot to head.
      const v = info?.widgets_values ?? this.widgets_values;
      const head = this.widgets?.find((w) => w.name === "head");
      if (head && v && v.length > 0) head.value = v[0];
      // widgets_values[0] is the head INT; [1:] is the saved display text.
      if (v && v.length > 1) populate(this, v.slice(1));
    };
  },

  loadedGraphNode(node) {
    if (node.comfyClass !== "CdlDatasetPreview") return;
    setTimeout(() => {
      // Nodes 2.0 may discard dynamically added widgets during configure.
      // RichWidget restores cdl_preview_text first; recreate the hidden
      // serialized slot afterwards so the next save keeps [head, text].
      const saved = node.properties?.cdl_preview_text
        ?? node.widgets?.find((w) => w.cdlShowText)?.value;
      if (saved) populate(node, saved);
    }, 0);
  },
});
