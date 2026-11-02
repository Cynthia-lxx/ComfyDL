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
  // text arrives as the ui payload's list (one entry: the whole clipped
  // table) or as a saved string — normalize to one string.
  const value = Array.isArray(text) ? text.join("\n") : String(text ?? "");
  removeDisplayWidgets(node);

  const w = ComfyWidgets.STRING(
    node, "cdl_show_text", ["STRING", { multiline: true }], app
  ).widget;
  w.cdlShowText = true;
  w.inputEl.readOnly = true;
  w.inputEl.style.opacity = "0.6";
  w.inputEl.style.fontFamily = "monospace";
  const lines = Math.min(
    value.split("\n").length + 1, MAX_WIDGET_LINES
  );
  w.inputEl.style.height = Math.min(28 + lines * 14, 760) + "px";
  w.value = value;
  app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "ComfyDL.ShowText",
  beforeRegisterNodeDef(nodeType) {
    if (nodeType.comfyClass !== "CdlDatasetPreview") return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      if (message && message.text) populate(this, message.text);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      // widgets_values[0] is the head INT; [1:] is the saved display text.
      const v = this.widgets_values;
      if (v && v.length > 1) populate(this, v.slice(1));
    };
  },
});
