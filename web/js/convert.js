/**
 * ComfyDL Convert (convert.js) - dynamic output socket for CdlConvert.
 *
 * Contract with the backend (comfydl/nodes/convert.py):
 *   - INPUT_TYPES: required combo "target_type" ("match link" + concrete
 *     types, keep in sync with TARGET_TYPES there) + optional wildcard
 *     input "value";
 *   - RETURN_TYPES: ("*",) - the output socket's declared type is a pure
 *     frontend link guard, so this module rewrites outputs[0].type and the
 *     backend never sees it.
 *
 * Behaviour (Reroute / dataops-Unpack pattern):
 *   - combo change -> outputs[0].type = the chosen type;
 *   - "match link" + the value input gets wired -> the combo auto-fills from
 *     the origin socket's type and the output socket follows (manual combo
 *     choice wins: an explicit user pick is never overwritten);
 *   - the value input disconnects -> if the combo was auto-filled it reverts
 *     to "match link" (an explicit pick survives disconnects);
 *   - onConfigure (workflow load) -> output type re-derived from the saved
 *     combo value.
 *
 * Containment: only the CdlConvert prototype is patched. Reduced-motion: no
 * animation of any kind.
 *
 * In: extension module (served from /extensions/comfydl/js/convert.js).
 * Out: registered node behaviour patch.
 */
import { app } from "/scripts/app.js";

const CONVERT_CLASS = "CdlConvert";
const DEFAULT_TARGET = "match link";
// Mirror of TARGET_TYPES in comfydl/nodes/convert.py.
const TARGET_TYPES = [
  "STRING", "INT", "FLOAT", "BOOLEAN",
  "TENSOR", "IMAGE", "MASK", "LATENT",
];

function configuring() {
  // During graph load LiteGraph restores the slots itself; reacting to the
  // connection events fired along the way would race the deserializer.
  return !!app.configuringGraph;
}

function findCombo(node) {
  return node.widgets?.find((w) => w.name === "target_type") || null;
}

/** Set the output socket's declared type (frontend link guard only). */
function syncOutput(node, target) {
  const out = node.outputs && node.outputs[0];
  if (!out) return;
  out.type = target === DEFAULT_TARGET ? "*" : target;
  app.graph.setDirtyCanvas(true, true);
}

function applyTarget(node, target, { fromLink }) {
  const combo = findCombo(node);
  if (!combo) return;
  node.__cdlConvertAuto = fromLink && target !== DEFAULT_TARGET;
  combo.value = target;
  syncOutput(node, target);
}

app.registerExtension({
  name: "ComfyDL.Convert",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== CONVERT_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const combo = findCombo(this);
      if (combo) {
        combo.callback = (value) => {
          // Manual pick: clear the auto-fill mark so a later disconnect
          // does not revert the user's explicit choice.
          this.__cdlConvertAuto = false;
          syncOutput(this, value);
        };
      }
      syncOutput(this, combo ? combo.value : DEFAULT_TARGET);
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (
      slotType, slot, event, link_info
    ) {
      const r = onConnectionsChange
        ? onConnectionsChange.apply(this, arguments)
        : undefined;
      if (configuring() || !link_info) return r;
      const isInput = slotType === 1 || slotType === "input";
      if (!isInput || this.inputs?.[slot]?.name !== "value") return r;

      const connected = event === true || event === 1 || event === "connect";
      if (!connected) {
        // Disconnect: revert an auto-filled combo, keep a manual pick.
        if (this.__cdlConvertAuto) {
          this.__cdlConvertAuto = false;
          applyTarget(this, DEFAULT_TARGET, { fromLink: false });
        }
        return r;
      }

      const combo = findCombo(this);
      if (!combo || combo.value !== DEFAULT_TARGET) return r; // manual pick wins
      const originNode = app.graph._nodes_by_id?.[link_info.origin_id];
      const originType = originNode?.outputs?.[link_info.origin_slot]?.type;
      if (originType && TARGET_TYPES.includes(originType)) {
        applyTarget(this, originType, { fromLink: true });
      }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure?.apply(this, arguments);
      // widgets_values[0] is the saved target_type combo; re-derive the
      // output socket's declared type from it (the serialized workflow
      // stores "*", which is not a legal combo value here).
      const saved = info?.widgets_values?.[0];
      const target = TARGET_TYPES.includes(saved) ? saved : DEFAULT_TARGET;
      this.__cdlConvertAuto = false;
      applyTarget(this, target, { fromLink: false });
      return r;
    };
  },
});
