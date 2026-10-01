import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { acceptEvent, balanceText, batchText, selectedParameters, taskProgress } from "./state.mjs";

let catalog;
const states = new WeakMap();
const llmStates = new WeakMap();
const NODE_IDS = {
  llm: "Sinyuk.LLMAPI",
  config: "Sinyuk.ImageAPI.Config",
  generate: "Sinyuk.ImageAPI.Generate",
  folder: "Sinyuk.ImageAPI.LoadFolder",
  batch: "Sinyuk.ImageAPI.BatchGenerate",
};
const PROVIDER_LABELS = { grsai: "GRSAI", runninghub: "RunningHub" };
const widget = (node, name) => node?.widgets?.find((item) => item.name === name);
const parameters = (node) => Object.fromEntries((node.widgets ?? [])
  .filter((item) => item.name.startsWith("model."))
  .map((item) => [item.name.slice(6), item.value]));

function registerLLMProgressListener() {
  api.addEventListener("llm.progress", ({ detail }) => {
    const node = app.graph?.getNodeById(detail.node_id);
    const state = llmStates.get(node);
    if (!state) return;
    state.status.value = detail.stage ?? "Working";
    state.status.options.tooltip = `${detail.stage ?? "Working"} (${Math.round(detail.progress ?? 0)}%)`;
    node.setDirtyCanvas(true, true);
  });
}

function showStatus(node, text, level = "error", tooltip = text) {
  const state = states.get(node);
  if (!state) return;
  state.status.value = text;
  state.status.options.tooltip = tooltip;
  state.status.options.level = level;
  node.setDirtyCanvas(true, true);
}

function resetRemoteState(node, provider = providerId(node)) {
  const state = states.get(node);
  if (!state) return;
  node.properties.image_api_ui_token = crypto.randomUUID?.()
    ?? Array.from(crypto.getRandomValues(new Uint8Array(16)), (value) => value.toString(16).padStart(2, "0")).join("");
  state.modelRequest += 1;
  state.modelAbort?.abort();
  state.modelAbort = null;
  state.sequence = 0;
  state.usageShown = false;
  state.balance.value = provider === "runninghub" ? balanceText({ state: "unsupported" }) : balanceText({});
  state.status.value = "Idle";
  state.status.options.tooltip = "";
  if (state.directory) state.directory.value = "";
  node.setDirtyCanvas(true, true);
}

function connectedConfigNode(node) {
  const input = node.inputs?.find((item) => item.name === "api_config");
  const link = input?.link == null ? null : app.graph?.links?.[input.link];
  return link ? app.graph?.getNodeById(link.origin_id) : null;
}

function baseUrl(node) {
  return String(widget(connectedConfigNode(node), "base_url")?.value ?? "").trim() || activeCatalog(node).base_url;
}

function providerId(node) {
  const value = widget(connectedConfigNode(node), "provider")?.value;
  return value === "runninghub" ? "runninghub" : "grsai";
}

function activeCatalog(node, provider = providerId(node)) {
  return catalog.providers[provider] ?? catalog.providers.grsai;
}

function syncMaskInput(node, profile) {
  const input = node.inputs?.find((item) => item.name === "mask");
  if (!input) return;
  const connected = input.link != null;
  const model = widget(node, "model")?.value;
  const supported = profile?.family === "gpt_image"
    || (providerId(node) === "runninghub" && String(model).startsWith("rh:gpt-image-2.5-"));
  input.hidden = !supported && !connected;
  input.label = supported ? "MASK" : "MASK (GPT Image 2.5 only)";
  input.tooltip = supported
    ? "Optional mask; applies only to the first input image."
    : "This mask is only supported by GPT Image 2.5 Sunburst and Flare.";
}

function usageText(usage) {
  if (!usage || typeof usage !== "object") return "";
  const parts = [];
  if (usage.consumeCoins) parts.push(`RH coins: ${usage.consumeCoins}`);
  if (usage.consumeMoney) parts.push(`Runtime cost: ${usage.consumeMoney}`);
  if (usage.thirdPartyConsumeMoney) parts.push(`API cost: ${usage.thirdPartyConsumeMoney}`);
  if (usage.taskCostTime) parts.push(`Time: ${usage.taskCostTime}`);
  return parts.join(" · ");
}

function syncProvider(node, providerIdOverride = providerId(node)) {
  const provider = activeCatalog(node, providerIdOverride);
  const model = widget(node, "model");
  if (!model) return;
  const ids = Object.keys(provider.models);
  model.options.values = ids;
  model.options.getOptionLabel = (value) => provider.models[value]?.label ?? value;
  if (!ids.includes(model.value) && !states.get(node)?.configuring) model.value = provider.default_model;
  resetRemoteState(node, providerIdOverride);
  decorateParameters(node, providerIdOverride);
}

function resetConfigConsumers(configNode) {
  for (const linkId of configNode.outputs?.[0]?.links ?? []) {
    const link = app.graph?.links?.[linkId];
    const target = link ? app.graph?.getNodeById(link.target_id) : null;
    if (states.has(target)) syncProvider(target);
  }
}

function modelWarning(node, message) {
  const detail = message || "The selected model is currently unavailable.";
  showStatus(node, `Model unavailable: ${detail}`, "warning", detail);
  const toast = app.extensionManager?.toast;
  if (toast?.add) {
    toast.add({ severity: "warn", summary: "Model unavailable", detail, life: 6000 });
  }
}

async function checkModel(node, model) {
  if (providerId(node) !== "grsai") return;
  const state = states.get(node);
  const request = ++state.modelRequest;
  state.modelAbort?.abort();
  state.modelAbort = new AbortController();
  try {
    const response = await api.fetchApi("/image-api/model-status", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, base_url: baseUrl(node) }),
      signal: state.modelAbort.signal,
    });
    if (!response.ok || request !== state.modelRequest) return;
    const result = await response.json();
    if (result.ok && result.available === false) modelWarning(node, result.error);
  } catch (error) {
    if (error?.name !== "AbortError") console.debug("Model status check unavailable");
  }
}

function decorateParameters(node, provider = providerId(node)) {
  const state = states.get(node);
  const profile = activeCatalog(node, provider).models[widget(node, "model")?.value];
  if (!profile) {
    showStatus(node, "Configuration error: model removed or disabled");
    return;
  }
  const modelWidget = widget(node, "model");
  modelWidget.options.tooltip = profile.description || "";
  syncMaskInput(node, profile);
  for (const [name, rule] of Object.entries(profile.parameters)) {
    const item = widget(node, `model.${name}`);
    if (!item) continue;
    item.disabled = rule.options.length === 1;
    item.label = name === "aspectRatio" ? (profile.family === "gpt_image" ? "Image Size" : "Aspect Ratio")
      : ["imageSize", "resolution"].includes(name) ? "Resolution" : "Quality";
    const labels = Object.fromEntries(rule.options.map((option) => [option.value, option.label]));
    // Keep the options object stable because mounted ComfyUI widgets may retain its reference.
    item.options.disabled = item.disabled;
    item.options.getOptionLabel = (value) => labels[value] ?? value;
    // The canvas host hides disabled widget values by default. Keep fixed choices visible.
    Object.defineProperty(item, "_displayValue", {
      configurable: true,
      get() { return labels[this.value] ?? String(this.value); },
    });
    if (!item.imageApiWrapped) {
      const callback = item.callback;
      item.callback = function (...args) {
        const result = callback?.apply(this, args);
        state.previous = parameters(node);
        validateSelection(node, profile);
        return result;
      };
      item.imageApiWrapped = true;
    }
  }
  state.previous = parameters(node);
  validateSelection(node, profile);
  node.setDirtyCanvas(true, true);
}

function validateSelection(node, profile) {
  const invalid = Object.entries(profile.parameters).some(([name, rule]) =>
    !rule.options.some((option) => option.value === widget(node, `model.${name}`)?.value));
  const status = states.get(node).status;
  if (invalid) showStatus(node, "Configuration error: parameter removed or invalid");
  else if (status.value.startsWith("Configuration error")) {
    status.value = "Idle";
    status.options.tooltip = "";
  }
}

function configureProviderNode(node) {
  node.properties ??= {};
  const provider = widget(node, "provider");
  const baseUrlWidget = widget(node, "base_url");
  const token = widget(node, "token");
  if (!provider || !baseUrlWidget || !token) return;

  provider.options.getOptionLabel = (value) => PROVIDER_LABELS[value] ?? value;
  node.properties.image_api_base_urls ??= {};
  let current = provider.value === "runninghub" ? "runninghub" : "grsai";
  if (!(current in node.properties.image_api_base_urls)) {
    node.properties.image_api_base_urls[current] = String(baseUrlWidget.value ?? "");
  }

  const refresh = () => {
    const runningHub = current === "runninghub";
    token.disabled = runningHub;
    token.options.disabled = runningHub;
    token.label = runningHub ? "Token (GRSAI only)" : "Token";
    token.options.tooltip = runningHub
      ? "RunningHub does not use the account Token. The saved GRSAI value is preserved."
      : "Optional GRSAI account token used only for balance checks.";
    baseUrlWidget.options.tooltip = `Optional ${PROVIDER_LABELS[current]} API host. Leave empty to use the Provider default.`;
    node.setDirtyCanvas(true, true);
  };

  const providerCallback = provider.callback;
  provider.callback = function (...args) {
    node.properties.image_api_base_urls[current] = String(baseUrlWidget.value ?? "");
    const result = providerCallback?.apply(this, args);
    const next = this.value === "runninghub" ? "runninghub" : "grsai";
    if (next !== current) {
      current = next;
      baseUrlWidget.value = node.properties.image_api_base_urls[current] ?? "";
    }
    refresh();
    resetConfigConsumers(node);
    return result;
  };

  const baseUrlCallback = baseUrlWidget.callback;
  baseUrlWidget.callback = function (...args) {
    const result = baseUrlCallback?.apply(this, args);
    node.properties.image_api_base_urls[current] = String(this.value ?? "");
    resetConfigConsumers(node);
    return result;
  };

  for (const name of ["api_key", "token"]) {
    const item = widget(node, name);
    if (!item) continue;
    const callback = item.callback;
    item.callback = function (...args) {
      const result = callback?.apply(this, args);
      resetConfigConsumers(node);
      return result;
    };
  }
  const configure = node.configure;
  node.configure = function (data) {
    const result = configure.call(this, data);
    current = provider.value === "runninghub" ? "runninghub" : "grsai";
    node.properties.image_api_base_urls ??= {};
    node.properties.image_api_base_urls[current] = String(baseUrlWidget.value ?? "");
    refresh();
    return result;
  };
  const serialize = node.onSerialize;
  node.onSerialize = function (data) {
    node.properties.image_api_base_urls[current] = String(baseUrlWidget.value ?? "");
    serialize?.call(this, data);
  };
  refresh();
}

app.registerExtension({
  name: "image-api.providers",
  async setup() {
    registerLLMProgressListener();
    const response = await api.fetchApi("/image-api/catalog");
    if (!response.ok) throw new Error("Image API configuration unavailable");
    catalog = await response.json();
    for (const event of ["image-api.balance", "image-api.progress", "image-api.batch"]) {
      api.addEventListener(event, ({ detail }) => {
        // Tokens also isolate separate workflows that reuse numeric node IDs.
        const node = app.graph?.getNodeById(detail.node_id);
        const state = states.get(node);
        if (!state) return;
        if (!acceptEvent(state, detail, node.properties.image_api_ui_token)) return;
        if (event === "image-api.balance") {
          state.balance.value = balanceText(detail);
          state.balance.options.tooltip = detail.message ?? "";
        } else if (event === "image-api.batch") {
          state.status.value = batchText(detail);
          state.status.options.tooltip = detail.directory ?? "";
          if (state.directory) state.directory.value = detail.directory ?? "";
        } else {
          const view = taskProgress(detail);
          state.status.value = view.text;
          state.status.options.tooltip = view.tooltip ?? view.text ?? "";
          const usage = usageText(detail.usage);
          if (event === "image-api.progress" && detail.stage === "succeeded"
              && usage && node.comfyClass === NODE_IDS.generate && !state.usageShown) {
            state.usageShown = true;
            app.extensionManager?.toast?.add?.({
              severity: "success",
              summary: "RunningHub task completed",
              detail: usage,
              life: 9000,
            });
            state.status.value = `Completed · ${usage}`;
            state.status.options.tooltip = usage;
          }
        }
        node.setDirtyCanvas(true, true);
      });
    }
  },
  nodeCreated(node) {
    if (node.comfyClass === NODE_IDS.folder) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Files: Not loaded";
      const list = document.createElement("pre");
      list.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0;max-height:200px;overflow:auto";
      details.style.cssText = "padding:6px;color:var(--input-text);font-size:12px";
      details.title = "Files loaded in natural filename order from the ComfyUI server folder.";
      details.append(summary, list);
      const filesWidget = node.addDOMWidget("image_api_files", "details", details, { serialize: false });
      filesWidget.serialize = false;
      const executed = node.onExecuted;
      node.onExecuted = function (message) {
        executed?.call(this, message);
        const files = message.image_api_files ?? [];
        summary.textContent = `Files: ${files.length}`;
        list.textContent = files.map((name, i) => `${i + 1}. ${name}`).join("\n");
        node.setDirtyCanvas(true, true);
      };
      return;
    }
    if (node.comfyClass === NODE_IDS.llm) {
      const status = node.addWidget("text", "Status", "Idle", () => {}, { serialize: false });
      status.serialize = false;
      status.disabled = false;
      status.options.readOnly = true;
      const state = { status };
      llmStates.set(node, state);
      return;
    }
    if (node.comfyClass === NODE_IDS.config) {
      configureProviderNode(node);
      return;
    }
    if (![NODE_IDS.generate, NODE_IDS.batch].includes(node.comfyClass) || !catalog) return;
    node.properties ??= {};
    const balance = node.addWidget("text", "balance", "", () => {}, { serialize: false });
    const status = node.addWidget("text", "Status", "Idle", () => {}, { serialize: false });
    const directory = node.comfyClass === NODE_IDS.batch
      ? node.addWidget("text", "output_directory", "", () => {}, { serialize: false }) : null;
    for (const item of [balance, directory].filter(Boolean)) {
      item.serialize = false;
      item.disabled = true;
      item.options.readOnly = true;
      Object.defineProperty(item, "displayName", { configurable: true, get: () => "" });
      Object.defineProperty(item, "_displayValue", { configurable: true, get() { return String(this.value); } });
    }
    balance.options.tooltip = "Latest account balance state. RunningHub does not provide this check.";
    if (directory) directory.options.tooltip = "Batch output folder on the ComfyUI server.";
    status.serialize = false;
    status.disabled = false;
    status.options.readOnly = true;
    const state = { balance, status, directory, sequence: 0, previous: {}, configuring: false,
      modelRequest: 0, modelAbort: null, usageShown: false };
    states.set(node, state);
    const model = widget(node, "model");
    model.value = activeCatalog(node).default_model;
    model.options.values = Object.keys(activeCatalog(node).models);
    model.options.getOptionLabel = (value) => activeCatalog(node).models[value]?.label ?? value;
    decorateParameters(node);
    resetRemoteState(node);

    if (directory) {
      const labelReferences = () => {
        for (const input of node.inputs ?? []) {
          const match = /^references\.reference_(\d+)$/.exec(input.name);
          if (match) {
            input.label = `Reference ${match[1]}`;
            input.tooltip = "One ordered reference source. Each task receives one image from every connected source.";
          }
        }
      };
      labelReferences();
      const connections = node.onConnectionsChange;
      node.onConnectionsChange = function (type, slot, connected, ...rest) {
        // Native Autogrow compacts disconnected middle inputs. Preserve image numbering instead.
        if (type === 1 && !connected && this.inputs?.[slot]?.name.startsWith("references.")) {
          this.setDirtyCanvas(true, true);
          return;
        }
        const result = connections?.call(this, type, slot, connected, ...rest);
        if (this.inputs?.[slot]?.name === "api_config") syncProvider(this);
        labelReferences();
        return result;
      };
    } else {
      const connections = node.onConnectionsChange;
      node.onConnectionsChange = function (type, slot, connected, ...rest) {
        const result = connections?.call(this, type, slot, connected, ...rest);
        if (this.inputs?.[slot]?.name === "api_config") syncProvider(this);
        if (this.inputs?.[slot]?.name === "mask") {
          const profile = activeCatalog(this).models[widget(this, "model")?.value];
          syncMaskInput(this, profile);
        }
        return result;
      };
    }

    const callback = model.callback;
    model.callback = function (...args) {
      const previous = state.previous;
      const result = callback?.apply(this, args);
      const profile = activeCatalog(node).models[model.value];
      if (profile && !state.configuring) {
        for (const [name, value] of Object.entries(selectedParameters(profile, previous))) {
          const item = widget(node, `model.${name}`);
          if (item) item.value = value;
        }
        state.status.value = "Idle";
        state.status.options.tooltip = "";
        void checkModel(node, model.value);
      }
      decorateParameters(node);
      return result;
    };
    const configure = node.configure;
    node.configure = function (data) {
      state.configuring = true;
      try {
        const result = configure.call(this, data);
        const saved = data.properties?.image_api_selection;
        if (saved) {
          // Restore literal saved values, including invalid ones. Never silently migrate a workflow.
          model.value = saved.model;
          for (const [name, value] of Object.entries(saved.parameters ?? {})) {
            const item = widget(node, `model.${name}`);
            if (item) item.value = value;
          }
        }
        const savedProvider = saved?.provider
          ?? (saved?.model?.startsWith("rh:") ? "runninghub" : providerId(node));
        syncProvider(node, savedProvider);
        return result;
      } finally {
        state.configuring = false;
      }
    };
    const serialize = node.onSerialize;
    node.onSerialize = function (data) {
      serialize?.call(this, data);
      data.properties ??= {};
      data.properties.image_api_ui_token = node.properties.image_api_ui_token;
      data.properties.image_api_selection = {
        provider: providerId(node), model: model.value, parameters: parameters(node),
      };
    };
  },
});
