// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! External ("on-ramp") mode configuration.
//!
//! The EPP runs in one of two modes, selected at startup by `DYN_EPP_EXTERNAL`:
//!
//! * **Dynamo mode** (default): the EPP attaches to a Dynamo deployment —
//!   `DistributedRuntime` over etcd/NATS, Dynamo workers, model-card-derived
//!   preprocessor, and the KV router's own event-plane subscriber. See
//!   [`crate::epp::Router::from_discovery`].
//!
//! * **External mode** (`DYN_EPP_EXTERNAL=true`): the EPP fronts a fleet of
//!   *raw* `vllm serve` pods with **no Dynamo control plane**. Discovery is a
//!   plain Kubernetes label selector, KV-cache state comes from each pod's
//!   native vLLM ZMQ event stream, and the embedded router runs on the inert
//!   `mem` discovery backend (no etcd/NATS). See
//!   [`crate::epp::Router::from_external`].
//!
//! This module defines the configuration surface for external mode, parses it
//! from the environment, and (for now) provides a stub `Router::from_external`
//! constructor. The runtime wiring lands in later changes.

use anyhow::{Context, Result};

/// Default vLLM OpenAI HTTP port (the InferencePool `targetPort`).
const DEFAULT_TARGET_PORT: u16 = 8000;
/// Default vLLM `--kv-events-config` ZMQ PUB port.
const DEFAULT_KV_EVENT_PORT: u16 = 5557;

/// Configuration for external (raw-vLLM, runtime-free) mode.
///
/// All values are sourced from the environment by [`ExternalConfig::from_env`].
/// The corresponding deployment manifests live in
/// `deploy/inference-gateway/ext-proc/examples/onramp/`.
#[derive(Debug, Clone)]
pub struct ExternalConfig {
    /// Kubernetes label selector identifying the raw vLLM worker pods, e.g.
    /// `app=vllm-qwen`. Sourced from `DYN_EPP_POD_SELECTOR`.
    pub pod_selector: String,

    /// vLLM OpenAI HTTP port that the gateway forwards inference traffic to
    /// (the InferencePool `targetPort`). Sourced from `DYN_EPP_TARGET_PORT`.
    pub target_port: u16,

    /// Model id used to build the tokenizer/preprocessor offline (external
    /// mode has no Dynamo model card). Sourced from `DYN_MODEL_NAME`.
    pub model_name: String,

    /// KV-cache block size. MUST match vLLM's `--block-size`. Sourced from
    /// `DYN_KV_CACHE_BLOCK_SIZE`.
    pub block_size: u32,

    /// Whether to consume per-pod native vLLM ZMQ KV-cache events for precise
    /// prefix routing. Sourced from `DYN_EPP_KV_EVENTS` (default `true`).
    pub kv_events: bool,

    /// Port of each vLLM pod's `--kv-events-config` ZMQ PUB socket. Sourced
    /// from `DYN_EPP_KV_EVENT_PORT`.
    pub kv_event_port: u16,

    /// ZMQ topic configured in vLLM's `--kv-events-config` (`""` = no topic,
    /// the vLLM default). Sourced from `DYN_EPP_KV_EVENT_TOPIC`.
    pub kv_event_topic: String,

    /// Pod label key that distinguishes prefill vs decode pods in
    /// disaggregated deployments (e.g. `dynamo-role`). `None` ⇒ aggregated.
    /// Sourced from `DYN_EPP_ROLE_LABEL`.
    pub role_label: Option<String>,

    /// Fail requests rather than falling back to aggregated serving when
    /// prefill routing is unavailable. Sourced from `DYN_ENFORCE_DISAGG`.
    pub enforce_disagg: bool,

    /// Emit the selected prefill pod's `ip:port` as `x-prefiller-host-port`
    /// for a decode-side P/D routing sidecar. Sourced from
    /// `DYN_EPP_EMIT_PREFILLER_HOST_PORT`.
    pub emit_prefiller_host_port: bool,
}

impl ExternalConfig {
    /// Returns `true` when `DYN_EPP_EXTERNAL` is set to a truthy value.
    pub fn is_enabled() -> bool {
        env_bool("DYN_EPP_EXTERNAL", false)
    }

    /// Parse the external-mode configuration from the environment.
    ///
    /// Errors if a required value (`DYN_EPP_POD_SELECTOR`, `DYN_MODEL_NAME`,
    /// `DYN_KV_CACHE_BLOCK_SIZE`) is missing or malformed, so a
    /// misconfiguration fails fast at startup instead of silently 503-ing.
    pub fn from_env() -> Result<Self> {
        let pod_selector = require_env("DYN_EPP_POD_SELECTOR")?;
        let model_name = require_env("DYN_MODEL_NAME")?;
        let block_size = require_env("DYN_KV_CACHE_BLOCK_SIZE")?
            .parse::<u32>()
            .context("DYN_KV_CACHE_BLOCK_SIZE must be a positive integer")?;
        if block_size == 0 {
            anyhow::bail!("DYN_KV_CACHE_BLOCK_SIZE must be greater than 0");
        }

        let role_label = env_opt("DYN_EPP_ROLE_LABEL");

        Ok(Self {
            pod_selector,
            target_port: env_parse("DYN_EPP_TARGET_PORT", DEFAULT_TARGET_PORT)?,
            model_name,
            block_size,
            kv_events: env_bool("DYN_EPP_KV_EVENTS", true),
            kv_event_port: env_parse("DYN_EPP_KV_EVENT_PORT", DEFAULT_KV_EVENT_PORT)?,
            kv_event_topic: std::env::var("DYN_EPP_KV_EVENT_TOPIC").unwrap_or_default(),
            role_label,
            enforce_disagg: env_bool("DYN_ENFORCE_DISAGG", false),
            emit_prefiller_host_port: env_bool("DYN_EPP_EMIT_PREFILLER_HOST_PORT", false),
        })
    }

    /// Whether this configuration describes a disaggregated deployment
    /// (a prefill/decode role label is set).
    pub fn is_disaggregated(&self) -> bool {
        self.role_label.is_some()
    }
}

impl crate::epp::Router {
    /// Initialize the router in external ("on-ramp") mode.
    ///
    /// External mode fronts a fleet of raw `vllm serve` pods with no Dynamo
    /// control plane: discovery is a Kubernetes label selector, KV-cache state
    /// comes from each pod's native vLLM ZMQ event stream (republished onto the
    /// embedded router's event plane), and the runtime uses the inert `mem`
    /// discovery backend (no etcd/NATS). See [`ExternalConfig`] and
    /// `deploy/inference-gateway/ext-proc/examples/onramp/`.
    ///
    /// NOTE: this is currently a stub. The construction path (offline
    /// preprocessor, mem-backed component, label-selector reflector, and the
    /// ZMQ KV republisher) lands in subsequent changes. It is defined here so
    /// the `DYN_EPP_EXTERNAL` mode switch and configuration surface can be
    /// reviewed independently — and so this change touches no existing
    /// `epp.rs` (Dynamo-mode) code.
    pub async fn from_external(cfg: ExternalConfig) -> Result<Self> {
        anyhow::bail!(
            "external mode (DYN_EPP_EXTERNAL=true) is not yet implemented: \
             parsed config for model '{}' (block_size={}, selector='{}', disagg={}). \
             Use the default Dynamo mode for now.",
            cfg.model_name,
            cfg.block_size,
            cfg.pod_selector,
            cfg.is_disaggregated(),
        )
    }
}

fn require_env(key: &str) -> Result<String> {
    let value = std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    value.ok_or_else(|| {
        anyhow::anyhow!("external mode requires {key} to be set (DYN_EPP_EXTERNAL=true)")
    })
}

fn env_opt(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn env_parse<T>(key: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match env_opt(key) {
        Some(v) => v
            .parse::<T>()
            .map_err(|e| anyhow::anyhow!("{key} is invalid: {e}")),
        None => Ok(default),
    }
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .and_then(|v| match v.trim().to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Some(true),
            "false" | "0" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env access is process-global; serialize the tests that mutate it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear() {
        for k in [
            "DYN_EPP_EXTERNAL",
            "DYN_EPP_POD_SELECTOR",
            "DYN_EPP_TARGET_PORT",
            "DYN_MODEL_NAME",
            "DYN_KV_CACHE_BLOCK_SIZE",
            "DYN_EPP_KV_EVENTS",
            "DYN_EPP_KV_EVENT_PORT",
            "DYN_EPP_KV_EVENT_TOPIC",
            "DYN_EPP_ROLE_LABEL",
            "DYN_ENFORCE_DISAGG",
            "DYN_EPP_EMIT_PREFILLER_HOST_PORT",
        ] {
            unsafe { std::env::remove_var(k) };
        }
    }

    #[test]
    fn parses_minimal_agg_config_with_defaults() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear();
        unsafe {
            std::env::set_var("DYN_EPP_POD_SELECTOR", "app=vllm-qwen");
            std::env::set_var("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B");
            std::env::set_var("DYN_KV_CACHE_BLOCK_SIZE", "16");
        }

        let cfg = ExternalConfig::from_env().unwrap();
        assert_eq!(cfg.pod_selector, "app=vllm-qwen");
        assert_eq!(cfg.target_port, DEFAULT_TARGET_PORT);
        assert_eq!(cfg.kv_event_port, DEFAULT_KV_EVENT_PORT);
        assert!(cfg.kv_events);
        assert!(!cfg.is_disaggregated());
        clear();
    }

    #[test]
    fn missing_required_value_errors() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear();
        unsafe {
            std::env::set_var("DYN_EPP_POD_SELECTOR", "app=vllm-qwen");
        }
        // DYN_MODEL_NAME / DYN_KV_CACHE_BLOCK_SIZE missing.
        assert!(ExternalConfig::from_env().is_err());
        clear();
    }

    #[test]
    fn role_label_marks_disaggregated() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear();
        unsafe {
            std::env::set_var("DYN_EPP_POD_SELECTOR", "app=vllm-qwen");
            std::env::set_var("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B");
            std::env::set_var("DYN_KV_CACHE_BLOCK_SIZE", "16");
            std::env::set_var("DYN_EPP_ROLE_LABEL", "dynamo-role");
        }
        let cfg = ExternalConfig::from_env().unwrap();
        assert!(cfg.is_disaggregated());
        assert_eq!(cfg.role_label.as_deref(), Some("dynamo-role"));
        clear();
    }
}
