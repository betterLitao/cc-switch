//! 供应商连通性检查命令
//!
//! 注意：本检查只探测 base_url 是否可达，不发真实大模型请求，也不触碰故障转移
//! 熔断器（熔断器由真实转发流量驱动）。详见 `services::stream_check`。

use crate::app_config::AppType;
use crate::commands::{CodexOAuthState, CopilotAuthState, XaiOAuthState};
use crate::error::AppError;
use crate::proxy::providers::{AuthInfo, AuthStrategy};
use crate::services::model_check::{ModelCheckResult, ModelCheckService};
use crate::services::stream_check::{
    HealthStatus, StreamCheckConfig, StreamCheckResult, StreamCheckService,
};
use crate::store::AppState;
use std::collections::HashSet;
use tauri::State;

/// 连通性检查（单个供应商）
#[tauri::command]
pub async fn stream_check_provider(
    state: State<'_, AppState>,
    copilot_state: State<'_, CopilotAuthState>,
    app_type: AppType,
    provider_id: String,
) -> Result<StreamCheckResult, AppError> {
    let config = state.db.get_stream_check_config()?;

    let providers = state.db.get_all_providers(app_type.as_str())?;
    let provider = providers
        .get(&provider_id)
        .ok_or_else(|| AppError::Message(format!("供应商 {provider_id} 不存在")))?;

    // Copilot 端点是动态的（随 OAuth token 解析），需预先取出 host 再探测；
    // 其余供应商传 None，由服务层从 settings_config 提取 base_url。无需鉴权。
    let base_url_override = resolve_copilot_base_url_override(provider, &copilot_state).await?;
    let result =
        StreamCheckService::check_with_retry(&app_type, provider, &config, base_url_override)
            .await?;

    // 记录日志
    let _ =
        state
            .db
            .save_stream_check_log(&provider_id, &provider.name, app_type.as_str(), &result);

    Ok(result)
}

/// 模型可用性检查（单个供应商）
///
/// 与 `stream_check_provider` 完全分开：这里会发送一次最小真实流式请求，
/// 用于验证当前保存的模型、认证、协议转换和上游响应，而不是只验证 URL 可达。
#[tauri::command]
pub async fn model_check_provider(
    state: State<'_, AppState>,
    copilot_state: State<'_, CopilotAuthState>,
    codex_state: State<'_, CodexOAuthState>,
    xai_state: State<'_, XaiOAuthState>,
    app_type: AppType,
    provider_id: String,
    requested_model: Option<String>,
) -> Result<ModelCheckResult, AppError> {
    let providers = state.db.get_all_providers(app_type.as_str())?;
    let provider = providers
        .get(&provider_id)
        .ok_or_else(|| AppError::Message(format!("供应商 {provider_id} 不存在")))?;

    let (auth_override, base_url_override, managed_account_id) = resolve_model_check_overrides(
        &app_type,
        provider,
        &copilot_state,
        &codex_state,
        &xai_state,
    )
    .await?;

    ModelCheckService::check(
        &app_type,
        provider,
        requested_model,
        auth_override,
        base_url_override,
        None,
        managed_account_id,
    )
    .await
}

/// 批量连通性检查
#[tauri::command]
pub async fn stream_check_all_providers(
    state: State<'_, AppState>,
    copilot_state: State<'_, CopilotAuthState>,
    app_type: AppType,
    proxy_targets_only: bool,
) -> Result<Vec<(String, StreamCheckResult)>, AppError> {
    let config = state.db.get_stream_check_config()?;
    let providers = state.db.get_all_providers(app_type.as_str())?;

    let allowed_ids: Option<HashSet<String>> = if proxy_targets_only {
        let mut ids = HashSet::new();
        if let Ok(Some(current_id)) = state.db.get_current_provider(app_type.as_str()) {
            ids.insert(current_id);
        }
        if let Ok(queue) = state.db.get_failover_queue(app_type.as_str()) {
            for item in queue {
                ids.insert(item.provider_id);
            }
        }
        Some(ids)
    } else {
        None
    };

    let mut results = Vec::new();
    for (id, provider) in providers {
        // Official OAuth providers intentionally have no user-configured probe
        // target. Never turn their runtime adapter defaults into unauthenticated
        // network probes against first-party endpoints.
        if provider.category.as_deref() == Some("official") {
            continue;
        }
        if let Some(ids) = &allowed_ids {
            if !ids.contains(&id) {
                continue;
            }
        }

        let base_url_override =
            resolve_copilot_base_url_override(&provider, &copilot_state).await?;
        let result =
            StreamCheckService::check_with_retry(&app_type, &provider, &config, base_url_override)
                .await
                .unwrap_or_else(|e| StreamCheckResult {
                    status: HealthStatus::Failed,
                    success: false,
                    message: e.to_string(),
                    response_time_ms: None,
                    http_status: None,
                    model_used: String::new(),
                    tested_at: chrono::Utc::now().timestamp(),
                    retry_count: 0,
                    error_category: None,
                });

        let _ = state
            .db
            .save_stream_check_log(&id, &provider.name, app_type.as_str(), &result);

        results.push((id, result));
    }

    Ok(results)
}

/// 获取连通性检查配置
#[tauri::command]
pub fn get_stream_check_config(state: State<'_, AppState>) -> Result<StreamCheckConfig, AppError> {
    state.db.get_stream_check_config()
}

/// 保存连通性检查配置
#[tauri::command]
pub fn save_stream_check_config(
    state: State<'_, AppState>,
    config: StreamCheckConfig,
) -> Result<(), AppError> {
    state.db.save_stream_check_config(&config)
}

/// Copilot 供应商的 base_url 需要从 OAuth 管理器动态解析（按账号或默认端点）。
/// `is_full_url` 的供应商已是完整地址，无需解析。
async fn resolve_copilot_base_url_override(
    provider: &crate::provider::Provider,
    copilot_state: &State<'_, CopilotAuthState>,
) -> Result<Option<String>, AppError> {
    let is_copilot = is_copilot_provider(provider);
    let is_full_url = provider
        .meta
        .as_ref()
        .and_then(|meta| meta.is_full_url)
        .unwrap_or(false);

    if !is_copilot || is_full_url {
        return Ok(None);
    }

    let auth_manager = copilot_state.0.read().await;
    let account_id = provider
        .meta
        .as_ref()
        .and_then(|meta| meta.managed_account_id_for("github_copilot"));

    let endpoint = match account_id.as_deref() {
        Some(id) => auth_manager.get_api_endpoint(id).await,
        None => auth_manager.get_default_api_endpoint().await,
    };

    Ok(Some(endpoint))
}

/// 真实模型检查需要把托管 OAuth 认证解析成一次性请求凭据；普通 Provider
/// 继续由 `ModelCheckService` 从保存的配置中提取，避免命令层重复实现认证规则。
async fn resolve_model_check_overrides(
    app_type: &AppType,
    provider: &crate::provider::Provider,
    copilot_state: &State<'_, CopilotAuthState>,
    codex_state: &State<'_, CodexOAuthState>,
    xai_state: &State<'_, XaiOAuthState>,
) -> Result<(Option<AuthInfo>, Option<String>, Option<String>), AppError> {
    if matches!(app_type, AppType::Claude | AppType::ClaudeDesktop) && provider.is_github_copilot()
    {
        let auth_manager = copilot_state.0.read().await;
        let account_id = provider
            .meta
            .as_ref()
            .and_then(|meta| meta.managed_account_id_for("github_copilot"));
        let token = match account_id.as_deref() {
            Some(id) => auth_manager
                .get_valid_token_for_account(id)
                .await
                .map_err(|error| AppError::Message(error.to_string()))?,
            None => auth_manager
                .get_valid_token()
                .await
                .map_err(|error| AppError::Message(error.to_string()))?,
        };
        let endpoint = match account_id.as_deref() {
            Some(id) => auth_manager.get_api_endpoint(id).await,
            None => auth_manager.get_default_api_endpoint().await,
        };

        return Ok((
            Some(AuthInfo::new(token, AuthStrategy::GitHubCopilot)),
            Some(endpoint),
            None,
        ));
    }

    if matches!(
        app_type,
        AppType::Claude | AppType::ClaudeDesktop | AppType::Codex
    ) && provider.is_codex_oauth()
    {
        let auth_manager = codex_state.0.read().await;
        let account_id = match provider
            .meta
            .as_ref()
            .and_then(|meta| meta.managed_account_id_for("codex_oauth"))
        {
            Some(id) => Some(id),
            None => auth_manager.default_account_id().await,
        };
        let token = match account_id.as_deref() {
            Some(id) => auth_manager
                .get_valid_token_for_account(id)
                .await
                .map_err(|error| AppError::Message(error.to_string()))?,
            None => auth_manager
                .get_valid_token()
                .await
                .map_err(|error| AppError::Message(error.to_string()))?,
        };

        return Ok((
            Some(AuthInfo::new(token, AuthStrategy::CodexOAuth)),
            None,
            account_id,
        ));
    }

    if matches!(
        app_type,
        AppType::Claude | AppType::ClaudeDesktop | AppType::Codex | AppType::GrokBuild
    ) && provider.is_xai_oauth()
    {
        let auth_manager = xai_state.0.read().await;
        let account_id = match provider
            .meta
            .as_ref()
            .and_then(|meta| meta.managed_account_id_for("xai_oauth"))
        {
            Some(id) => Some(id),
            None => auth_manager.default_account_id().await,
        };
        let token = match account_id.as_deref() {
            Some(id) => auth_manager
                .get_valid_token_for_account(id)
                .await
                .map_err(|error| AppError::Message(error.to_string()))?,
            None => auth_manager
                .get_valid_token()
                .await
                .map_err(|error| AppError::Message(error.to_string()))?,
        };

        return Ok((
            Some(AuthInfo::new(token, AuthStrategy::XaiOAuth)),
            None,
            account_id,
        ));
    }

    Ok((None, None, None))
}

fn is_copilot_provider(provider: &crate::provider::Provider) -> bool {
    provider
        .meta
        .as_ref()
        .and_then(|meta| meta.provider_type.as_deref())
        == Some("github_copilot")
        || provider
            .settings_config
            .pointer("/env/ANTHROPIC_BASE_URL")
            .and_then(|value| value.as_str())
            .map(|url| url.contains("githubcopilot.com"))
            .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::is_copilot_provider;
    use crate::provider::{Provider, ProviderMeta};
    use serde_json::json;

    #[test]
    fn copilot_provider_detection_accepts_provider_type_or_base_url() {
        let typed_provider = Provider {
            id: "p1".to_string(),
            name: "typed".to_string(),
            settings_config: json!({}),
            website_url: None,
            category: None,
            created_at: None,
            sort_index: None,
            notes: None,
            meta: Some(ProviderMeta {
                provider_type: Some("github_copilot".to_string()),
                ..Default::default()
            }),
            icon: None,
            icon_color: None,
            in_failover_queue: false,
        };
        assert!(is_copilot_provider(&typed_provider));

        let url_provider = Provider {
            id: "p2".to_string(),
            name: "url".to_string(),
            settings_config: json!({
                "env": {
                    "ANTHROPIC_BASE_URL": "https://api.githubcopilot.com"
                }
            }),
            website_url: None,
            category: None,
            created_at: None,
            sort_index: None,
            notes: None,
            meta: None,
            icon: None,
            icon_color: None,
            in_failover_queue: false,
        };
        assert!(is_copilot_provider(&url_provider));
    }

    #[test]
    fn copilot_full_url_metadata_is_available_for_override_guard() {
        let provider = Provider {
            id: "p3".to_string(),
            name: "relay".to_string(),
            settings_config: json!({}),
            website_url: None,
            category: None,
            created_at: None,
            sort_index: None,
            notes: None,
            meta: Some(ProviderMeta {
                provider_type: Some("github_copilot".to_string()),
                is_full_url: Some(true),
                ..Default::default()
            }),
            icon: None,
            icon_color: None,
            in_failover_queue: false,
        };

        assert!(is_copilot_provider(&provider));
        assert_eq!(
            provider.meta.as_ref().and_then(|meta| meta.is_full_url),
            Some(true)
        );
    }
}
