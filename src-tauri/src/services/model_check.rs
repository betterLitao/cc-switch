//! 供应商模型可用性检查服务。
//!
//! `stream_check` 只回答“地址是否能收到 HTTP 响应”；本服务回答“当前 Provider
//! 保存的认证、协议和模型能否完成一次最小真实请求”。两条路径故意分开，避免
//! 网络探测的语义被重新改成有计费风险的模型请求。

use futures::StreamExt;
use reqwest::header::HeaderValue;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

use crate::app_config::AppType;
use crate::error::AppError;
use crate::provider::Provider;
use crate::proxy::gemini_url::{normalize_gemini_model_id, resolve_gemini_native_url};
use crate::proxy::model_mapper::{apply_model_mapping, strip_one_m_suffix_for_upstream_from_body};
use crate::proxy::providers::copilot_auth;
use crate::proxy::providers::transform::anthropic_to_openai;
use crate::proxy::providers::transform_gemini::anthropic_to_gemini;
use crate::proxy::providers::transform_responses::anthropic_to_responses;
use crate::proxy::providers::{
    codex_provider_uses_chat_completions, should_convert_codex_responses_to_anthropic, AuthInfo,
    AuthStrategy, ClaudeAdapter, CodexAdapter, GeminiAdapter, ProviderAdapter,
};
use crate::services::stream_check::HealthStatus;

const MODEL_CHECK_TIMEOUT_SECS: u64 = 45;
const MODEL_CHECK_DEGRADED_THRESHOLD_MS: u64 = 6_000;
const MODEL_CHECK_PROMPT: &str = "Reply with OK.";
const ERROR_BODY_MAX_CHARS: usize = 600;

const DEFAULT_CLAUDE_MODEL: &str = "claude-haiku-4-5-20251001";
const DEFAULT_CODEX_MODEL: &str = "gpt-4o-mini";
const DEFAULT_GEMINI_MODEL: &str = "gemini-2.5-flash";
const CODEX_OAUTH_ORIGINATOR: &str = "codex_cli_rs";
const CODEX_OAUTH_CLIENT_VERSION: &str = "0.144.1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCheckResult {
    pub status: HealthStatus,
    pub success: bool,
    pub message: String,
    pub response_time_ms: Option<u64>,
    pub http_status: Option<u16>,
    pub model_used: String,
    pub tested_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_category: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum CheckProtocol {
    Anthropic,
    OpenAi,
    Gemini,
}

pub struct ModelCheckService;

impl ModelCheckService {
    /// 发送一次最小流式请求，只等待首个非空响应块，然后立即关闭响应流。
    ///
    /// 该操作会消耗少量模型额度，因此只由用户主动点击的命令调用；它不会
    /// 自动运行，也不会修改故障转移熔断器或 Provider 配置。
    pub async fn check(
        app_type: &AppType,
        provider: &Provider,
        requested_model: Option<String>,
        auth_override: Option<AuthInfo>,
        base_url_override: Option<String>,
        claude_api_format_override: Option<String>,
        managed_account_id: Option<String>,
    ) -> Result<ModelCheckResult, AppError> {
        let model = Self::resolve_test_model(app_type, provider, requested_model.as_deref())
            .ok_or_else(|| {
                AppError::localized(
                    "model_check_model_missing",
                    "该供应商没有可用于测试的模型，请先在 Provider 配置中填写模型。",
                    "This provider has no model configured for testing. Add a model to the Provider first.",
                )
            })?;

        let started = Instant::now();
        let client = crate::proxy::http_client::get();
        let timeout = Duration::from_secs(MODEL_CHECK_TIMEOUT_SECS);

        let result = match app_type {
            AppType::Claude | AppType::ClaudeDesktop => {
                let adapter = ClaudeAdapter::new();
                let base_url = base_url_override
                    .or_else(|| adapter.extract_base_url(provider).ok())
                    .ok_or_else(|| AppError::Message("base_url 未配置".to_string()))?;
                let auth = auth_override
                    .or_else(|| adapter.extract_auth(provider))
                    .ok_or_else(|| AppError::Message("API Key 或托管认证未配置".to_string()))?;
                Self::check_claude(
                    &client,
                    &base_url,
                    &auth,
                    &model,
                    timeout,
                    provider,
                    claude_api_format_override.as_deref(),
                    None,
                    managed_account_id.as_deref(),
                )
                .await
            }
            AppType::Codex | AppType::GrokBuild => {
                Self::check_codex(
                    &client,
                    provider,
                    &model,
                    timeout,
                    auth_override,
                    base_url_override,
                    managed_account_id.as_deref(),
                )
                .await
            }
            AppType::Gemini => {
                let adapter = GeminiAdapter::new();
                let base_url = base_url_override
                    .or_else(|| adapter.extract_base_url(provider).ok())
                    .ok_or_else(|| AppError::Message("base_url 未配置".to_string()))?;
                let auth = auth_override
                    .or_else(|| adapter.extract_auth(provider))
                    .ok_or_else(|| {
                        AppError::Message("Gemini API Key 或 OAuth 未配置".to_string())
                    })?;
                Self::check_gemini(&client, &base_url, &auth, &model, timeout, provider, None).await
            }
            AppType::OpenCode => Self::check_opencode(&client, provider, &model, timeout).await,
            AppType::OpenClaw | AppType::Pi => {
                Self::check_structured_provider(&client, app_type, provider, &model, timeout).await
            }
            AppType::Hermes => Self::check_hermes(&client, provider, &model, timeout).await,
        };

        let elapsed = started.elapsed().as_millis() as u64;
        Ok(Self::build_result(result, elapsed, &model))
    }

    /// 与生产路由使用同一套 Provider 配置优先级，前端不需要再次填写 URL、Key 或模型。
    pub fn resolve_test_model(
        app_type: &AppType,
        provider: &Provider,
        requested_model: Option<&str>,
    ) -> Option<String> {
        let requested = requested_model
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string);
        if requested.is_some() {
            return requested;
        }

        let from_value = |value: Option<&Value>| {
            value
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
        };

        match app_type {
            AppType::Claude | AppType::ClaudeDesktop => from_value(
                provider
                    .settings_config
                    .get("env")
                    .and_then(|env| env.get("ANTHROPIC_MODEL")),
            )
            .or_else(|| {
                provider.meta.as_ref().and_then(|meta| {
                    meta.claude_desktop_model_routes
                        .values()
                        .map(|route| route.model.trim())
                        .find(|model| !model.is_empty())
                        .map(ToString::to_string)
                })
            })
            .or_else(|| {
                [
                    "ANTHROPIC_DEFAULT_SONNET_MODEL",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                    "ANTHROPIC_DEFAULT_OPUS_MODEL",
                ]
                .into_iter()
                .find_map(|key| {
                    from_value(
                        provider
                            .settings_config
                            .get("env")
                            .and_then(|env| env.get(key)),
                    )
                })
            })
            .or_else(|| Some(DEFAULT_CLAUDE_MODEL.to_string())),
            AppType::Codex | AppType::GrokBuild => {
                crate::proxy::providers::codex_provider_upstream_model(provider)
                    .or_else(|| Some(DEFAULT_CODEX_MODEL.to_string()))
            }
            AppType::Gemini => from_value(
                provider
                    .settings_config
                    .get("env")
                    .and_then(|env| env.get("GEMINI_MODEL")),
            )
            .or_else(|| from_value(provider.settings_config.get("model")))
            .or_else(|| Some(DEFAULT_GEMINI_MODEL.to_string())),
            AppType::OpenCode | AppType::OpenClaw | AppType::Hermes | AppType::Pi => {
                Self::first_model(provider.settings_config.get("models"))
            }
        }
    }

    fn first_model(value: Option<&Value>) -> Option<String> {
        match value {
            Some(Value::Array(items)) => items.iter().find_map(|item| {
                item.get("id")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|model| !model.is_empty())
                    .map(ToString::to_string)
            }),
            Some(Value::Object(models)) => models.keys().find_map(|model| {
                let model = model.trim();
                (!model.is_empty()).then(|| model.to_string())
            }),
            _ => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn check_claude(
        client: &Client,
        base_url: &str,
        auth: &AuthInfo,
        model: &str,
        timeout: Duration,
        provider: &Provider,
        api_format_override: Option<&str>,
        extra_headers: Option<&serde_json::Map<String, Value>>,
        managed_account_id: Option<&str>,
    ) -> Result<(u16, String), AppError> {
        if matches!(
            auth.strategy,
            AuthStrategy::CodexOAuth | AuthStrategy::XaiOAuth
        ) && auth.api_key.ends_with("_placeholder")
        {
            return Err(AppError::localized(
                "model_check_managed_auth",
                "该 Provider 使用托管 OAuth，请先在认证中心完成登录。",
                "This Provider uses managed OAuth. Finish sign-in in the authentication center first.",
            ));
        }

        let api_format = api_format_override
            .unwrap_or_else(|| crate::proxy::providers::get_claude_api_format(provider));
        let is_full_url = provider
            .meta
            .as_ref()
            .and_then(|meta| meta.is_full_url)
            .unwrap_or(false);
        let url = Self::resolve_claude_url(base_url, auth.strategy, api_format, is_full_url, model);

        let mut body = json!({
            "model": model,
            "max_tokens": 8,
            "messages": [{ "role": "user", "content": MODEL_CHECK_PROMPT }],
            "stream": true
        });

        let (mapped_body, _, _) = apply_model_mapping(body, provider);
        body = strip_one_m_suffix_for_upstream_from_body(mapped_body);

        let body = match api_format {
            "openai_responses" => anthropic_to_responses(
                body,
                provider
                    .meta
                    .as_ref()
                    .and_then(|meta| meta.prompt_cache_key.as_deref())
                    .or(Some(provider.id.as_str())),
                provider.is_codex_oauth(),
                provider.codex_fast_mode_enabled(),
            ),
            "openai_chat" => anthropic_to_openai(body),
            "gemini_native" => anthropic_to_gemini(body),
            _ => Ok(body),
        }
        .map_err(|error| AppError::Message(format!("构造模型测试请求失败: {error}")))?;

        let is_copilot = auth.strategy == AuthStrategy::GitHubCopilot;
        let is_gemini = api_format == "gemini_native";
        let is_openai = matches!(api_format, "openai_chat" | "openai_responses");

        let mut request = client
            .post(&url)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .header("accept-encoding", "identity");

        if is_copilot {
            let request_id = uuid::Uuid::new_v4().to_string();
            request = request
                .header("authorization", format!("Bearer {}", auth.api_key))
                .header("user-agent", copilot_auth::COPILOT_USER_AGENT)
                .header("editor-version", copilot_auth::COPILOT_EDITOR_VERSION)
                .header(
                    "editor-plugin-version",
                    copilot_auth::COPILOT_PLUGIN_VERSION,
                )
                .header(
                    "copilot-integration-id",
                    copilot_auth::COPILOT_INTEGRATION_ID,
                )
                .header("x-github-api-version", copilot_auth::COPILOT_API_VERSION)
                .header("openai-intent", "conversation-agent")
                .header("x-initiator", "user")
                .header("x-interaction-type", "conversation-agent")
                .header("x-request-id", &request_id)
                .header("x-agent-task-id", &request_id);
        } else if is_gemini {
            let adapter = GeminiAdapter::new();
            for (name, value) in adapter
                .get_auth_headers(auth)
                .map_err(|error| AppError::Message(error.to_string()))?
            {
                request = request.header(name, value);
            }
        } else if is_openai {
            if auth.strategy == AuthStrategy::CodexOAuth {
                for (name, value) in ClaudeAdapter::new()
                    .get_auth_headers(auth)
                    .map_err(|error| AppError::Message(error.to_string()))?
                {
                    request = request.header(name, value);
                }
            } else {
                request = request.header("authorization", format!("Bearer {}", auth.api_key));
            }
        } else {
            for (name, value) in ClaudeAdapter::new()
                .get_auth_headers(auth)
                .map_err(|error| AppError::Message(error.to_string()))?
            {
                request = request.header(name, value);
            }
            request = request
                .header("anthropic-version", "2023-06-01")
                .header(
                    "anthropic-beta",
                    "claude-code-20250219,interleaved-thinking-2025-05-14",
                )
                .header("anthropic-dangerous-direct-browser-access", "true")
                .header("accept-language", "*")
                .header("user-agent", "claude-cli/2.1.2 (external, cli)")
                .header("x-app", "cli");
        }

        if let Some(headers) = extra_headers {
            request = Self::apply_extra_headers(request, headers);
        }
        if auth.strategy == AuthStrategy::CodexOAuth {
            if let Some(account_id) = managed_account_id {
                request = request.header("chatgpt-account-id", account_id);
            }
        }
        if !is_copilot {
            if let Some(user_agent) = Self::custom_user_agent(provider) {
                request = request.header("user-agent", user_agent);
            }
        }

        Self::send_json_and_validate(
            request,
            &body,
            timeout,
            model,
            if is_gemini {
                CheckProtocol::Gemini
            } else if is_openai {
                CheckProtocol::OpenAi
            } else {
                CheckProtocol::Anthropic
            },
        )
        .await
    }

    async fn check_codex(
        client: &Client,
        provider: &Provider,
        model: &str,
        timeout: Duration,
        auth_override: Option<AuthInfo>,
        base_url_override: Option<String>,
        managed_account_id: Option<&str>,
    ) -> Result<(u16, String), AppError> {
        let adapter = CodexAdapter::new();
        let base_url = base_url_override
            .or_else(|| adapter.extract_base_url(provider).ok())
            .ok_or_else(|| AppError::Message("base_url 未配置".to_string()))?;
        let auth = auth_override
            .or_else(|| adapter.extract_auth(provider))
            .ok_or_else(|| AppError::Message("API Key 或托管认证未配置".to_string()))?;

        // Codex 的 Anthropic 上游需要发送 Messages 格式，不能把 Responses body
        // 直接打到 /messages；走同一个 Claude 请求构造器可保持鉴权和转换一致。
        if should_convert_codex_responses_to_anthropic(provider, "/responses") {
            return Self::check_claude(
                client,
                &base_url,
                &auth,
                model,
                timeout,
                provider,
                Some("anthropic"),
                None,
                managed_account_id,
            )
            .await;
        }

        if matches!(
            auth.strategy,
            AuthStrategy::CodexOAuth | AuthStrategy::XaiOAuth
        ) && auth.api_key.ends_with("_placeholder")
        {
            return Err(AppError::localized(
                "model_check_managed_auth",
                "该 Provider 使用托管 OAuth，请先在认证中心完成登录。",
                "This Provider uses managed OAuth. Finish sign-in in the authentication center first.",
            ));
        }

        let uses_chat = codex_provider_uses_chat_completions(provider);
        let endpoint = if uses_chat {
            "/chat/completions"
        } else {
            "/responses"
        };
        let is_full_url = provider
            .meta
            .as_ref()
            .and_then(|meta| meta.is_full_url)
            .unwrap_or(false);
        let urls = Self::resolve_codex_urls(&base_url, is_full_url, endpoint);
        let (actual_model, reasoning_effort) = Self::parse_model_with_effort(model);
        let mut body = if uses_chat {
            json!({
                "model": actual_model,
                "messages": [{ "role": "user", "content": MODEL_CHECK_PROMPT }],
                "max_tokens": 1,
                "stream": true
            })
        } else {
            json!({
                "model": actual_model,
                "input": [{ "role": "user", "content": MODEL_CHECK_PROMPT }],
                "stream": true,
                "store": false
            })
        };

        if uses_chat {
            crate::proxy::providers::apply_codex_chat_upstream_model(provider, &mut body);
            if let Some(effort) = reasoning_effort.as_deref() {
                if crate::proxy::providers::transform::supports_reasoning_effort(&actual_model) {
                    body["reasoning_effort"] = json!(effort);
                }
            }
        } else {
            crate::proxy::providers::apply_codex_upstream_model(provider, &mut body);
            if let Some(effort) = reasoning_effort {
                body["reasoning"] = json!({ "effort": effort });
            }
        }

        let user_agent = Self::custom_user_agent(provider).unwrap_or_else(|| {
            HeaderValue::from_static("codex_cli_rs/0.80.0 (Windows 10; x86_64) Terminal")
        });

        for (index, url) in urls.iter().enumerate() {
            let mut request = client
                .post(url)
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .header("accept-encoding", "identity")
                .header("user-agent", user_agent.clone())
                .header("originator", "codex_cli_rs");
            for (name, value) in adapter
                .get_auth_headers(&auth)
                .map_err(|error| AppError::Message(error.to_string()))?
            {
                request = request.header(name, value);
            }
            if auth.strategy == AuthStrategy::CodexOAuth {
                request = request
                    .header("originator", CODEX_OAUTH_ORIGINATOR)
                    .header("version", CODEX_OAUTH_CLIENT_VERSION);
                if let Some(account_id) = managed_account_id {
                    request = request.header("chatgpt-account-id", account_id);
                }
            }

            match Self::send_json_and_validate(
                request,
                &body,
                timeout,
                &actual_model,
                CheckProtocol::OpenAi,
            )
            .await
            {
                Ok(result) => return Ok(result),
                Err(AppError::HttpStatus { status: 404, .. }) if index == 0 && urls.len() > 1 => {
                    continue
                }
                Err(error) => return Err(error),
            }
        }

        Err(AppError::Message("没有可用的 Codex 请求端点".to_string()))
    }

    async fn check_gemini(
        client: &Client,
        base_url: &str,
        auth: &AuthInfo,
        model: &str,
        timeout: Duration,
        provider: &Provider,
        extra_headers: Option<&serde_json::Map<String, Value>>,
    ) -> Result<(u16, String), AppError> {
        let normalized_model = normalize_gemini_model_id(model);
        let endpoint = format!("/v1beta/models/{normalized_model}:streamGenerateContent?alt=sse");
        let is_full_url = provider
            .meta
            .as_ref()
            .and_then(|meta| meta.is_full_url)
            .unwrap_or(false);
        let url = resolve_gemini_native_url(base_url, &endpoint, is_full_url);
        let body = json!({
            "contents": [{
                "role": "user",
                "parts": [{ "text": MODEL_CHECK_PROMPT }]
            }]
        });

        let adapter = GeminiAdapter::new();
        let mut request = client
            .post(url)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");
        for (name, value) in adapter
            .get_auth_headers(auth)
            .map_err(|error| AppError::Message(error.to_string()))?
        {
            request = request.header(name, value);
        }
        if let Some(headers) = extra_headers {
            request = Self::apply_extra_headers(request, headers);
        }
        if let Some(user_agent) = Self::custom_user_agent(provider) {
            request = request.header("user-agent", user_agent);
        }

        Self::send_json_and_validate(request, &body, timeout, model, CheckProtocol::Gemini).await
    }

    async fn check_structured_provider(
        client: &Client,
        app_type: &AppType,
        provider: &Provider,
        model: &str,
        timeout: Duration,
    ) -> Result<(u16, String), AppError> {
        let app_name = match app_type {
            AppType::OpenClaw => "OpenClaw",
            AppType::Pi => "Pi",
            _ => unreachable!("structured provider check only supports OpenClaw and Pi"),
        };

        if provider
            .settings_config
            .get("authHeader")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(AppError::localized(
                "model_check_custom_auth_header",
                format!(
                    "该 {app_name} Provider 使用自定义认证头，无法安全推断测试请求。请直接在 {app_name} 中测试。"
                ),
                format!(
                    "This {app_name} Provider uses a custom auth header that CC Switch cannot safely infer. Test it directly in {app_name}."
                ),
            ));
        }

        let base_url = Self::required_string(provider.settings_config.get("baseUrl"), "baseUrl")?;
        let api_key = Self::required_string(provider.settings_config.get("apiKey"), "apiKey")?;
        let protocol = provider
            .settings_config
            .get("api")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| AppError::Message(format!("{app_name} 缺少 api 协议配置")))?;
        let headers = provider
            .settings_config
            .get("headers")
            .and_then(Value::as_object);

        match protocol {
            "openai-completions" => {
                Self::check_claude(
                    client,
                    &base_url,
                    &AuthInfo::new(api_key, AuthStrategy::Bearer),
                    model,
                    timeout,
                    provider,
                    Some("openai_chat"),
                    headers,
                    None,
                )
                .await
            }
            "openai-responses" => {
                Self::check_claude(
                    client,
                    &base_url,
                    &AuthInfo::new(api_key, AuthStrategy::Bearer),
                    model,
                    timeout,
                    provider,
                    Some("openai_responses"),
                    headers,
                    None,
                )
                .await
            }
            "anthropic-messages" => {
                Self::check_claude(
                    client,
                    &base_url,
                    &AuthInfo::new(api_key, AuthStrategy::ClaudeAuth),
                    model,
                    timeout,
                    provider,
                    Some("anthropic"),
                    headers,
                    None,
                )
                .await
            }
            "google-generative-ai" => {
                Self::check_gemini(
                    client,
                    &base_url,
                    &AuthInfo::new(api_key, AuthStrategy::Google),
                    model,
                    timeout,
                    provider,
                    headers,
                )
                .await
            }
            "bedrock-converse-stream" => Err(AppError::localized(
                "model_check_bedrock_not_supported",
                format!(
                    "AWS Bedrock 需要 SigV4 签名，当前不支持独立模型测试。请直接通过 {app_name} 验证。"
                ),
                format!(
                    "AWS Bedrock requires SigV4 signing and is not supported by the standalone model test. Verify it through {app_name}."
                ),
            )),
            other => Err(AppError::Message(format!(
                "{app_name} 暂不支持协议: {other}"
            ))),
        }
    }

    async fn check_opencode(
        client: &Client,
        provider: &Provider,
        model: &str,
        timeout: Duration,
    ) -> Result<(u16, String), AppError> {
        let npm = provider
            .settings_config
            .get("npm")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| AppError::Message("OpenCode 缺少 npm 协议配置".to_string()))?;
        let options = provider
            .settings_config
            .get("options")
            .ok_or_else(|| AppError::Message("OpenCode 缺少 options 配置".to_string()))?;
        let api_key = Self::required_string(options.get("apiKey"), "options.apiKey")?;
        let base_url = options
            .get("baseURL")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .or_else(|| match npm {
                "@ai-sdk/openai" => Some("https://api.openai.com/v1".to_string()),
                "@ai-sdk/anthropic" => Some("https://api.anthropic.com".to_string()),
                "@ai-sdk/google" => Some("https://generativelanguage.googleapis.com".to_string()),
                _ => None,
            })
            .ok_or_else(|| AppError::Message("OpenCode 缺少 options.baseURL".to_string()))?;
        let headers = options.get("headers").and_then(Value::as_object);

        match npm {
            "@ai-sdk/openai-compatible" => {
                Self::check_claude(
                    client,
                    &base_url,
                    &AuthInfo::new(api_key, AuthStrategy::Bearer),
                    model,
                    timeout,
                    provider,
                    Some("openai_chat"),
                    headers,
                    None,
                )
                .await
            }
            "@ai-sdk/openai" => {
                Self::check_claude(
                    client,
                    &base_url,
                    &AuthInfo::new(api_key, AuthStrategy::Bearer),
                    model,
                    timeout,
                    provider,
                    Some("openai_responses"),
                    headers,
                    None,
                )
                .await
            }
            "@ai-sdk/anthropic" => {
                Self::check_claude(
                    client,
                    &base_url,
                    &AuthInfo::new(api_key, AuthStrategy::ClaudeAuth),
                    model,
                    timeout,
                    provider,
                    Some("anthropic"),
                    headers,
                    None,
                )
                .await
            }
            "@ai-sdk/google" => {
                Self::check_gemini(
                    client,
                    &base_url,
                    &AuthInfo::new(api_key, AuthStrategy::Google),
                    model,
                    timeout,
                    provider,
                    headers,
                )
                .await
            }
            "@ai-sdk/amazon-bedrock" => Err(AppError::localized(
                "model_check_bedrock_not_supported",
                "AWS Bedrock 需要 SigV4 签名，当前不支持独立模型测试。请直接通过 OpenCode 验证。",
                "AWS Bedrock requires SigV4 signing and is not supported by the standalone model test. Verify it through OpenCode.",
            )),
            other => Err(AppError::Message(format!("OpenCode 暂不支持 SDK 包: {other}"))),
        }
    }

    async fn check_hermes(
        client: &Client,
        provider: &Provider,
        model: &str,
        timeout: Duration,
    ) -> Result<(u16, String), AppError> {
        let base_url = Self::required_string(provider.settings_config.get("base_url"), "base_url")?;
        let api_key = Self::required_string(provider.settings_config.get("api_key"), "api_key")?;
        let api_mode = provider
            .settings_config
            .get("api_mode")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| AppError::Message("Hermes 缺少 api_mode 配置".to_string()))?;
        let (format, strategy) = match api_mode {
            "chat_completions" => ("openai_chat", AuthStrategy::Bearer),
            "anthropic_messages" => ("anthropic", AuthStrategy::ClaudeAuth),
            "codex_responses" => ("openai_responses", AuthStrategy::Bearer),
            "bedrock_converse" => {
                return Err(AppError::localized(
                    "model_check_bedrock_not_supported",
                    "AWS Bedrock 需要 SigV4 签名，当前不支持独立模型测试。请直接通过 Hermes 验证。",
                    "AWS Bedrock requires SigV4 signing and is not supported by the standalone model test. Verify it through Hermes.",
                ));
            }
            other => return Err(AppError::Message(format!("Hermes 暂不支持协议: {other}"))),
        };

        Self::check_claude(
            client,
            &base_url,
            &AuthInfo::new(api_key, strategy),
            model,
            timeout,
            provider,
            Some(format),
            None,
            None,
        )
        .await
    }

    fn apply_extra_headers(
        mut request: RequestBuilder,
        headers: &serde_json::Map<String, Value>,
    ) -> RequestBuilder {
        for (key, value) in headers {
            if let Some(value) = value.as_str() {
                request = request.header(key, value);
            }
        }
        request
    }

    fn required_string(value: Option<&Value>, field: &str) -> Result<String, AppError> {
        value
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .ok_or_else(|| AppError::Message(format!("缺少 {field}")))
    }

    fn custom_user_agent(provider: &Provider) -> Option<HeaderValue> {
        provider
            .meta
            .as_ref()
            .and_then(|meta| meta.custom_user_agent_header().ok().flatten())
    }

    fn resolve_claude_url(
        base_url: &str,
        auth_strategy: AuthStrategy,
        api_format: &str,
        is_full_url: bool,
        model: &str,
    ) -> String {
        if api_format == "gemini_native" {
            let model = normalize_gemini_model_id(model);
            return resolve_gemini_native_url(
                base_url,
                &format!("/v1beta/models/{model}:streamGenerateContent?alt=sse"),
                is_full_url,
            );
        }
        if is_full_url {
            return base_url.to_string();
        }

        let base = base_url.trim_end_matches('/');
        let is_copilot = auth_strategy == AuthStrategy::GitHubCopilot;
        match (is_copilot, api_format) {
            (true, "openai_responses") => format!("{base}/v1/responses"),
            (true, _) => format!("{base}/chat/completions"),
            (_, "openai_responses") if base.ends_with("/v1") => format!("{base}/responses"),
            (_, "openai_responses") => format!("{base}/v1/responses"),
            (_, "openai_chat") if base.ends_with("/v1") => format!("{base}/chat/completions"),
            (_, "openai_chat") => format!("{base}/v1/chat/completions"),
            (_, _) if base.ends_with("/v1") => format!("{base}/messages"),
            (_, _) => format!("{base}/v1/messages"),
        }
    }

    fn resolve_codex_urls(base_url: &str, is_full_url: bool, endpoint: &str) -> Vec<String> {
        if is_full_url {
            return vec![base_url.to_string()];
        }

        let base = base_url.trim_end_matches('/');
        let endpoint = endpoint.trim_start_matches('/');
        let primary = CodexAdapter::new().build_url(base, endpoint);
        let fallback = format!("{base}/{endpoint}");
        if primary == fallback {
            vec![primary]
        } else {
            vec![primary, fallback]
        }
    }

    fn parse_model_with_effort(model: &str) -> (String, Option<String>) {
        let Some(index) = model.find('@').or_else(|| model.find('#')) else {
            return (model.to_string(), None);
        };
        let actual = model[..index].trim();
        let effort = model[index + 1..].trim();
        if actual.is_empty() {
            return (model.to_string(), None);
        }
        (
            actual.to_string(),
            (!effort.is_empty()).then(|| effort.to_string()),
        )
    }

    async fn send_json_and_validate(
        request: RequestBuilder,
        body: &Value,
        timeout: Duration,
        model: &str,
        protocol: CheckProtocol,
    ) -> Result<(u16, String), AppError> {
        let response = request
            .timeout(timeout)
            .json(body)
            .send()
            .await
            .map_err(Self::map_request_error)?;
        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Self::http_status_error(status, body));
        }

        let mut stream = response.bytes_stream();
        let mut pending = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|error| AppError::Message(format!("读取模型响应失败: {error}")))?;
            if chunk.is_empty() {
                continue;
            }
            pending.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline) = pending.find('\n') {
                let line = pending[..newline].to_string();
                pending.drain(..=newline);
                if let Some(result) = Self::validate_response_line(&line, protocol) {
                    result?;
                    return Ok((status, model.to_string()));
                }
            }
        }

        if let Some(result) = Self::validate_response_line(&pending, protocol) {
            result?;
            return Ok((status, model.to_string()));
        }

        Err(AppError::Message(
            "模型响应为空，未收到有效数据".to_string(),
        ))
    }

    fn validate_response_line(line: &str, protocol: CheckProtocol) -> Option<Result<(), AppError>> {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with(':')
            || trimmed.starts_with("id:")
            || trimmed.starts_with("retry:")
        {
            return None;
        }

        if let Some(event) = trimmed.strip_prefix("event:") {
            return match event.trim().to_ascii_lowercase().as_str() {
                "error" | "response.failed" => {
                    Some(Err(AppError::Message("上游返回了错误事件".to_string())))
                }
                _ => None,
            };
        }

        let payload = trimmed
            .strip_prefix("data:")
            .map(str::trim)
            .unwrap_or(trimmed);
        if payload.is_empty() || payload.eq_ignore_ascii_case("[DONE]") {
            return None;
        }

        if serde_json::from_str::<Value>(payload)
            .ok()
            .and_then(|value| value.get("type").and_then(Value::as_str).map(str::to_owned))
            .as_deref()
            == Some("ping")
        {
            return None;
        }

        if Self::looks_like_error_payload(payload, protocol) {
            return Some(Err(AppError::Message("上游返回了错误响应体".to_string())));
        }

        Some(Ok(()))
    }

    fn looks_like_error_payload(text: &str, _protocol: CheckProtocol) -> bool {
        if let Ok(value) = serde_json::from_str::<Value>(text) {
            if value.get("error").is_some_and(|error| !error.is_null()) {
                return true;
            }
            if matches!(
                value.get("type").and_then(Value::as_str),
                Some("error" | "response.failed")
            ) {
                return true;
            }
        }

        let lower = text.to_ascii_lowercase();
        lower.contains("\"error\"")
            && !lower.contains("\"error\":null")
            && !lower.contains("message_start")
            && !lower.contains("candidates")
    }

    fn build_result(
        result: Result<(u16, String), AppError>,
        response_time_ms: u64,
        fallback_model: &str,
    ) -> ModelCheckResult {
        match result {
            Ok((status, model)) => ModelCheckResult {
                status: if response_time_ms <= MODEL_CHECK_DEGRADED_THRESHOLD_MS {
                    HealthStatus::Operational
                } else {
                    HealthStatus::Degraded
                },
                success: true,
                message: "模型请求成功".to_string(),
                response_time_ms: Some(response_time_ms),
                http_status: Some(status),
                model_used: model,
                tested_at: chrono::Utc::now().timestamp(),
                error_category: None,
            },
            Err(error) => {
                let (http_status, message, error_category) = match &error {
                    AppError::HttpStatus { status, body } => {
                        let category = Self::classify_http_error(*status, body);
                        (
                            Some(*status),
                            format!("{} (HTTP {status})", Self::category_message(category)),
                            Some(category.to_string()),
                        )
                    }
                    _ => {
                        let category = Self::classify_message(&error.to_string());
                        (None, error.to_string(), Some(category.to_string()))
                    }
                };
                ModelCheckResult {
                    status: HealthStatus::Failed,
                    success: false,
                    message,
                    response_time_ms: Some(response_time_ms),
                    http_status,
                    model_used: fallback_model.to_string(),
                    tested_at: chrono::Utc::now().timestamp(),
                    error_category,
                }
            }
        }
    }

    fn classify_http_error(status: u16, body: &str) -> &'static str {
        let lower = body.to_ascii_lowercase();
        if [
            "quota_exceeded",
            "quota exceeded",
            "billing",
            "payment required",
            "coding_plan_hour_quota_exceeded",
            "coding_plan_week_quota_exceeded",
            "coding_plan_month_quota_exceeded",
        ]
        .iter()
        .any(|indicator| lower.contains(indicator))
            || status == 402
        {
            return "quotaExceeded";
        }
        if status == 429 {
            return "rateLimited";
        }
        if status == 401 || status == 403 || lower.contains("invalid api key") {
            return "auth";
        }
        if (status == 400 || status == 404 || status == 422)
            && lower.contains("model")
            && [
                "model_not_found",
                "model not found",
                "does not exist",
                "invalid model",
                "unknown model",
                "not_found_error",
                "is not a valid model",
            ]
            .iter()
            .any(|indicator| lower.contains(indicator))
        {
            return "modelNotFound";
        }
        if status == 404 {
            return "endpoint";
        }
        if status >= 500 {
            return "upstream";
        }
        "protocol"
    }

    fn classify_message(message: &str) -> &'static str {
        let lower = message.to_ascii_lowercase();
        if lower.contains("api key")
            || lower.contains("认证")
            || lower.contains("托管 oauth")
            || lower.contains("managed oauth")
        {
            "auth"
        } else if lower.contains("timeout")
            || lower.contains("timed out")
            || lower.contains("connection")
            || lower.contains("dns")
            || lower.contains("tls")
        {
            "network"
        } else if lower.contains("响应") || lower.contains("response") {
            "protocol"
        } else {
            "configuration"
        }
    }

    fn category_message(category: &str) -> &'static str {
        match category {
            "auth" => "鉴权失败",
            "modelNotFound" => "模型不存在或无权访问",
            "endpoint" => "接口地址不存在",
            "quotaExceeded" => "额度或计费限制",
            "rateLimited" => "请求被限流",
            "upstream" => "上游服务错误",
            "network" => "网络请求失败",
            "configuration" => "Provider 配置不完整",
            _ => "协议或请求格式错误",
        }
    }

    fn http_status_error(status: u16, body: String) -> AppError {
        let body = body.trim();
        let body = if body.chars().count() > ERROR_BODY_MAX_CHARS {
            let mut truncated: String = body.chars().take(ERROR_BODY_MAX_CHARS).collect();
            truncated.push('…');
            truncated
        } else {
            body.to_string()
        };
        AppError::HttpStatus { status, body }
    }

    fn map_request_error(error: reqwest::Error) -> AppError {
        if error.is_timeout() {
            AppError::Message("请求超时".to_string())
        } else if error.is_connect() {
            AppError::Message(format!("连接失败: {error}"))
        } else {
            AppError::Message(format!("请求失败: {error}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(config: Value) -> Provider {
        Provider::with_id("test".to_string(), "Test".to_string(), config, None)
    }

    #[test]
    fn resolves_provider_model_before_default() {
        let provider = provider(json!({
            "env": { "ANTHROPIC_MODEL": "provider-model" }
        }));
        assert_eq!(
            ModelCheckService::resolve_test_model(&AppType::Claude, &provider, None),
            Some("provider-model".to_string())
        );
        assert_eq!(
            ModelCheckService::resolve_test_model(
                &AppType::Claude,
                &provider,
                Some("explicit-model")
            ),
            Some("explicit-model".to_string())
        );
    }

    #[test]
    fn resolves_first_model_from_additive_provider() {
        let provider = provider(json!({
            "models": [
                { "id": "first-model" },
                { "id": "second-model" }
            ]
        }));
        assert_eq!(
            ModelCheckService::resolve_test_model(&AppType::OpenClaw, &provider, None),
            Some("first-model".to_string())
        );
    }

    #[test]
    fn classifies_model_and_auth_errors() {
        assert_eq!(
            ModelCheckService::classify_http_error(
                404,
                r#"{"error":{"code":"model_not_found","message":"model missing"}}"#
            ),
            "modelNotFound"
        );
        assert_eq!(
            ModelCheckService::classify_http_error(401, "invalid api key"),
            "auth"
        );
        assert_eq!(
            ModelCheckService::classify_http_error(429, "slow down"),
            "rateLimited"
        );
        assert_eq!(
            ModelCheckService::classify_http_error(404, "Not Found"),
            "endpoint"
        );
    }

    #[test]
    fn parses_codex_reasoning_suffix() {
        assert_eq!(
            ModelCheckService::parse_model_with_effort("gpt-5@low"),
            ("gpt-5".to_string(), Some("low".to_string()))
        );
        assert_eq!(
            ModelCheckService::parse_model_with_effort("gpt-4o"),
            ("gpt-4o".to_string(), None)
        );
    }

    #[test]
    fn builds_claude_endpoints_for_protocols() {
        assert_eq!(
            ModelCheckService::resolve_claude_url(
                "https://relay.example/v1",
                AuthStrategy::Bearer,
                "openai_chat",
                false,
                "gpt-4o"
            ),
            "https://relay.example/v1/chat/completions"
        );
        assert_eq!(
            ModelCheckService::resolve_claude_url(
                "https://relay.example",
                AuthStrategy::ClaudeAuth,
                "anthropic",
                false,
                "claude-sonnet"
            ),
            "https://relay.example/v1/messages"
        );
    }

    #[test]
    fn validates_sse_control_lines_before_accepting_model_data() {
        assert!(
            ModelCheckService::validate_response_line(": keep-alive", CheckProtocol::OpenAi)
                .is_none()
        );
        assert!(
            ModelCheckService::validate_response_line("event: ping", CheckProtocol::OpenAi)
                .is_none()
        );
        assert!(
            ModelCheckService::validate_response_line("id: 1", CheckProtocol::OpenAi).is_none()
        );
        assert!(
            ModelCheckService::validate_response_line("data: [DONE]", CheckProtocol::OpenAi)
                .is_none()
        );
        assert!(ModelCheckService::validate_response_line(
            r#"data: {"type":"ping"}"#,
            CheckProtocol::OpenAi
        )
        .is_none());
        assert!(ModelCheckService::validate_response_line(
            r#"data: {"type":"message_start"}"#,
            CheckProtocol::Anthropic
        )
        .expect("message event")
        .is_ok());
        assert!(ModelCheckService::validate_response_line(
            r#"data: {"error":{"message":"invalid model"}}"#,
            CheckProtocol::OpenAi
        )
        .expect("error event")
        .is_err());
    }
}
