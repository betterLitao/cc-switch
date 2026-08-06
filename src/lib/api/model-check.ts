import { invoke } from "@tauri-apps/api/core";
import type { AppId } from "./types";

export type ModelCheckHealthStatus = "operational" | "degraded" | "failed";

export interface ModelCheckResult {
  status: ModelCheckHealthStatus;
  success: boolean;
  message: string;
  responseTimeMs?: number;
  httpStatus?: number;
  modelUsed: string;
  testedAt: number;
  errorCategory?: string;
}

/**
 * 使用已保存的 Provider 配置发送一次最小真实模型请求。
 * requestedModel 为空时由后端从 Provider 配置中选择模型。
 */
export async function modelCheckProvider(
  appType: AppId,
  providerId: string,
  requestedModel?: string,
): Promise<ModelCheckResult> {
  return invoke("model_check_provider", {
    appType,
    providerId,
    requestedModel: requestedModel?.trim() || undefined,
  });
}
