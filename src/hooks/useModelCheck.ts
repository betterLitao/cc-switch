import { useCallback, useState } from "react";
import { toast } from "sonner";
import { useTranslation } from "react-i18next";
import {
  modelCheckProvider,
  type ModelCheckResult,
} from "@/lib/api/model-check";
import type { AppId } from "@/lib/api";

/**
 * 供应商模型可用性检查。
 *
 * 该操作是用户主动触发的真实请求，可能消耗少量额度；它与只探测 HTTP
 * 响应的 useStreamCheck 分开维护，避免两个按钮的语义和加载状态混淆。
 */
export function useModelCheck(appId: AppId) {
  const { t } = useTranslation();
  const [checkingIds, setCheckingIds] = useState<Set<string>>(new Set());

  const checkProvider = useCallback(
    async (
      providerId: string,
      providerName: string,
    ): Promise<ModelCheckResult | null> => {
      setCheckingIds((prev) => new Set(prev).add(providerId));

      try {
        const result = await modelCheckProvider(appId, providerId);
        const model = result.modelUsed || "?";
        const responseTime = result.responseTimeMs ?? "?";

        if (result.success && result.status === "operational") {
          toast.success(
            t("modelCheck.success", {
              providerName,
              model,
              responseTimeMs: responseTime,
              defaultValue: `${providerName} 的模型 ${model} 可用 (${responseTime}ms)`,
            }),
            { closeButton: true },
          );
        } else if (result.success) {
          toast.warning(
            t("modelCheck.degraded", {
              providerName,
              model,
              responseTimeMs: responseTime,
              defaultValue: `${providerName} 的模型 ${model} 可用但响应较慢 (${responseTime}ms)`,
            }),
          );
        } else {
          toast.error(
            t("modelCheck.failed", {
              providerName,
              model,
              message: result.message,
              defaultValue: `${providerName} 的模型 ${model} 不可用: ${result.message}`,
            }),
            {
              description: t("modelCheck.failedHint", {
                defaultValue:
                  "这是真实模型请求结果；请检查模型名、API Key、接口协议和额度。",
              }),
              duration: 9000,
              closeButton: true,
            },
          );
        }

        return result;
      } catch (error) {
        toast.error(
          t("modelCheck.error", {
            providerName,
            error: String(error),
            defaultValue: `${providerName} 模型测试出错: ${String(error)}`,
          }),
          { duration: 9000, closeButton: true },
        );
        return null;
      } finally {
        setCheckingIds((prev) => {
          const next = new Set(prev);
          next.delete(providerId);
          return next;
        });
      }
    },
    [appId, t],
  );

  const isChecking = useCallback(
    (providerId: string) => checkingIds.has(providerId),
    [checkingIds],
  );

  return { checkProvider, isChecking };
}
