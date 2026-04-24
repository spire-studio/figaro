function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asText(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export function extractErrorMessage(error: unknown): string | null {
  if (error instanceof Error) {
    return asText(error.message);
  }

  const direct = asText(error);
  if (direct) return direct;

  if (!isRecord(error)) return null;

  const message = asText(error.message);
  if (message) return message;

  const detail = error.detail;
  const detailText = asText(detail);
  if (detailText) return detailText;

  if (Array.isArray(detail) && detail.length > 0) {
    return "Request validation failed";
  }

  if (isRecord(detail)) {
    const nestedMessage = asText(detail.message);
    if (nestedMessage) return nestedMessage;
    const nestedDetail = asText(detail.detail);
    if (nestedDetail) return nestedDetail;
  }

  return null;
}

export async function getHttpErrorMessage(response: Response): Promise<string> {
  const fallback = `HTTP ${response.status}`;
  try {
    const payload = (await response.json()) as unknown;
    return extractErrorMessage(payload) ?? fallback;
  } catch {
    return fallback;
  }
}
