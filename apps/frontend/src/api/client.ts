import createClient from "openapi-fetch";
import type { paths } from "./openapi";

export const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export const api = createClient<paths>({
  baseUrl
});
