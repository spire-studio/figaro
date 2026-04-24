import createClient from "openapi-fetch";
import type { paths } from "./openapi";

export const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const api = createClient<paths>({
  baseUrl
});
