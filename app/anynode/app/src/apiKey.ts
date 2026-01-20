/**
 * Retrieves the API key from the window object.
 * The key is injected at runtime by the Docker entrypoint script via `env.js`.
 * This is a secure way to handle secrets in a client-side application without
 * embedding them in the build files.
 * @returns The API key string, or null if not found.
 */
export function getApiKey(): string | null {
  const apiKey = (window as any).VITE_API_KEY;
  if (apiKey && apiKey !== 'undefined' && apiKey.length > 0) {
    return apiKey;
  }
  return null;
}
