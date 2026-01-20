window.HERMES_CONFIG = {
  // When served through nginx in this bundle, API is on same origin under /api
  API_BASE: (location.origin + '/api').replace(/\/+$/, ''),
  GUEST_ENABLED: true
};
