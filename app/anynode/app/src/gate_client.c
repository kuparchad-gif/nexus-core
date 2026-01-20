#include <string.h>
#include "esp_log.h"
#include "esp_http_client.h"
#include "cJSON.h"

#define ROUTER_URL "{{ROUTER_URL}}"
static const char *TAG = "GATE";

esp_err_t gate_send(const char* text, char* out, size_t out_sz) {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "text", text);
    cJSON_AddStringToObject(root, "mode", NULL);
    cJSON_AddNumberToObject(root, "max_new_tokens", 64);
    char *payload = cJSON_PrintUnformatted(root);

    esp_http_client_config_t cfg = {
        .url = ROUTER_URL,
        .timeout_ms = 8000
    };
    esp_http_client_handle_t h = esp_http_client_init(&cfg);
    if (!h) { cJSON_free(payload); cJSON_Delete(root); return ESP_FAIL; }

    esp_http_client_set_method(h, HTTP_METHOD_POST);
    esp_http_client_set_header(h, "Content-Type", "application/json");
    esp_http_client_set_post_field(h, payload, strlen(payload));

    esp_err_t err = ESP_OK;
    if ((err = esp_http_client_perform(h)) == ESP_OK) {
        int len = esp_http_client_read_response(h, out, out_sz - 1);
        if (len < 0) { err = ESP_FAIL; }
        else { out[len] = 0; }
    }

    esp_http_client_cleanup(h);
    cJSON_free(payload);
    cJSON_Delete(root);
    return err;
}
