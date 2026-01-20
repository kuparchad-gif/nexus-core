#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_event.h"
#include "esp_netif.h"

extern void wifi_init_sta(void);
extern esp_err_t gate_send(const char* text, char* out, size_t out_sz);

static const char *TAG = "APP";

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_sta();

    vTaskDelay(pdMS_TO_TICKS(3000)); // wait for DHCP

    char buf[2048];
    while (1) {
        ESP_LOGI(TAG, "Sending to router...");
        if (gate_send("sensor: room check", buf, sizeof(buf)) == ESP_OK) {
            ESP_LOGI(TAG, "Router response: %s", buf);
        } else {
            ESP_LOGW(TAG, "Router call failed");
        }
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
