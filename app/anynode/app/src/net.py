import network, time
SSID="{{WIFI_SSID}}"
[REDACTED-SECRET-LINE]

def wifi_connect():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
[REDACTED-SECRET-LINE]
        for _ in range(60):
            if wlan.isconnected(): break
            time.sleep(0.5)
    print("WiFi:", wlan.ifconfig())

