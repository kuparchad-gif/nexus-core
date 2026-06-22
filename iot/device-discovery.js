export class IoTDiscovery {
  constructor(env) {
    this.env = env;
    this.discoveredDevices = new Map();
    this.providers = new Map();
    this.discoveryInterval = null;
    this.onDiscovery = null;
  }

  async startDiscovery() {
    console.log(`📡 IoT Discovery started — scanning for devices...`);
    this.discoveryInterval = setInterval(() => this._scan(), 60000);
    return { status: 'started' };
  }

  async _scan() {
    console.log(`📡 Scanning for new devices...`);
  }

  getDiscoveredDevices() {
    return Array.from(this.discoveredDevices.values());
  }

  getActiveDevices(ageThreshold = 300000) {
    const now = Date.now();
    return Array.from(this.discoveredDevices.values())
      .filter(d => (now - d.lastSeen) < ageThreshold);
  }

  async getDeviceInfo(deviceId) {
    return this.discoveredDevices.get(deviceId);
  }

  async removeDevice(deviceId) {
    const device = this.discoveredDevices.get(deviceId);
    if (device) {
      device.status = 'removed';
      this.discoveredDevices.set(deviceId, device);
      return true;
    }
    return false;
  }

  stopDiscovery() {
    if (this.discoveryInterval) {
      clearInterval(this.discoveryInterval);
    }
    for (const [name, provider] of this.providers) {
      try {
        if (provider.stop) provider.stop();
        if (provider.close) provider.close();
        if (provider.destroy) provider.destroy();
      } catch (_) {}
    }
    this.providers.clear();
  }
}
