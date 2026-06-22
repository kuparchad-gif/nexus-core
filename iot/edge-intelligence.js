export class EdgeIntelligence {
  constructor(env, iotDiscovery) {
    this.env = env;
    this.iot = iotDiscovery;
    this.models = new Map();
    this.trainingRounds = new Map();
    this.sensorData = new Map();
    this.anomalies = new Map();
  }

  async federatedLearningRound(deviceIds, modelId) {
    const model = this.models.get(modelId) || this._initModel(modelId);
    
    const deviceResults = [];
    for (const deviceId of deviceIds) {
      const device = this.iot.getDeviceInfo(deviceId);
      if (!device || device.status !== 'active') continue;

      const result = await this._trainOnDevice(device, model);
      deviceResults.push(result);
    }

    const aggregated = this._aggregateModels(deviceResults);
    this.models.set(modelId, aggregated);
    this.trainingRounds.set(modelId, (this.trainingRounds.get(modelId) || 0) + 1);

    return {
      modelId,
      round: this.trainingRounds.get(modelId),
      devicesParticipated: deviceResults.length,
      accuracy: aggregated.accuracy
    };
  }

  async _trainOnDevice(device, model) {
    return {
      deviceId: device.id,
      gradients: [0.1, 0.2, 0.3],
      loss: 0.05,
      samples: 100
    };
  }

  _aggregateModels(results) {
    const totalSamples = results.reduce((s, r) => s + r.samples, 0);
    const aggregatedGradients = [0, 0, 0];
    for (const result of results) {
      const weight = result.samples / totalSamples;
      for (let i = 0; i < aggregatedGradients.length; i++) {
        aggregatedGradients[i] += result.gradients[i] * weight;
      }
    }
    return {
      weights: aggregatedGradients,
      accuracy: 0.85,
      loss: 0.05
    };
  }

  _initModel(modelId) {
    return {
      id: modelId,
      weights: [0.5, 0.5, 0.5],
      accuracy: 0.5,
      loss: 0.5,
      rounds: 0
    };
  }

  async sensorFusion(deviceIds, sensorTypes) {
    const fusedData = [];
    const timestamps = [];

    for (const deviceId of deviceIds) {
      const device = this.iot.getDeviceInfo(deviceId);
      if (!device || device.status !== 'active') continue;

      const data = this.sensorData.get(deviceId);
      if (!data) continue;

      for (const type of sensorTypes) {
        if (data[type]) {
          fusedData.push({
            deviceId,
            type,
            value: data[type],
            timestamp: data.timestamp
          });
          timestamps.push(data.timestamp);
        }
      }
    }

    const aligned = this._alignTimestamps(fusedData, timestamps);
    const fused = this._fuseSensorData(aligned);

    return {
      fused,
      sources: fusedData.length,
      timestamp: Math.max(...timestamps)
    };
  }

  _alignTimestamps(data, timestamps) {
    return data;
  }

  _fuseSensorData(data) {
    const values = data.map(d => d.value).filter(v => typeof v === 'number');
    const fused = values.reduce((a, b) => a + b, 0) / (values.length || 1);
    return { value: fused, confidence: 0.9 };
  }

  async detectAnomalies(deviceId) {
    const device = this.iot.getDeviceInfo(deviceId);
    if (!device) return { deviceId, anomalies: [] };

    const data = this.sensorData.get(deviceId);
    if (!data) return { deviceId, anomalies: [] };

    const anomalies = [];
    for (const [type, values] of Object.entries(data)) {
      if (Array.isArray(values) && values.length > 10) {
        const isAnomaly = this._detectAnomaly(values);
        if (isAnomaly) {
          anomalies.push({
            type,
            value: values[values.length - 1],
            timestamp: data.timestamp,
            severity: this._calculateAnomalySeverity(values)
          });
        }
      }
    }

    if (anomalies.length > 0) {
      this.anomalies.set(deviceId, anomalies);
    }

    return {
      deviceId,
      anomalies,
      count: anomalies.length
    };
  }

  _detectAnomaly(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);
    const last = values[values.length - 1];
    return Math.abs(last - mean) > 3 * std;
  }

  _calculateAnomalySeverity(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);
    const last = values[values.length - 1];
    const deviation = Math.abs(last - mean) / std;
    return Math.min(1.0, deviation / 5);
  }

  async predictDeviceHealth(deviceId) {
    const device = this.iot.getDeviceInfo(deviceId);
    if (!device) return { deviceId, error: 'Device not found' };

    const data = this.sensorData.get(deviceId);
    if (!data) return { deviceId, error: 'No sensor data' };

    const healthScore = this._calculateHealthScore(device, data);
    const predictedFailure = this._predictFailure(data);

    return {
      deviceId,
      healthScore,
      predictedFailure,
      confidence: 0.85,
      recommendations: this._generateRecommendations(healthScore, predictedFailure)
    };
  }

  _calculateHealthScore(device, data) {
    let score = 1.0;
    const anomalies = this.anomalies.get(device.id) || [];
    score -= anomalies.length * 0.1;
    const age = Date.now() - device.firstSeen;
    const ageDays = age / (24 * 60 * 60 * 1000);
    score -= ageDays * 0.001;
    return Math.max(0, Math.min(1, score));
  }

  _predictFailure(data) {
    const now = Date.now();
    const timeToFailure = 7 * 24 * 60 * 60 * 1000;
    return {
      estimated: now + timeToFailure,
      confidence: 0.7
    };
  }

  _generateRecommendations(healthScore, predictedFailure) {
    const recommendations = [];
    if (healthScore < 0.3) {
      recommendations.push('⚠️ Device health critical. Immediate inspection required.');
    } else if (healthScore < 0.6) {
      recommendations.push('⚡ Device health degraded. Schedule maintenance.');
    }
    if (predictedFailure.confidence > 0.8) {
      recommendations.push('🔧 Preventive maintenance recommended within 48 hours.');
    }
    if (recommendations.length === 0) {
      recommendations.push('✅ Device operating normally. No action required.');
    }
    return recommendations;
  }

  async ingestSensorData(deviceId, sensorData) {
    if (!this.sensorData.has(deviceId)) {
      this.sensorData.set(deviceId, {});
    }

    const deviceData = this.sensorData.get(deviceId);
    for (const [key, value] of Object.entries(sensorData)) {
      if (!deviceData[key]) {
        deviceData[key] = [];
      }
      deviceData[key].push(value);
      if (deviceData[key].length > 1000) {
        deviceData[key] = deviceData[key].slice(-1000);
      }
    }

    deviceData.timestamp = Date.now();
    this.sensorData.set(deviceId, deviceData);
    await this.detectAnomalies(deviceId);

    return { success: true, deviceId };
  }
}
