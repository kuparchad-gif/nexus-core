// path: frontend/src/lib/api.ts
import axios from 'axios';
import { Cube, HealthStatus, PredictPayload, PredictResponse } from '../types';

const apiClient = axios.create({
  baseURL: '/api', // Proxied by Vite in development
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getCubes = () => apiClient.get<Cube[]>('/cognikubes');

export const getCubeHealth = (cubeName: string) => apiClient.get<HealthStatus>(`/cognikubes/${cubeName}/health`);

// Stubs for start/stop functionality
export const startCube = (cubeName: string) => apiClient.post(`/cognikubes/${cubeName}/start`);
export const stopCube = (cubeName: string) => apiClient.post(`/cognikubes/${cubeName}/stop`);

export const getCubeLogs = (cubeName: string) => apiClient.get<string[]>(`/cognikubes/${cubeName}/logs`);

export const postPredict = (payload: PredictPayload) => apiClient.post<PredictResponse>('/predict', payload);
