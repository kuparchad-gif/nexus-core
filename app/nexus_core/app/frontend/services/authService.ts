/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { User } from '../types';

const SESSION_KEY = 'aethereal-nexus-session';

const defaultUser: User = {
    id: 'user-01',
    name: 'Aethereal',
    role: 'admin',
};

export const authService = {
    login: async (username: string, password: string): Promise<User | null> => {
        // Simulate an API call
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                if (username === 'aethereal' && password === 'password') {
                    sessionStorage.setItem(SESSION_KEY, JSON.stringify(defaultUser));
                    resolve(defaultUser);
                } else {
                    reject(new Error('Invalid username or password'));
                }
            }, 500);
        });
    },

    logout: (): void => {
        sessionStorage.removeItem(SESSION_KEY);
    },

    checkSession: (): User | null => {
        try {
            const sessionData = sessionStorage.getItem(SESSION_KEY);
            if (sessionData) {
                return JSON.parse(sessionData) as User;
            }
            return null;
        } catch (error) {
            console.error("Failed to parse session data", error);
            return null;
        }
    },

    getUser: (): User | null => {
        return authService.checkSession();
    },
};
