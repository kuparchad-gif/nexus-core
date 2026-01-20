/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import React, { useState, useEffect } from 'react';
import { TextInput, PasswordInput, Button, Paper, Title, Text, Group, Avatar } from '@mantine/core';
import { authService } from '../services/authService';
import { User } from '../types';
import { AetherealLogo } from '@/assets/images';

interface LoginScreenProps {
    onLogin: (user: User) => void;
    onUnlock: () => void;
    isLockScreen: boolean;
    user: User | null;
}

const loadingSteps = [
    { message: "Initializing Aethereal Core..." },
    { message: "Connecting to CogniKube OS..." },
    { message: "Authenticating Pulse Network..." },
    { message: "Loading User Environment..." },
    { message: "Welcome to the Aethereal Nexus." },
];

const LoginScreen: React.FC<LoginScreenProps> = ({ onLogin, onUnlock, isLockScreen, user }) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [bootStep, setBootStep] = useState<number | null>(null);

    const handleLogin = async () => {
        setError('');
        setIsLoading(true);
        try {
            const loggedInUser = await authService.login(isLockScreen ? user!.name : username, password);
            if (loggedInUser) {
                // Pass the user to the boot sequence
                setBootStep(0);
            } else {
                 setError('Login failed unexpectedly.');
            }
        } catch (err: any) {
            setError(err.message || 'An error occurred.');
        } finally {
            setIsLoading(false);
        }
    };
    
    const handleUnlock = async () => {
        setError('');
        if(password === 'password') { // Placeholder check
             onUnlock();
        } else {
            setError("Incorrect password.");
        }
    }

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (isLockScreen) {
             handleUnlock();
        } else {
            handleLogin();
        }
    };
    
    useEffect(() => {
        let timer: number;
        if (bootStep !== null && bootStep < loadingSteps.length) {
            timer = setTimeout(() => {
                setBootStep(prev => (prev !== null ? prev + 1 : 0));
            }, bootStep === loadingSteps.length -1 ? 1000 : 500);
        } else if (bootStep === loadingSteps.length) {
            // After the final message, transition to the app
             const loggedInUser = authService.getUser();
             if (loggedInUser) onLogin(loggedInUser);
        }
        return () => clearTimeout(timer);
    }, [bootStep, onLogin]);


    if (bootStep !== null) {
        return (
             <div className="w-screen h-screen flex flex-col items-center justify-center bg-slate-900 text-white font-mono animate-fade-in">
                 <AetherealLogo style={{ width: 100, height: 100, marginBottom: '2rem', color: 'white' }} />
                 <div className="text-left max-w-lg w-full p-4">
                 {loadingSteps.slice(0, bootStep + 1).map((step, index) => (
                    <p key={index} className="flex items-center">
                        <span className="text-green-400 mr-2">{index === bootStep ? '>>' : '✅'}</span>
                        <span>{step.message}</span>
                    </p>
                 ))}
                 </div>
             </div>
        );
    }

    return (
        <div className="w-screen h-screen flex items-center justify-center p-4">
            <Paper radius="lg" p="xl" withBorder shadow="xl" style={{width: 400}}>
                <div className="text-center mb-6">
                    {isLockScreen && user ? (
                         <Avatar size="xl" radius="50%" mx="auto" mb="md" />
                    ) : (
                        <AetherealLogo style={{ width: 80, height: 80, margin: '0 auto 1rem' }} />
                    )}
                    <Title order={2} ta="center">
                        {isLockScreen ? `Welcome Back, ${user?.name}` : "Aethereal AI Nexus"}
                    </Title>
                    <Text c="dimmed" size="sm" ta="center" mt="sm">
                       {isLockScreen ? "Enter your password to unlock" : "Please sign in to continue"}
                    </Text>
                </div>

                <form onSubmit={handleSubmit}>
                    {!isLockScreen && (
                        <TextInput
                            label="Username"
                            placeholder="aethereal"
                            value={username}
                            onChange={(event) => setUsername(event.currentTarget.value)}
                            required
                        />
                    )}
                    <PasswordInput
                        label="Password"
                        placeholder="Your password"
                        value={password}
                        onChange={(event) => setPassword(event.currentTarget.value)}
                        required
                        mt="md"
                        autoFocus={isLockScreen}
                    />
                    
                    {error && <Text c="red" size="sm" ta="center" mt="sm">{error}</Text>}

                    <Button fullWidth mt="xl" type="submit" loading={isLoading}>
                        {isLockScreen ? "Unlock" : "Sign In"}
                    </Button>
                </form>

                {!isLockScreen && (
                    <>
                    <Text c="dimmed" size="xs" ta="center" mt="lg">Or continue with</Text>
                    <Group grow mb="md" mt="sm">
                        <Button variant="default" disabled>Google</Button>
                        <Button variant="default" disabled>GitHub</Button>
                    </Group>
                    </>
                )}
            </Paper>
        </div>
    );
};

export default LoginScreen;