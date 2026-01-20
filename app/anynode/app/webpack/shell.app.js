// src/App.js
import React, { useState } from 'react';
import './App.css';

function App() {
    const [prompt, setPrompt] = useState('');
    const [response, setResponse] = useState('');

    const submitPrompt = async () => {
        const res = await fetch('https://aethereal-nexus-viren--viren-cloud-llm-server.modal.run/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });
        const data = await res.json();
        setResponse(data.text);
    };

    return (
        <div className="container mx-auto p-4">
            <h1 className="text-2xl font-bold">Lillith Nexus Dashboard</h1>
            <textarea
                className="w-full p-2 border"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter coding or truth-finding prompt..."
            />
            <button
                className="bg-blue-500 text-white p-2 mt-2"
                onClick={submitPrompt}
            >
                Submit to Lillith
            </button>
            <div className="mt-4">
                <h2 className="text-xl">Response:</h2>
                <p>{response}</p>
            </div>
        </div>
    );
}

export default App;