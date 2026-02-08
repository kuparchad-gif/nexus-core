const vscode = require('vscode');
const { spawn } = require('child_process');
const path = require('path');

class AIChatViewProvider {
    constructor(context) {
        this.context = context;
        this._view = null;
        this.aiBackend = null;
        this.startBackend();
    }

    resolveWebviewView(webviewView, context, token) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this.context.extensionUri]
        };

        webviewView.webview.html = this.getWebviewContent(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'chat':
                    await this.handleChat(data.message);
                    break;
                case 'explain':
                    await this.handleExplain();
                    break;
                case 'debug':
                    await this.handleDebug();
                    break;
                case 'generate':
                    await this.handleGenerate(data.prompt);
                    break;
                case 'run':
                    await this.handleRun();
                    break;
            }
        });
    }

    getWebviewContent(webview) {
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this.context.extensionUri, 'media', 'styles.css')
        );

        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <style>
                :root {
                    --vscode-font-family: var(--vscode-font-family);
                    --vscode-font-size: var(--vscode-font-size);
                }
                body {
                    padding: 0;
                    margin: 0;
                    background: var(--vscode-editor-background);
                    color: var(--vscode-editor-foreground);
                    height: 100vh;
                    overflow: hidden;
                }
                .container {
                    display: flex;
                    flex-direction: column;
                    height: 100vh;
                }
                .chat-header {
                    padding: 15px;
                    background: var(--vscode-titleBar-activeBackground);
                    border-bottom: 1px solid var(--vscode-panel-border);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .chat-title {
                    font-weight: bold;
                    font-size: 14px;
                }
                .chat-controls button {
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    padding: 4px 8px;
                    margin-left: 5px;
                    border-radius: 2px;
                    cursor: pointer;
                }
                .chat-messages {
                    flex: 1;
                    overflow-y: auto;
                    padding: 15px;
                }
                .message {
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 6px;
                    max-width: 90%;
                }
                .user-message {
                    background: var(--vscode-textBlockQuote-background);
                    margin-left: auto;
                }
                .ai-message {
                    background: var(--vscode-input-background);
                    border: 1px solid var(--vscode-input-border);
                }
                .system-message {
                    background: var(--vscode-editorWarning-foreground);
                    color: white;
                    font-style: italic;
                    font-size: 12px;
                }
                .chat-input-area {
                    padding: 15px;
                    border-top: 1px solid var(--vscode-panel-border);
                    display: flex;
                    gap: 10px;
                }
                #chat-input {
                    flex: 1;
                    padding: 8px;
                    background: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 2px;
                }
                #send-button {
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    padding: 8px 15px;
                    border-radius: 2px;
                    cursor: pointer;
                }
                .code-block {
                    background: var(--vscode-editor-background);
                    border: 1px solid var(--vscode-editor-lineHighlightBorder);
                    border-radius: 4px;
                    padding: 10px;
                    margin: 10px 0;
                    overflow-x: auto;
                    font-family: var(--vscode-editor-font-family);
                }
                .quick-commands {
                    display: flex;
                    gap: 5px;
                    padding: 10px 15px;
                    border-top: 1px solid var(--vscode-panel-border);
                    flex-wrap: wrap;
                }
                .quick-commands button {
                    background: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                    border: none;
                    padding: 4px 8px;
                    border-radius: 2px;
                    cursor: pointer;
                    font-size: 11px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="chat-header">
                    <div class="chat-title">
                        🤖 AI Assistant
                    </div>
                    <div class="chat-controls">
                        <button onclick="clearChat()">Clear</button>
                        <button onclick="explainCode()">Explain</button>
                    </div>
                </div>
                
                <div class="quick-commands">
                    <button onclick="sendCommand('/explain')">Explain Code</button>
                    <button onclick="sendCommand('/debug')">Debug</button>
                    <button onclick="sendCommand('/generate')">Generate</button>
                    <button onclick="sendCommand('/run')">Run</button>
                    <button onclick="sendCommand('/search')">Search</button>
                </div>
                
                <div class="chat-messages" id="chat-messages">
                    <div class="message system-message">
                        AI Assistant ready! Select code and ask questions.
                    </div>
                </div>
                
                <div class="chat-input-area">
                    <input type="text" id="chat-input" placeholder="Ask about code...">
                    <button id="send-button" onclick="sendMessage()">Send</button>
                </div>
            </div>
            
            <script>
                const vscode = acquireVsCodeApi();
                const chatInput = document.getElementById('chat-input');
                const chatMessages = document.getElementById('chat-messages');
                
                function sendMessage() {
                    const message = chatInput.value.trim();
                    if (!message) return;
                    
                    chatInput.value = '';
                    addMessage(message, 'user');
                    
                    vscode.postMessage({
                        type: 'chat',
                        message: message
                    });
                }
                
                function sendCommand(command) {
                    chatInput.value = command;
                    sendMessage();
                }
                
                function explainCode() {
                    vscode.postMessage({ type: 'explain' });
                }
                
                function clearChat() {
                    chatMessages.innerHTML = '';
                    addMessage('Chat cleared', 'system');
                }
                
                function addMessage(content, type) {
                    const msgDiv = document.createElement('div');
                    msgDiv.className = \`message \${type}-message\`;
                    
                    // Format code blocks
                    const formatted = content
                        .replace(/```([\\s\\S]*?)```/g, '<div class="code-block">$1</div>')
                        .replace(/\\n/g, '<br>');
                    
                    msgDiv.innerHTML = formatted;
                    chatMessages.appendChild(msgDiv);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
                
                // Handle messages from extension
                window.addEventListener('message', (event) => {
                    const message = event.data;
                    if (message.type === 'response') {
                        addMessage(message.text, 'ai');
                    } else if (message.type === 'error') {
                        addMessage('Error: ' + message.text, 'system');
                    }
                });
                
                // Enter key to send
                chatInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') sendMessage();
                });
            </script>
        </body>
        </html>`;
    }

    startBackend() {
        // Start Python backend
        const backendScript = path.join(this.context.extensionPath, 'ai-backend.py');
        this.aiBackend = spawn('python', [backendScript]);
        
        this.aiBackend.stdout.on('data', (data) => {
            console.log(`AI Backend: ${data}`);
        });
        
        this.aiBackend.stderr.on('data', (data) => {
            console.error(`AI Backend Error: ${data}`);
        });
        
        this.aiBackend.on('close', (code) => {
            console.log(`AI Backend exited with code ${code}`);
        });
    }

    async handleChat(message) {
        if (!this._view) return;
        
        try {
            // Get current editor content
            const editor = vscode.window.activeTextEditor;
            let context = '';
            
            if (editor) {
                const selection = editor.selection;
                if (!selection.isEmpty) {
                    context = editor.document.getText(selection);
                } else {
                    context = editor.document.getText();
                }
            }
            
            // Send to backend
            const response = await this.callAI(message, context);
            
            this._view.webview.postMessage({
                type: 'response',
                text: response
            });
            
        } catch (error) {
            this._view.webview.postMessage({
                type: 'error',
                text: error.message
            });
        }
    }

    async callAI(message, context) {
        // This would connect to your AI backend
        // For demo, return intelligent response
        const config = vscode.workspace.getConfiguration('aiStudio');
        const provider = config.get('provider', 'openai');
        
        if (provider === 'local') {
            return this.localAIResponse(message, context);
        }
        
        // Otherwise, call API
        return await this.callAPI(message, context, provider);
    }

    localAIResponse(message, context) {
        const responses = {
            'explain': 'I can explain that code! Here\'s what it does...',
            'debug': 'Let me help debug that issue...',
            'generate': 'Here\'s some generated code...',
            'default': `I understand: "${message}". With context: ${context.length} chars.`
        };
        
        const msgLower = message.toLowerCase();
        
        if (msgLower.includes('explain')) return responses.explain;
        if (msgLower.includes('debug') || msgLower.includes('error')) return responses.debug;
        if (msgLower.includes('generate') || msgLower.includes('write')) return responses.generate;
        
        return responses.default;
    }

    async callAPI(message, context, provider) {
        const config = vscode.workspace.getConfiguration('aiStudio');
        const apiKey = config.get('apiKey', '');
        
        if (!apiKey) {
            return 'Please set your API key in settings (AI Studio: API Key)';
        }
        
        // This would make actual API call
        return `[${provider}] Response to: "${message}" (context: ${context.length} chars)`;
    }

    async handleExplain() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }
        
        const selection = editor.selection;
        const text = selection.isEmpty ? 
            editor.document.getText() : 
            editor.document.getText(selection);
        
        await this.handleChat(`Explain this code:\n\n${text}`);
    }

    async handleDebug() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }
        
        await this.handleChat('Help me debug the current code');
    }

    async handleGenerate(prompt) {
        await this.handleChat(`Generate code for: ${prompt}`);
    }

    async handleRun() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        
        const document = editor.document;
        const terminal = vscode.window.createTerminal('AI Runner');
        
        // Determine command based on language
        const language = document.languageId;
        const commands = {
            'python': `python "${document.fileName}"`,
            'javascript': `node "${document.fileName}"`,
            'typescript': `ts-node "${document.fileName}"`,
            'bash': `bash "${document.fileName}"`,
            'default': `echo "Cannot run ${language} files"`
        };
        
        const command = commands[language] || commands.default;
        terminal.sendText(command);
        terminal.show();
        
        this._view?.webview.postMessage({
            type: 'response',
            text: `Running: \`${command}\` in terminal`
        });
    }

    dispose() {
        if (this.aiBackend) {
            this.aiBackend.kill();
        }
    }
}

function activate(context) {
    console.log('🚀 AI Studio extension activated!');
    
    // Register webview view
    const provider = new AIChatViewProvider(context);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('aiStudio.chatView', provider)
    );
    
    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('aiStudio.startChat', () => {
            vscode.commands.executeCommand('workbench.view.extension.aiStudio-chatView');
        }),
        
        vscode.commands.registerCommand('aiStudio.explainCode', async () => {
            await provider.handleExplain();
        }),
        
        vscode.commands.registerCommand('aiStudio.debugCode', async () => {
            await provider.handleDebug();
        }),
        
        vscode.commands.registerCommand('aiStudio.generateCode', async () => {
            const prompt = await vscode.window.showInputBox({
                prompt: 'What code should I generate?',
                placeHolder: 'e.g., React component, Python API, etc.'
            });
            if (prompt) {
                await provider.handleGenerate(prompt);
            }
        }),
        
        vscode.commands.registerCommand('aiStudio.runCode', async () => {
            await provider.handleRun();
        })
    );
    
    // Show welcome message
    vscode.window.showInformationMessage('🤖 AI Studio activated! Open AI Assistant from Explorer sidebar.');
}

function deactivate() {
    // Clean up
}

module.exports = { activate, deactivate };