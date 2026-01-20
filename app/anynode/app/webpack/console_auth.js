async function authenticateUser() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    const payload = {
        username: username,
        password: password
    };

    try {
        const response = await fetch('/eden_console/authenticate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (result.success) {
            document.getElementById('message').innerText = 'Authentication successful! Redirecting...';
            window.location.href = '/console-viewer';
        } else {
            document.getElementById('message').innerText = 'Authentication failed. Please try again.';
        }
    } catch (error) {
        console.error('Error during authentication:', error);
        document.getElementById('message').innerText = 'Server error. Please try later.';
    }
}
