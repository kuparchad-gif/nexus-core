async function verifyCode() {
  const input = document.getElementById('codeInput').value;
  const message = document.getElementById('statusMessage');

  // Replace this with your backend endpoint or verification logic
  const expected = "123456"; // TEMP: Replace with server-generated TOTP validator

  if (input === expected) {
    message.style.color = 'green';
    message.textContent = "Access granted. Welcome.";
    setTimeout(() => {
      window.location.href = "/Frontend/mailbox/dashboard.html"; // future secured view
    }, 1000);
  } else {
    message.style.color = 'red';
    message.textContent = "Invalid code. Try again.";
  }
}
