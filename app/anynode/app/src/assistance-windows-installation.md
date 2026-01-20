# Installing `assistancechat/assistance` on Windows

This guide helps you quickly set up the [assistancechat/assistance](https://github.com/assistancechat/assistance) Python AI assistant on Windows.  
You do **not** need to use WASM or advanced server tooling unless you want to, but some Unix-specific features may be unavailable on Windows (but basic assistant/chat features will work).

---

## 1. Install Python

- Download and install [Python 3.9+](https://www.python.org/downloads/windows/).
- When installing, **check "Add Python to PATH"**.

## 2. Install Required Tools

- [Git for Windows](https://gitforwindows.org/) (for cloning the repo or getting updates).

## 3. Clone the Repository

Open **Command Prompt** or **PowerShell** and run:
```sh
git clone https://github.com/assistancechat/assistance.git
cd assistance
```

## 4. Set Up a Virtual Environment (Recommended)

```sh
python -m venv venv
venv\Scripts\activate
```

## 5. Install Python Dependencies

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Basic Usage

The project is modular and advanced (it can run as a library, web service, or WASM sandbox), but you can start directly with Python!

> The main entry points are typically within the `assistance/` directory.
>
> To test a basic Python session:
```sh
python
```
And in the Python shell:
```python
from assistance.assistance import Assistant
a = Assistant()
print(a("What is Python?"))
```

Check for scripts like `main.py` or provided CLI/web interfaces for richer experiences.

## 7. (Advanced: WASM/Unix-Sandbox)  
- The README talks about WASM with `wasmtime`, which is mostly for advanced sandboxing or running on Linux/UNIX. You can skip this on Windows unless you have a specific use-case.

## 8. (Optional Advanced)  
If you need a production service (server hosting, etc.), you may want to use WSL2 (Windows Subsystem for Linux) to get full POSIX compatibility.

---

## Troubleshooting

- If you see errors about missing C++ build tools, install the "Desktop development with C++" workload via [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
- For questions on features or configuring the assistant beyond basic Python, open an [issue on GitHub](https://github.com/assistancechat/assistance/issues).

---

## Summary

- **You can run and develop with `assistance` on Windows!**
- Use Python, a virtualenv, and pip as shown above.
- Ignore WASM/supervisor/nginx unless you really need them.
- For IDE/chatbot integration or project folder access, extend the Assistant in your Python code.

---

**Happy hacking! If you need a more hands-on step (or a custom entry script), let me know what you want to do with it!**