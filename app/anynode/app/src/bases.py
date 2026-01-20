
Understood your request and have refactored the provided code. The changes made are as follows:

1. The absolute Windows file path has been replaced with a dynamically calculated path using `os`, which makes it more portable across different operating systems and environments.
2. I kept the `sys.path.insert()` function call, which ensures that the parent directory of the script is added to the system path, allowing for relative imports.
3. I have removed any references or usage of ROOT variables since they are not present in your provided code snippet. If you were using such variables for specifying file paths, I would recommend updating them to use relative paths instead.
4. The logging configuration remains the same as no changes were needed in this section.
