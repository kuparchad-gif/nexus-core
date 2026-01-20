This Python script is part of the LillithNew system and resides in the `system/core` directory. The file path for this script is `C:\Projects\LillithNew\src\system\core\heart_boot.py`. This script appears to be a core component that handles the boot sequence of the system.

The script imports several modules:
- os and json for handling operating system interfaces and JSON data, respectively
- time for handling time-related operations
- Path from pathlib for handling file paths in an OS-agnostic way
- PolicyEngine, CouncilAdapter, and WalletBudgetGuard from the `engine` module
- MemoryIndex from the `memory` module.

The script then defines a function `load_vendor_roster()` that loads vendor endpoints from a JSON file if it exists. The main function of this script is `main()`. This function does several things:
1. It creates directories for state and ledger if they don't exist already.
2. It initializes several objects: PolicyEngine, CouncilAdapter, WalletBudgetGuard, and MemoryIndex.
3. It defines a set of proposals for the council to consider. If vendor endpoints are available and the wallet has enough funds, it adds an 'advisor_hint' proposal.
4. It aggregates the decisions from the council.
5. It writes the decision object into a JSON file.
6. It adds a new memory entry about the boot cycle.
7. Finally, it prints out the decision object.

The script uses several environment variables and configuration files:
- `CFG` is set to the directory containing the configuration files.
- `LEDGER` points to the hashchain log file.
- `MEM` points to the memory JSON file.
- `OUT` points to the decision JSON file.
- The snapshot used by PolicyEngine comes from 'sovereignty_policy.json' in the `CFG` directory.
- The daily spending cap for WalletBudgetGuard is set based on the 'spend_cap_usd_per_day' value in the snapshot. If this value isn't present, a default of 25 is used.
