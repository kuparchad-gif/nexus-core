// ts_router.ts
// Purpose: Expose frontend logic callable from Python bridge

export const tsRouter = {
  reloadUI: () => {
    console.log("[TS Router] Reloading UI...");
    // Example: trigger Vue/React/HTML component refresh
  },
  alertStatus: (msg: string) => {
    alert(`[TS Router] ${msg}`);
  }
};

// Example call (optional testing)
if (typeof window !== "undefined") {
  // Simulate receiving a message from Python
  tsRouter.alertStatus("Connected to Python bridge!");
}
