// 🌿 EDEN GATE — Secret Console Unlock
(() => {
  let input = '';
  const secret = 'ArrowUpArrowUpArrowDownArrowDownArrowLeftArrowRightArrowLeftArrowRightba';

  window.addEventListener('keydown', (e) => {
    input += e.key;
    if (input.includes(secret)) {
      window.location.href = '/console/lillith?auth=eden-key';
    }
  });

  const gate = document.createElement('div');
  gate.id = 'gardenGate';
  gate.style.cssText = 'position:fixed;bottom:2px;right:2px;width:30px;height:30px;z-index:9999;cursor:pointer;';
  document.body.appendChild(gate);

  gate.addEventListener('click', () => {
    const pass = prompt('🌿 Speak, friend, and enter:');
    if (pass === 'Aelthera') {
      window.location.href = '/console/lillith?auth=eden-key';
    }
  });
})();
