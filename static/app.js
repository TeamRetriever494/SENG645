// Register the service worker
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('service-worker.js')
    .then(reg => console.log('Service Worker registered ✅', reg))
    .catch(err => console.error('Service Worker registration failed ❌', err));
}

// Handle PWA install prompt
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (e) => {
  // Prevent the mini-infobar from appearing on mobile
  e.preventDefault();
  // Stash the event so it can be triggered later
  deferredPrompt = e;

  // Create a custom install button
  const installBtn = document.createElement('button');
  installBtn.textContent = 'Install App';
  installBtn.style.position = 'fixed';
  installBtn.style.bottom = '20px';
  installBtn.style.right = '20px';
  installBtn.style.padding = '10px 15px';
  installBtn.style.fontSize = '16px';
  installBtn.style.zIndex = '9999';
  document.body.appendChild(installBtn);

  installBtn.addEventListener('click', () => {
    // Show the install prompt
    deferredPrompt.prompt();

    // Wait for the user's response
    deferredPrompt.userChoice.then(choice => {
      if (choice.outcome === 'accepted') {
        console.log('User accepted the install prompt ✅');
      } else {
        console.log('User dismissed the install prompt ❌');
      }
      deferredPrompt = null;
      installBtn.remove();
    });
  });
});
  