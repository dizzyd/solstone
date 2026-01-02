// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 sol pbc

/**
 * Error Handler for App System
 * Captures JavaScript errors and provides visual feedback
 *
 * Features:
 * - Catches window errors and unhandled promise rejections
 * - Updates status icon to red with red glow on error
 * - Displays error log at bottom of viewport
 * - Provides modal for manual error display via window.showError()
 */

(function(){
  const statusIcon = document.querySelector('.facet-bar .status-icon');
  const errorLog = document.getElementById('error-log');
  const errorModal = document.getElementById('errorModal');
  const errorMessage = document.getElementById('errorMessage');
  const closeButton = errorModal ? errorModal.querySelector('.close') : null;

  // Escape HTML to prevent XSS
  function escapeHtml(text) {
    return String(text).replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // Log error to bottom panel
  function logError(text) {
    if (errorLog) {
      errorLog.insertAdjacentHTML(
        'beforeend',
        escapeHtml(text) + '<br>'
      );
      errorLog.style.display = 'block';
    }
  }

  // Mark status icon as error state (red with glow)
  function markError() {
    if (statusIcon) {
      statusIcon.textContent = '🔴';
      statusIcon.classList.add('error');
    }
  }

  // Global error handler
  window.addEventListener('error', (e) => {
    markError();
    logError(`❌ ${e.message} @ ${e.filename}:${e.lineno}`);
  });

  // Unhandled promise rejection handler
  window.addEventListener('unhandledrejection', (e) => {
    markError();
    logError(`⚠️ Promise: ${e.reason}`);
  });

  // Modal controls
  if (errorModal && closeButton && errorMessage) {
    // Provide global function for manual error display
    window.showError = (text) => {
      errorMessage.textContent = text;
      errorModal.style.display = 'block';
    };

    // Close button
    closeButton.onclick = () => {
      errorModal.style.display = 'none';
    };

    // Click outside to close
    window.addEventListener('click', (e) => {
      if (e.target === errorModal) {
        errorModal.style.display = 'none';
      }
    });
  }
})();
