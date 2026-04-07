(() => {
  if (typeof window === 'undefined' || window.__tilletiaAuthClientInstalled) {
    return;
  }
  window.__tilletiaAuthClientInstalled = true;

  const LOGIN_PATH = '/login';
  let redirecting = false;

  function isLoginPage() {
    return window.location.pathname === LOGIN_PATH;
  }

  function redirectToLogin() {
    if (redirecting || isLoginPage()) {
      return;
    }
    redirecting = true;
    window.location.assign(LOGIN_PATH);
  }

  function isUnauthorizedResponse(response) {
    if (!response) {
      return false;
    }
    if (response.status === 401) {
      return true;
    }
    if (!response.redirected || !response.url) {
      return false;
    }
    try {
      const responseUrl = new URL(response.url, window.location.origin);
      return responseUrl.pathname === LOGIN_PATH;
    } catch (error) {
      return false;
    }
  }

  if (typeof window.fetch === 'function') {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
      const response = await nativeFetch(...args);
      if (isUnauthorizedResponse(response)) {
        redirectToLogin();
      }
      return response;
    };
  }

  const nativeOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (...args) {
    this.addEventListener('load', () => {
      if (this.status === 401) {
        redirectToLogin();
      }
    });
    return nativeOpen.apply(this, args);
  };
})();
