// Heal Ayur - Utility Functions
// Global utility functions and helpers

// Notification System
function showNotification(message, type = 'info', duration = 4000) {
  const notification = document.createElement('div');
  notification.className = `notification-toast ${type} animate-slideInRight`;
  
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️';
  notification.innerHTML = `
    <div style="display: flex; align-items: center; gap: 0.5rem;">
      <span style="font-size: 1.2rem;">${icon}</span>
      <span>${message}</span>
      <button onclick="this.parentElement.parentElement.remove()" style="margin-left: auto; background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 1.2rem;">&times;</button>
    </div>
  `;
  
  document.body.appendChild(notification);
  
  setTimeout(() => {
    if (notification.parentElement) {
      notification.style.animation = 'slideOutRight 0.3s ease-out';
      setTimeout(() => notification.remove(), 300);
    }
  }, duration);
}

// Background Animation System
function initializeBackground() {
  const canvas = document.getElementById('bgCanvas');
  if (!canvas) return;
  
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x0a0a0a, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
  camera.position.z = 8;

  // Create multiple geometric shapes with different materials
  const geometries = [
    new THREE.IcosahedronGeometry(1, 0),
    new THREE.OctahedronGeometry(1, 0),
    new THREE.TetrahedronGeometry(1, 0)
  ];

  const materials = [
    new THREE.MeshStandardMaterial({ color: 0x4caf50, wireframe: true, opacity: 0.2, transparent: true }),
    new THREE.MeshStandardMaterial({ color: 0x81c784, wireframe: true, opacity: 0.15, transparent: true }),
    new THREE.MeshStandardMaterial({ color: 0xc8e6c9, wireframe: true, opacity: 0.1, transparent: true })
  ];

  const shapes = [];
  for(let i = 0; i < 20; i++) {
    const geometry = geometries[Math.floor(Math.random() * geometries.length)];
    const material = materials[Math.floor(Math.random() * materials.length)];
    const mesh = new THREE.Mesh(geometry, material);
    
    mesh.position.set(
      (Math.random() - 0.5) * 30,
      (Math.random() - 0.5) * 30,
      (Math.random() - 0.5) * 30
    );
    
    mesh.rotation.set(
      Math.random() * Math.PI * 2,
      Math.random() * Math.PI * 2,
      Math.random() * Math.PI * 2
    );
    
    mesh.scale.setScalar(Math.random() * 0.8 + 0.2);
    mesh.userData = {
      rotationSpeed: {
        x: (Math.random() - 0.5) * 0.01,
        y: (Math.random() - 0.5) * 0.01,
        z: (Math.random() - 0.5) * 0.01
      }
    };
    
    scene.add(mesh);
    shapes.push(mesh);
  }

  // Enhanced lighting
  const ambientLight = new THREE.AmbientLight(0x4caf50, 0.3);
  scene.add(ambientLight);
  
  const pointLight1 = new THREE.PointLight(0x4caf50, 0.8, 100);
  pointLight1.position.set(10, 10, 10);
  scene.add(pointLight1);
  
  const pointLight2 = new THREE.PointLight(0x81c784, 0.6, 100);
  pointLight2.position.set(-10, -10, 5);
  scene.add(pointLight2);

  // Animate shapes with individual rotation speeds
  function animate() {
    requestAnimationFrame(animate);
    
    shapes.forEach(shape => {
      shape.rotation.x += shape.userData.rotationSpeed.x;
      shape.rotation.y += shape.userData.rotationSpeed.y;
      shape.rotation.z += shape.userData.rotationSpeed.z;
    });
    
    // Slowly rotate the camera
    camera.position.x = Math.sin(Date.now() * 0.0005) * 2;
    camera.position.y = Math.cos(Date.now() * 0.0003) * 1;
    camera.lookAt(scene.position);
    
    renderer.render(scene, camera);
  }
  animate();

  // Responsive resize handler
  function handleResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }
  
  window.addEventListener('resize', handleResize);
  
  // Pause animation when page is not visible (performance optimization)
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      renderer.setAnimationLoop(null);
    } else {
      renderer.setAnimationLoop(animate);
    }
  });
}

// Particle System
function createParticles() {
  const container = document.getElementById('particlesContainer');
  if (!container) return;
  
  const particleCount = 50;
  
  for (let i = 0; i < particleCount; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    const size = Math.random() * 4 + 2;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    particle.style.left = Math.random() * 100 + '%';
    particle.style.top = Math.random() * 100 + '%';
    particle.style.animationDelay = Math.random() * 6 + 's';
    particle.style.animationDuration = (Math.random() * 4 + 4) + 's';
    
    container.appendChild(particle);
  }
}

// Form Validation Utilities
function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

function validatePassword(password) {
  return {
    length: password.length >= 8,
    uppercase: /[A-Z]/.test(password),
    lowercase: /[a-z]/.test(password),
    number: /[0-9]/.test(password),
    special: /[^A-Za-z0-9]/.test(password)
  };
}

function calculatePasswordStrength(password) {
  const checks = validatePassword(password);
  return Object.values(checks).filter(Boolean).length;
}

// Local Storage Utilities
function saveToStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.error('Error saving to localStorage:', error);
    return false;
  }
}

function loadFromStorage(key, defaultValue = null) {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  } catch (error) {
    console.error('Error loading from localStorage:', error);
    return defaultValue;
  }
}

function removeFromStorage(key) {
  try {
    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error('Error removing from localStorage:', error);
    return false;
  }
}

// API Utilities
async function apiRequest(url, options = {}) {
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json'
    }
  };
  
  const mergedOptions = { ...defaultOptions, ...options };
  
  // Add auth token if available
  const token = loadFromStorage('session_token');
  if (token) {
    mergedOptions.headers.Authorization = `Bearer ${token}`;
  }
  
  try {
    const response = await fetch(url, mergedOptions);
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || `HTTP error! status: ${response.status}`);
    }
    
    return data;
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}

// Date and Time Utilities
function formatDate(date, options = {}) {
  const defaultOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  };
  
  return new Date(date).toLocaleDateString('en-US', { ...defaultOptions, ...options });
}

function formatTime(date) {
  return new Date(date).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit'
  });
}

function timeAgo(date) {
  const now = new Date();
  const past = new Date(date);
  const diffInSeconds = Math.floor((now - past) / 1000);
  
  if (diffInSeconds < 60) return 'just now';
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
  if (diffInSeconds < 2592000) return `${Math.floor(diffInSeconds / 86400)}d ago`;
  
  return formatDate(date);
}

// Performance Utilities
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

function throttle(func, limit) {
  let inThrottle;
  return function() {
    const args = arguments;
    const context = this;
    if (!inThrottle) {
      func.apply(context, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

// Device Detection
function isMobile() {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

function isTablet() {
  return /iPad|Android/i.test(navigator.userAgent) && window.innerWidth >= 768;
}

function isDesktop() {
  return !isMobile() && !isTablet();
}

// Animation Utilities
function animateValue(start, end, duration, callback) {
  const startTime = performance.now();
  
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    const value = start + (end - start) * easeOutCubic(progress);
    callback(value);
    
    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }
  
  requestAnimationFrame(update);
}

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

// Error Handling
function handleError(error, context = '') {
  console.error(`Error in ${context}:`, error);
  
  // Show user-friendly error message
  const message = error.message || 'An unexpected error occurred';
  showNotification(`❌ ${message}`, 'error');
  
  // Log to analytics (if available)
  if (window.analytics && window.analytics.track) {
    window.analytics.track('error', {
      message: error.message,
      context,
      stack: error.stack
    });
  }
}

// Initialize utilities when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Set up global error handling
  window.addEventListener('error', (event) => {
    handleError(event.error, 'Global');
  });
  
  window.addEventListener('unhandledrejection', (event) => {
    handleError(event.reason, 'Promise');
  });
  
  // Initialize performance monitoring
  if ('performance' in window) {
    window.addEventListener('load', () => {
      setTimeout(() => {
        const perfData = performance.getEntriesByType('navigation')[0];
        console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
      }, 0);
    });
  }
});

// Export utilities for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    showNotification,
    validateEmail,
    validatePassword,
    calculatePasswordStrength,
    saveToStorage,
    loadFromStorage,
    removeFromStorage,
    apiRequest,
    formatDate,
    formatTime,
    timeAgo,
    debounce,
    throttle,
    isMobile,
    isTablet,
    isDesktop,
    animateValue,
    handleError
  };
}
