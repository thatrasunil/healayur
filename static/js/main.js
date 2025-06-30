// Heal Ayur - Enhanced Main Application JavaScript
class HealAyurApp {
  constructor() {
    this.initializeElements();
    this.initializeEventListeners();
    this.initializeState();
    this.initializeAdvancedFeatures();
    this.loadingMessages = [
      "🔍 Analyzing your image with AI...",
      "🧠 Processing with advanced algorithms...",
      "🌿 Finding personalized remedies...",
      "📊 Calculating confidence scores...",
      "✨ Preparing results...",
      "🎯 Almost ready!"
    ];
    this.currentLoadingIndex = 0;
    this.retryCount = 0;
    this.maxRetries = 3;
  }

  initializeElements() {
    // Upload methods
    this.fileUploadMethod = document.getElementById('fileUploadMethod');
    this.webcamMethod = document.getElementById('webcamMethod');
    this.imageInput = document.getElementById('imageInput');
    
    // Webcam elements
    this.webcamContainer = document.getElementById('webcamContainer');
    this.videoElement = document.getElementById('videoElement');
    this.captureBtn = document.getElementById('captureBtn');
    this.stopWebcamBtn = document.getElementById('stopWebcamBtn');
    this.toggleRealTimeBtn = document.getElementById('toggleRealTimeBtn');
    
    // Action buttons
    this.analyzeBtn = document.getElementById('analyzeBtn');
    
    // Loading and results
    this.loadingContainer = document.getElementById('loadingContainer');
    this.loadingText = document.getElementById('loadingText');
    this.resultsContainer = document.getElementById('resultsContainer');
    
    // Stats
    this.totalAnalysesEl = document.getElementById('totalAnalyses');
  }

  initializeEventListeners() {
    // File upload
    if (this.imageInput) {
      this.imageInput.addEventListener('change', (e) => this.handleFileSelect(e));
    }
    if (this.fileUploadMethod) {
      this.fileUploadMethod.addEventListener('click', () => this.imageInput?.click());
    }
    
    // Webcam controls
    if (this.webcamMethod) {
      this.webcamMethod.addEventListener('click', () => this.toggleWebcam());
    }
    if (this.captureBtn) {
      this.captureBtn.addEventListener('click', () => this.captureAndAnalyze());
    }
    if (this.stopWebcamBtn) {
      this.stopWebcamBtn.addEventListener('click', () => this.stopWebcam());
    }
    if (this.toggleRealTimeBtn) {
      this.toggleRealTimeBtn.addEventListener('click', () => this.toggleRealTimeAnalysis());
    }
    
    // Analysis
    if (this.analyzeBtn) {
      this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
    }
    
    // Drag and drop
    this.setupDragAndDrop();
  }

  initializeState() {
    this.webcamActive = false;
    this.stream = null;
    this.selectedFile = null;
    this.realTimeActive = false;
    this.realTimeCount = 0;
    this.realTimeStartTime = null;
    this.continuousAnalysisInterval = null;
    this.isAnalyzing = false;
    this.analysisHistory = [];
    this.currentUser = null;
    this.offlineMode = false;
  }

  initializeAdvancedFeatures() {
    // Initialize PWA features
    this.initializePWA();

    // Initialize voice commands
    this.initializeVoiceCommands();

    // Initialize gesture controls
    this.initializeGestureControls();

    // Initialize performance monitoring
    this.initializePerformanceMonitoring();

    // Check for user authentication
    this.checkUserAuthentication();

    // Initialize offline support
    this.initializeOfflineSupport();
  }

  setupDragAndDrop() {
    if (!this.fileUploadMethod) return;
    
    const dropZone = this.fileUploadMethod;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => {
        dropZone.classList.add('active');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => {
        dropZone.classList.remove('active');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        this.handleFileSelect({ target: { files } });
      }
    });
  }

  handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      this.showError('Please select a valid image file (JPEG, PNG, GIF, BMP, or WebP)');
      return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
      this.showError('File size must be less than 16MB');
      return;
    }

    this.selectedFile = file;
    if (this.fileUploadMethod) {
      this.fileUploadMethod.classList.add('active');
    }
    if (this.analyzeBtn) {
      this.analyzeBtn.style.display = 'inline-flex';
    }
    
    // Update UI to show selected file
    const fileName = file.name.length > 20 ? file.name.substring(0, 20) + '...' : file.name;
    const fileText = this.fileUploadMethod?.querySelector('p');
    if (fileText) {
      fileText.textContent = `Selected: ${fileName}`;
    }
  }

  async toggleWebcam() {
    console.log('🔍 Toggle webcam called, current state:', this.webcamActive);

    if (!this.webcamActive) {
      try {
        console.log('📷 Requesting camera access...');

        this.stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          }
        });

        console.log('✅ Camera access granted, stream:', this.stream);

        if (this.videoElement) {
          this.videoElement.srcObject = this.stream;

          // Wait for video to load
          await new Promise((resolve) => {
            this.videoElement.onloadedmetadata = () => {
              console.log('✅ Video metadata loaded');
              console.log('📊 Video dimensions:', this.videoElement.videoWidth, 'x', this.videoElement.videoHeight);
              resolve();
            };
          });

          // Additional wait to ensure video is fully ready
          setTimeout(() => {
            console.log('📊 Final video check - Width:', this.videoElement.videoWidth, 'Height:', this.videoElement.videoHeight);
          }, 1000);
        }

        if (this.webcamContainer) {
          this.webcamContainer.style.display = 'block';
        }
        if (this.webcamMethod) {
          this.webcamMethod.classList.add('active');
        }
        this.webcamActive = true;

        // Hide other elements
        if (this.fileUploadMethod) {
          this.fileUploadMethod.style.opacity = '0.5';
        }
        if (this.analyzeBtn) {
          this.analyzeBtn.style.display = 'none';
        }
        if (this.resultsContainer) {
          this.resultsContainer.style.display = 'none';
        }

        // Show notification about real-time option
        showNotification('📷 Camera ready! Click "Capture" or "Start Real-Time" for analysis', 'info', 4000);

      } catch (err) {
        console.error('❌ Camera access error:', err);
        this.showError('Could not access camera: ' + err.message);
      }
    }
  }

  stopWebcam() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    
    // Stop continuous analysis
    if (this.continuousAnalysisInterval) {
      clearInterval(this.continuousAnalysisInterval);
      this.continuousAnalysisInterval = null;
    }
    
    // Remove real-time overlay
    const overlay = document.getElementById('realTimeOverlay');
    if (overlay) {
      overlay.remove();
    }
    
    if (this.webcamContainer) {
      this.webcamContainer.style.display = 'none';
    }
    if (this.webcamMethod) {
      this.webcamMethod.classList.remove('active');
    }
    this.webcamActive = false;
    this.realTimeActive = false;
    
    // Restore other elements
    if (this.fileUploadMethod) {
      this.fileUploadMethod.style.opacity = '1';
    }
    if (this.resultsContainer) {
      this.resultsContainer.style.display = 'none';
    }
    
    showNotification('🔴 Camera stopped', 'info', 2000);
  }

  async captureAndAnalyze() {
    console.log('🔍 Capture and analyze called');
    console.log('📊 Webcam active:', this.webcamActive);
    console.log('📊 Video element:', this.videoElement);
    console.log('📊 Video width:', this.videoElement?.videoWidth);
    console.log('📊 Video height:', this.videoElement?.videoHeight);

    if (!this.webcamActive) {
      this.showError('Camera is not active. Please start the camera first.');
      return;
    }

    if (!this.videoElement) {
      this.showError('Video element not found. Please refresh the page and try again.');
      return;
    }

    if (!this.videoElement.videoWidth || this.videoElement.videoWidth === 0) {
      this.showError('Camera not ready. Please wait a moment for the camera to initialize and try again.');
      return;
    }

    try {
      // Create canvas to capture frame
      const canvas = document.createElement('canvas');
      canvas.width = this.videoElement.videoWidth;
      canvas.height = this.videoElement.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(this.videoElement, 0, 0);

      // Convert to base64
      const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
      const base64Image = dataUrl.split(',')[1];

      console.log('✅ Image captured successfully, size:', base64Image.length);

      // Stop webcam and analyze
      this.stopWebcam();
      await this.analyzeImageData(base64Image, true);
    } catch (error) {
      console.error('❌ Error during capture:', error);
      this.showError('Failed to capture image: ' + error.message);
    }
  }

  // Advanced Features Implementation
  initializePWA() {
    // Check if PWA is installable
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      this.deferredPrompt = e;
      this.showPWAInstallPrompt();
    });

    // Register service worker
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/static/sw.js')
        .then(registration => {
          console.log('✅ Service Worker registered:', registration);
        })
        .catch(error => {
          console.log('❌ Service Worker registration failed:', error);
        });
    }
  }

  initializeVoiceCommands() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = false;
      this.recognition.interimResults = false;
      this.recognition.lang = 'en-US';

      this.recognition.onresult = (event) => {
        const command = event.results[0][0].transcript.toLowerCase();
        this.processVoiceCommand(command);
      };

      this.recognition.onerror = (event) => {
        console.error('Voice recognition error:', event.error);
      };
    }
  }

  initializeGestureControls() {
    let startX, startY;

    document.addEventListener('touchstart', (e) => {
      startX = e.touches[0].clientX;
      startY = e.touches[0].clientY;
    });

    document.addEventListener('touchend', (e) => {
      if (!startX || !startY) return;

      const endX = e.changedTouches[0].clientX;
      const endY = e.changedTouches[0].clientY;

      const diffX = startX - endX;
      const diffY = startY - endY;

      if (Math.abs(diffX) > Math.abs(diffY)) {
        if (diffX > 50) {
          // Swipe left - open chat
          if (window.chatBot) window.chatBot.toggleChat();
        } else if (diffX < -50) {
          // Swipe right - close chat or go back
          if (window.chatBot && window.chatBot.isOpen) {
            window.chatBot.toggleChat();
          }
        }
      }
    });
  }

  initializePerformanceMonitoring() {
    // Monitor page load performance
    window.addEventListener('load', () => {
      const perfData = performance.getEntriesByType('navigation')[0];
      console.log('📊 Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
    });

    // Monitor memory usage (if available)
    if ('memory' in performance) {
      setInterval(() => {
        const memory = performance.memory;
        if (memory.usedJSHeapSize > memory.jsHeapSizeLimit * 0.9) {
          console.warn('⚠️ High memory usage detected');
        }
      }, 30000);
    }
  }

  checkUserAuthentication() {
    // Check if user is logged in
    fetch('/api/check-auth', {
      method: 'GET',
      credentials: 'include'
    })
    .then(response => response.json())
    .then(data => {
      if (data.authenticated) {
        this.currentUser = data.user;
        this.updateUIForLoggedInUser();
      }
    })
    .catch(error => {
      console.log('User not authenticated');
    });
  }

  initializeOfflineSupport() {
    window.addEventListener('online', () => {
      this.offlineMode = false;
      this.showNotification('🌐 Back online!', 'success');
    });

    window.addEventListener('offline', () => {
      this.offlineMode = true;
      this.showNotification('📱 Offline mode activated', 'info');
    });
  }

  async analyzeImage() {
    console.log('🔬 Starting image analysis...');

    if (this.isAnalyzing) {
      this.showError('Analysis already in progress. Please wait...');
      return;
    }

    if (!this.selectedFile) {
      this.showError('Please select an image first');
      return;
    }

    this.isAnalyzing = true;
    console.log('📁 Selected file:', this.selectedFile.name, this.selectedFile.size, 'bytes');

    try {
      const formData = new FormData();
      formData.append('image', this.selectedFile);

      console.log('📤 Sending file upload request...');
      await this.sendAnalysisRequest('/analyze', {
        method: 'POST',
        body: formData
      });
    } catch (error) {
      console.error('❌ Analysis error:', error);
      this.showError('Analysis failed: ' + error.message);
    } finally {
      this.isAnalyzing = false;
    }
  }

  async analyzeImageData(base64Image, isWebcam = false) {
    console.log('📷 Starting webcam image analysis...');
    console.log('📊 Base64 image length:', base64Image.length);

    await this.sendAnalysisRequest('/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ image_base64: base64Image })
    });
  }

  async sendAnalysisRequest(url, options) {
    this.showLoading();
    this.startLoadingAnimation();

    try {
      console.log('🚀 Sending analysis request to:', url);

      // Add retry logic
      let lastError;
      for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
        try {
          console.log(`🔄 Analysis attempt ${attempt}/${this.maxRetries}`);

          const response = await fetch(url, {
            ...options,
            credentials: 'include'
          });

          console.log('📥 Response status:', response.status);

          const data = await response.json();
          console.log('📊 Response data:', data);

          if (!response.ok) {
            throw new Error(data.error || `HTTP ${response.status}: Analysis failed`);
          }

          if (data.success && data.result) {
            console.log('✅ Analysis successful, displaying results');
            this.displayResults(data.result);
            this.saveAnalysisHistory(data.result);
            this.showNotification('✅ Analysis completed successfully!', 'success');
            return;
          } else {
            throw new Error(data.error || 'Analysis failed - no result returned');
          }

        } catch (error) {
          console.error(`❌ Attempt ${attempt} failed:`, error);
          lastError = error;

          if (attempt < this.maxRetries) {
            const delay = Math.pow(2, attempt) * 1000;
            console.log(`⏳ Retrying in ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
          }
        }
      }

      // If all retries failed
      throw lastError;
        this.updateStats();
        showNotification('✅ Analysis completed successfully!', 'success');
      } else {
        throw new Error(data.error || 'Unexpected response format');
      }

    } catch (error) {
      console.error('❌ Analysis error:', error);
      this.showError('Network error. Please try again: ' + error.message);
    } finally {
      this.hideLoading();
    }
  }

  showLoading() {
    if (this.loadingContainer) {
      this.loadingContainer.style.display = 'block';
    }
    if (this.resultsContainer) {
      this.resultsContainer.style.display = 'none';
    }
    this.currentLoadingIndex = 0;
    
    // Animate loading messages
    this.loadingInterval = setInterval(() => {
      if (this.loadingText) {
        this.loadingText.textContent = this.loadingMessages[this.currentLoadingIndex];
        this.currentLoadingIndex = (this.currentLoadingIndex + 1) % this.loadingMessages.length;
      }
    }, 1000);
  }

  hideLoading() {
    if (this.loadingContainer) {
      this.loadingContainer.style.display = 'none';
    }
    if (this.loadingInterval) {
      clearInterval(this.loadingInterval);
    }
  }

  displayResults(result) {
    const { condition, confidence, remedies, processing_time, analysis_id } = result;
    
    // Create results HTML
    const resultsHTML = `
      <div class="condition-card animate-slideInUp">
        <div class="condition-header">
          <h2 class="condition-title">${this.formatConditionName(condition)}</h2>
          <div class="confidence-badge">${confidence}% Confidence</div>
        </div>
        
        <div class="remedies-grid">
          ${remedies.map(remedy => this.createRemedyCard(remedy)).join('')}
        </div>
        
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(76, 175, 80, 0.2); font-size: 0.85rem; color: var(--text-muted); text-align: center;">
          <i class="fas fa-clock"></i> Processed in ${processing_time}s • 
          <i class="fas fa-flask"></i> Analysis ID: ${analysis_id}
        </div>
      </div>
    `;
    
    if (this.resultsContainer) {
      this.resultsContainer.innerHTML = resultsHTML;
      this.resultsContainer.style.display = 'block';
    }
    
    // Animate effectiveness bars
    setTimeout(() => {
      document.querySelectorAll('.effectiveness-fill').forEach(bar => {
        const effectiveness = bar.dataset.effectiveness;
        bar.style.width = effectiveness + '%';
      });
    }, 500);
  }

  createRemedyCard(remedy) {
    const typeColor = remedy.type === 'primary' ? 'var(--primary-color)' : 'var(--secondary-color)';
    const ingredients = remedy.ingredients ? remedy.ingredients.map(ing => 
      `<span class="ingredient-tag">${ing}</span>`
    ).join('') : '';
    
    return `
      <div class="remedy-card hover-lift">
        <div class="remedy-header">
          <h3 class="remedy-title">${remedy.title}</h3>
          <span class="remedy-type" style="background: ${typeColor};">${remedy.type}</span>
        </div>
        
        <div class="remedy-meta">
          ${remedy.effectiveness ? `
            <div class="meta-item">
              <i class="fas fa-star"></i>
              <span>${remedy.effectiveness}% Effective</span>
            </div>
          ` : ''}
          ${remedy.difficulty ? `
            <div class="meta-item">
              <i class="fas fa-layer-group"></i>
              <span>${remedy.difficulty}</span>
            </div>
          ` : ''}
          ${remedy.time_to_prepare ? `
            <div class="meta-item">
              <i class="fas fa-clock"></i>
              <span>${remedy.time_to_prepare}</span>
            </div>
          ` : ''}
        </div>
        
        <div class="remedy-content">
          ${remedy.ingredients ? `
            <div class="remedy-section">
              <h4><i class="fas fa-leaf"></i> Ingredients</h4>
              <div class="ingredients-list">${ingredients}</div>
            </div>
          ` : ''}
          
          ${remedy.preparation ? `
            <div class="remedy-section">
              <h4><i class="fas fa-mortar-pestle"></i> Preparation</h4>
              <p>${remedy.preparation}</p>
            </div>
          ` : ''}
          
          ${remedy.application ? `
            <div class="remedy-section">
              <h4><i class="fas fa-hand-holding-medical"></i> Application</h4>
              <p>${remedy.application}</p>
            </div>
          ` : ''}
        </div>
        
        ${remedy.effectiveness ? `
          <div class="effectiveness-bar">
            <div class="effectiveness-fill" data-effectiveness="${remedy.effectiveness}"></div>
          </div>
        ` : ''}
      </div>
    `;
  }

  formatConditionName(condition) {
    return condition.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  showError(message) {
    showNotification(message, 'error');
  }

  updateStats() {
    // Simulate stats update
    if (this.totalAnalysesEl) {
      const currentCount = parseInt(this.totalAnalysesEl.textContent.replace(/\D/g, '')) || 1000;
      this.totalAnalysesEl.textContent = (currentCount + 1) + '+';
    }
  }
  // Additional utility methods
  startLoadingAnimation() {
    this.currentLoadingIndex = 0;
    if (this.loadingInterval) {
      clearInterval(this.loadingInterval);
    }
    this.loadingInterval = setInterval(() => {
      if (this.loadingText) {
        this.loadingText.textContent = this.loadingMessages[this.currentLoadingIndex];
        this.currentLoadingIndex = (this.currentLoadingIndex + 1) % this.loadingMessages.length;
      }
    }, 2000);
  }

  saveAnalysisHistory(result, isWebcam = false) {
    const historyItem = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      condition: result.condition,
      confidence: result.confidence,
      source: isWebcam ? 'webcam' : 'upload',
      remedies: result.remedies || []
    };

    this.analysisHistory.unshift(historyItem);

    // Keep only last 10 analyses
    if (this.analysisHistory.length > 10) {
      this.analysisHistory = this.analysisHistory.slice(0, 10);
    }

    // Save to localStorage
    try {
      localStorage.setItem('healayur_history', JSON.stringify(this.analysisHistory));
    } catch (error) {
      console.warn('Could not save to localStorage:', error);
    }
  }

  loadAnalysisHistory() {
    try {
      const saved = localStorage.getItem('healayur_history');
      if (saved) {
        this.analysisHistory = JSON.parse(saved);
      }
    } catch (error) {
      console.warn('Could not load from localStorage:', error);
      this.analysisHistory = [];
    }
  }

  updateUIForLoggedInUser() {
    // Update UI elements for logged-in user
    const userInfo = document.querySelector('.user-info');
    if (userInfo && this.currentUser) {
      userInfo.style.display = 'flex';
      const username = userInfo.querySelector('.username');
      if (username) {
        username.textContent = this.currentUser.username;
      }
    }
  }

  processVoiceCommand(command) {
    console.log('🎤 Voice command:', command);

    if (command.includes('analyze') || command.includes('scan')) {
      if (this.selectedFile) {
        this.analyzeImage();
      } else {
        this.showNotification('Please select an image first', 'info');
      }
    } else if (command.includes('camera') || command.includes('webcam')) {
      this.toggleWebcam();
    } else if (command.includes('chat') || command.includes('help')) {
      if (window.chatBot) {
        window.chatBot.toggleChat();
      }
    } else {
      this.showNotification('Command not recognized. Try "analyze", "camera", or "chat"', 'info');
    }
  }

  showPWAInstallPrompt() {
    const prompt = document.createElement('div');
    prompt.className = 'pwa-install-prompt show';
    prompt.innerHTML = `
      <div>
        <strong>📱 Install Heal Ayur</strong>
        <p>Get the full app experience!</p>
      </div>
      <button class="pwa-install-btn" onclick="this.parentElement.installPWA()">Install</button>
      <button class="pwa-install-btn" onclick="this.parentElement.remove()">×</button>
    `;

    prompt.installPWA = () => {
      if (this.deferredPrompt) {
        this.deferredPrompt.prompt();
        this.deferredPrompt.userChoice.then((choiceResult) => {
          if (choiceResult.outcome === 'accepted') {
            console.log('User accepted the PWA install prompt');
          }
          this.deferredPrompt = null;
        });
      }
      prompt.remove();
    };

    document.body.appendChild(prompt);

    // Auto-hide after 10 seconds
    setTimeout(() => {
      if (prompt.parentElement) {
        prompt.remove();
      }
    }, 10000);
  }

  showNotification(message, type = 'info', duration = 3000) {
    // Use the global showNotification function if available
    if (typeof showNotification === 'function') {
      showNotification(message, type, duration);
    } else {
      // Fallback notification
      console.log(`${type.toUpperCase()}: ${message}`);
    }
  }
}

// Initialize the application when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
  app = new HealAyurApp();
  // Make app globally available
  window.app = app;
  console.log('✅ HealAyur app initialized and available globally');
});
