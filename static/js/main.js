// Heal Ayur - Main Application JavaScript
class HealAyurApp {
  constructor() {
    this.initializeElements();
    this.initializeEventListeners();
    this.initializeState();
    this.loadingMessages = [
      "🔍 Analyzing your image...",
      "🧠 AI is thinking...",
      "🌿 Finding ancient remedies...",
      "📊 Calculating confidence...",
      "✨ Almost ready..."
    ];
    this.currentLoadingIndex = 0;
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
    if (!this.webcamActive) {
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          } 
        });
        
        if (this.videoElement) {
          this.videoElement.srcObject = this.stream;
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
        showNotification('📷 Camera ready! Click "Start Real-Time" for continuous analysis', 'info', 4000);
        
      } catch (err) {
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
    if (!this.webcamActive || !this.videoElement?.videoWidth) {
      this.showError('Camera not ready. Please wait a moment and try again.');
      return;
    }

    // Create canvas to capture frame
    const canvas = document.createElement('canvas');
    canvas.width = this.videoElement.videoWidth;
    canvas.height = this.videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(this.videoElement, 0, 0);

    // Convert to base64
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    const base64Image = dataUrl.split(',')[1];

    // Stop webcam and analyze
    this.stopWebcam();
    await this.analyzeImageData(base64Image, true);
  }

  async analyzeImage() {
    if (!this.selectedFile) {
      this.showError('Please select an image first');
      return;
    }

    const formData = new FormData();
    formData.append('image', this.selectedFile);
    
    await this.sendAnalysisRequest('/analyze', {
      method: 'POST',
      body: formData
    });
  }

  async analyzeImageData(base64Image, isWebcam = false) {
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
    
    try {
      const response = await fetch(url, options);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Analysis failed');
      }

      if (data.success && data.result) {
        this.displayResults(data.result);
        this.updateStats();
      } else {
        throw new Error(data.error || 'Unexpected response format');
      }
      
    } catch (error) {
      this.showError('Analysis failed: ' + error.message);
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
}

// Initialize the application when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
  app = new HealAyurApp();
});
