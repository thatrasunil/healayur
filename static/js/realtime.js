// Heal Ayur - Real-time Features
class RealTimeManager {
  constructor(app) {
    this.app = app;
    this.socket = null;
    this.isConnected = false;
    this.initializeWebSocket();
  }

  initializeWebSocket() {
    // Initialize WebSocket connection for real-time features
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws`;
      
      this.socket = new WebSocket(wsUrl);
      
      this.socket.onopen = () => {
        this.isConnected = true;
        console.log('WebSocket connected');
        showNotification('🔗 Real-time connection established', 'success', 2000);
      };
      
      this.socket.onmessage = (event) => {
        this.handleWebSocketMessage(JSON.parse(event.data));
      };
      
      this.socket.onclose = () => {
        this.isConnected = false;
        console.log('WebSocket disconnected');
        // Attempt to reconnect after 3 seconds
        setTimeout(() => this.initializeWebSocket(), 3000);
      };
      
      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (error) {
      console.log('WebSocket not available, using fallback methods');
    }
  }

  handleWebSocketMessage(data) {
    switch (data.type) {
      case 'analysis_result':
        this.handleAnalysisResult(data.payload);
        break;
      case 'user_count':
        this.updateUserCount(data.payload.count);
        break;
      case 'chat_message':
        this.handleChatMessage(data.payload);
        break;
      default:
        console.log('Unknown WebSocket message type:', data.type);
    }
  }

  sendMessage(type, payload) {
    if (this.socket && this.isConnected) {
      this.socket.send(JSON.stringify({ type, payload }));
    }
  }

  // Real-time Analysis Methods
  startContinuousAnalysis() {
    // Reset counters
    this.app.realTimeCount = 0;
    this.app.realTimeStartTime = Date.now();
    
    // Show real-time analysis indicator
    this.showRealTimeIndicator();
    
    // Start continuous analysis every 2.5 seconds
    this.app.continuousAnalysisInterval = setInterval(() => {
      if (this.app.webcamActive && this.app.videoElement?.videoWidth > 0) {
        this.performRealTimeAnalysis();
      }
    }, 2500);
    
    showNotification('🔴 LIVE: Real-time analysis started! Analyzing every 2.5 seconds', 'success', 4000);
  }

  async performRealTimeAnalysis() {
    try {
      const startTime = performance.now();
      this.app.realTimeCount++;
      
      // Create canvas to capture frame
      const canvas = document.createElement('canvas');
      canvas.width = this.app.videoElement.videoWidth;
      canvas.height = this.app.videoElement.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(this.app.videoElement, 0, 0);

      // Convert to base64
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
      const base64Image = dataUrl.split(',')[1];

      // Send for analysis via WebSocket if available, otherwise use HTTP
      if (this.isConnected) {
        this.sendMessage('analyze_realtime', { image: base64Image });
      } else {
        // Fallback to HTTP request
        const response = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: base64Image })
        });

        const data = await response.json();
        
        if (data.success && data.result) {
          const endTime = performance.now();
          const totalTime = ((endTime - startTime) / 1000).toFixed(2);
          
          data.result.realTimeMetrics = {
            analysisNumber: this.app.realTimeCount,
            totalTime: totalTime,
            avgTime: this.app.realTimeStartTime ? 
              (((Date.now() - this.app.realTimeStartTime) / 1000) / this.app.realTimeCount).toFixed(2) : totalTime
          };
          
          this.updateRealTimeResults(data.result);
        }
      }
      
    } catch (error) {
      console.log('Real-time analysis error:', error);
      if (this.app.realTimeCount % 5 === 0) {
        showNotification('⚠️ Real-time analysis experiencing issues', 'warning', 2000);
      }
    }
  }

  handleAnalysisResult(result) {
    // Handle real-time analysis result from WebSocket
    const endTime = performance.now();
    const totalTime = ((endTime - this.analysisStartTime) / 1000).toFixed(2);
    
    result.realTimeMetrics = {
      analysisNumber: this.app.realTimeCount,
      totalTime: totalTime,
      avgTime: this.app.realTimeStartTime ? 
        (((Date.now() - this.app.realTimeStartTime) / 1000) / this.app.realTimeCount).toFixed(2) : totalTime
    };
    
    this.updateRealTimeResults(result);
  }

  updateRealTimeResults(result) {
    const overlay = document.getElementById('realTimeOverlay');
    if (overlay) {
      const timestamp = new Date().toLocaleTimeString();
      const confidenceColor = result.confidence > 80 ? '#4caf50' : result.confidence > 60 ? '#ff9800' : '#f44336';
      
      overlay.innerHTML = `
        <div style="background: rgba(0,0,0,0.9); color: white; padding: 1rem; border-radius: 12px; margin: 1rem; border: 2px solid #4caf50; backdrop-filter: blur(10px);">
          <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
            <div style="width: 10px; height: 10px; background: #4caf50; border-radius: 50%; animation: pulse 1s infinite; box-shadow: 0 0 10px #4caf50;"></div>
            <strong style="color: #4caf50; font-family: 'Orbitron', monospace;">🔴 LIVE ANALYSIS</strong>
          </div>
          <div style="font-size: 0.9rem; line-height: 1.4;">
            <div style="margin-bottom: 0.25rem;">🎯 <strong style="color: #81c784;">${this.app.formatConditionName(result.condition)}</strong></div>
            <div style="margin-bottom: 0.25rem;">📊 Confidence: <span style="color: ${confidenceColor}; font-weight: bold;">${result.confidence}%</span></div>
            <div style="margin-bottom: 0.25rem;">⚡ Speed: <span style="color: #81c784;">${result.processing_time}s</span></div>
            <div style="margin-bottom: 0.25rem;">🕒 Last: <span style="color: #ccc; font-size: 0.8rem;">${timestamp}</span></div>
            ${result.realTimeMetrics ? `
              <div style="margin-bottom: 0.25rem;">📈 Analysis #${result.realTimeMetrics.analysisNumber}</div>
              <div style="margin-bottom: 0.25rem;">⏱️ Total: ${result.realTimeMetrics.totalTime}s | Avg: ${result.realTimeMetrics.avgTime}s</div>
            ` : ''}
            <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(76, 175, 80, 0.3); font-size: 0.8rem; color: #aaa;">
              💡 ${result.remedies ? result.remedies.length : 0} remedies available
            </div>
          </div>
        </div>
      `;
    }
  }

  showRealTimeIndicator() {
    if (!document.getElementById('realTimeOverlay') && this.app.webcamContainer) {
      const overlay = document.createElement('div');
      overlay.id = 'realTimeOverlay';
      overlay.style.cssText = `
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 100;
        pointer-events: none;
      `;
      this.app.webcamContainer.style.position = 'relative';
      this.app.webcamContainer.appendChild(overlay);
    }
  }

  toggleRealTimeAnalysis() {
    if (!this.app.webcamActive) {
      showNotification('Please start the camera first', 'warning');
      return;
    }

    if (this.app.realTimeActive) {
      // Stop real-time analysis
      if (this.app.continuousAnalysisInterval) {
        clearInterval(this.app.continuousAnalysisInterval);
        this.app.continuousAnalysisInterval = null;
      }
      
      const overlay = document.getElementById('realTimeOverlay');
      if (overlay) {
        overlay.remove();
      }
      
      this.app.realTimeActive = false;
      if (this.app.toggleRealTimeBtn) {
        this.app.toggleRealTimeBtn.innerHTML = '<i class="fas fa-play"></i> <span>Start Real-Time</span>';
        this.app.toggleRealTimeBtn.classList.remove('btn-warning');
        this.app.toggleRealTimeBtn.classList.add('btn-secondary');
      }
      
      showNotification('⏸️ Real-time analysis paused', 'info', 2000);
      
    } else {
      // Start real-time analysis
      this.startContinuousAnalysis();
      this.app.realTimeActive = true;
      if (this.app.toggleRealTimeBtn) {
        this.app.toggleRealTimeBtn.innerHTML = '<i class="fas fa-pause"></i> <span>Pause Real-Time</span>';
        this.app.toggleRealTimeBtn.classList.remove('btn-secondary');
        this.app.toggleRealTimeBtn.classList.add('btn-warning');
      }
      
      showNotification('🔴 LIVE: Real-time analysis started!', 'success', 3000);
    }
  }

  updateUserCount(count) {
    const liveUsers = document.getElementById('liveUsers');
    if (liveUsers) {
      liveUsers.textContent = count;
    }
  }

  handleChatMessage(message) {
    // Handle incoming chat messages from WebSocket
    if (window.chatManager) {
      window.chatManager.addMessage(message.text, 'ai');
    }
  }
}

// Add real-time analysis method to main app
if (typeof HealAyurApp !== 'undefined') {
  HealAyurApp.prototype.toggleRealTimeAnalysis = function() {
    if (this.realTimeManager) {
      this.realTimeManager.toggleRealTimeAnalysis();
    }
  };
}

// Initialize real-time manager when app is ready
document.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    if (window.app) {
      window.app.realTimeManager = new RealTimeManager(window.app);
    }
  }, 1000);
});
