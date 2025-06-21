// Heal Ayur - Intelligent Chatbot
class ChatBot {
  constructor() {
    this.isOpen = false;
    this.isTyping = false;
    this.conversationHistory = [];
    this.initializeElements();
    this.initializeEventListeners();
    this.loadKnowledgeBase();
  }

  initializeElements() {
    this.aiAssistant = document.getElementById('aiAssistant');
    this.aiChat = document.getElementById('aiChat');
    this.chatMessages = document.getElementById('chatMessages');
    this.chatInput = document.getElementById('chatInput');
    this.sendButton = document.querySelector('#aiChat button[onclick="sendChatMessage()"]');
  }

  initializeEventListeners() {
    if (this.aiAssistant) {
      this.aiAssistant.addEventListener('click', () => this.toggleChat());
    }
    
    if (this.chatInput) {
      this.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendMessage();
        }
      });
    }
    
    if (this.sendButton) {
      this.sendButton.addEventListener('click', () => this.sendMessage());
    }
  }

  loadKnowledgeBase() {
    this.knowledgeBase = {
      greetings: [
        "Hello! I'm your AI healing assistant powered by Google Gemini. How can I help you today? 🌿",
        "Hi there! I'm here to help you with natural remedies and healing advice using advanced AI. 🌱",
        "Welcome! I'm your personal Ayurvedic assistant with Gemini AI. What would you like to know? ✨"
      ],
      
      conditions: {
        acne: {
          description: "Acne is a common skin condition caused by clogged pores and bacteria.",
          causes: ["Hormonal changes", "Excess oil production", "Bacteria", "Genetics"],
          prevention: ["Keep skin clean", "Avoid touching face", "Use non-comedogenic products"],
          remedies: ["Turmeric-honey mask", "Neem face wash", "Tea tree oil treatment"]
        },
        
        burns: {
          description: "Burns are tissue damage from heat, chemicals, electricity, or radiation.",
          firstAid: ["Cool with cold water", "Remove from heat source", "Don't use ice"],
          remedies: ["Aloe vera gel", "Cold milk compress", "Honey application"]
        },
        
        rash: {
          description: "Skin rashes are areas of irritated or swollen skin.",
          causes: ["Allergies", "Infections", "Heat", "Stress"],
          remedies: ["Oatmeal bath", "Coconut oil", "Chamomile compress"]
        }
      },
      
      ingredients: {
        turmeric: "Anti-inflammatory and antibacterial properties. Great for skin healing.",
        honey: "Natural antibacterial and moisturizing. Helps heal wounds and acne.",
        aloe: "Cooling and healing properties. Excellent for burns and irritation.",
        neem: "Powerful antibacterial and antifungal. Traditional acne treatment.",
        coconut_oil: "Moisturizing and antimicrobial. Good for dry skin and eczema."
      },
      
      faqs: [
        {
          question: "How accurate is the AI analysis?",
          answer: "Our AI has 95% accuracy rate and analyzes over 15 different skin conditions using advanced computer vision."
        },
        {
          question: "Are these remedies safe?",
          answer: "These are traditional remedies used for thousands of years. However, always patch test first and consult a doctor for serious conditions."
        },
        {
          question: "How long do remedies take to work?",
          answer: "Most remedies show results within 1-2 weeks of consistent use. Some may work faster depending on the condition."
        }
      ]
    };
  }

  toggleChat() {
    this.isOpen = !this.isOpen;
    if (this.aiChat) {
      this.aiChat.style.display = this.isOpen ? 'flex' : 'none';
    }
    
    if (this.isOpen && this.conversationHistory.length === 0) {
      // Send welcome message
      setTimeout(() => {
        this.addMessage(this.getRandomGreeting(), 'ai');
      }, 500);
    }
    
    if (this.isOpen && this.chatInput) {
      this.chatInput.focus();
    }
  }

  async sendMessage() {
    const message = this.chatInput?.value.trim();
    if (!message) return;

    // Add user message
    this.addMessage(message, 'user');
    this.conversationHistory.push({ role: 'user', content: message });

    // Clear input
    if (this.chatInput) {
      this.chatInput.value = '';
    }

    // Show typing indicator
    this.showTypingIndicator();

    try {
      // Send message to Gemini API
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: message,
          history: this.conversationHistory.slice(-10) // Send last 10 messages for context
        })
      });

      const data = await response.json();

      this.hideTypingIndicator();

      if (data.success) {
        // Add AI response
        this.addMessage(data.response, 'ai');
        this.conversationHistory.push({ role: 'ai', content: data.response });

        // Show model info if available
        if (data.model && data.model.includes('gemini')) {
          console.log(`💬 Response from ${data.model} (Google Gemini API)`);
        }
      } else {
        // Show error message to user
        const errorMessage = data.error || 'AI chat service is temporarily unavailable. Please try again in a moment.';
        this.addMessage(`⚠️ ${errorMessage}`, 'ai');
        console.error('Chat API error:', data.error);
      }

    } catch (error) {
      console.error('Chat API error:', error);
      this.hideTypingIndicator();

      // Show error message to user
      const errorMessage = '⚠️ Unable to connect to AI chat service. Please check your connection and try again.';
      this.addMessage(errorMessage, 'ai');
    }
  }

  addMessage(text, sender) {
    if (!this.chatMessages) return;

    console.log('💬 Adding message:', sender, 'Length:', text.length);
    console.log('💬 Message preview:', text.substring(0, 100) + (text.length > 100 ? '...' : ''));

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender} animate-fadeInUp`;

    // Create message content with proper styling for long messages
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.style.cssText = `
      word-wrap: break-word;
      white-space: pre-wrap;
      max-width: 100%;
      overflow-wrap: break-word;
      line-height: 1.5;
      padding: 0.75rem;
      margin: 0.5rem 0;
      border-radius: 12px;
      background: ${sender === 'user' ? 'var(--primary-color)' : 'rgba(76, 175, 80, 0.1)'};
      color: ${sender === 'user' ? 'white' : 'var(--text-primary)'};
      border: ${sender === 'ai' ? '1px solid rgba(76, 175, 80, 0.2)' : 'none'};
      max-height: 300px;
      overflow-y: auto;
    `;

    // Use innerHTML for AI messages to support formatting, textContent for user messages for security
    if (sender === 'ai') {
      messageContent.innerHTML = this.formatAIMessage(text);
    } else {
      messageContent.textContent = text;
    }

    messageDiv.appendChild(messageContent);

    // Add timestamp
    const timestamp = document.createElement('div');
    timestamp.className = 'message-timestamp';
    timestamp.style.cssText = `
      font-size: 0.7rem;
      color: var(--text-muted);
      text-align: ${sender === 'user' ? 'right' : 'left'};
      margin-top: 0.25rem;
    `;
    timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    messageDiv.appendChild(timestamp);

    this.chatMessages.appendChild(messageDiv);
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;

    console.log('✅ Message added successfully');
  }

  formatAIMessage(text) {
    // Format AI messages with better readability
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold text
      .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic text
      .replace(/\n\n/g, '</p><p>') // Paragraphs
      .replace(/\n/g, '<br>') // Line breaks
      .replace(/^/, '<p>') // Start paragraph
      .replace(/$/, '</p>'); // End paragraph
  }

  showTypingIndicator() {
    if (!this.chatMessages) return;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    `;
    
    this.chatMessages.appendChild(typingDiv);
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    this.isTyping = true;
  }

  hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
      typingIndicator.remove();
    }
    this.isTyping = false;
  }

  // Note: Local response generation removed - chatbot now uses only Google Gemini API
}

// Global functions for backward compatibility
function toggleAIChat() {
  if (window.chatBot) {
    window.chatBot.toggleChat();
  }
}

function sendChatMessage() {
  if (window.chatBot) {
    window.chatBot.sendMessage();
  }
}

// Initialize chatbot
document.addEventListener('DOMContentLoaded', () => {
  window.chatBot = new ChatBot();
  window.chatManager = window.chatBot; // For real-time integration
});
