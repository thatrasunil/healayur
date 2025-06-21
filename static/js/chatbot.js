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
        "Hello! I'm your AI healing assistant. How can I help you today?",
        "Hi there! I'm here to help you with natural remedies and healing advice.",
        "Welcome! I'm your personal Ayurvedic assistant. What would you like to know?"
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

  sendMessage() {
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
    
    // Process message and respond
    setTimeout(() => {
      this.hideTypingIndicator();
      const response = this.generateResponse(message);
      this.addMessage(response, 'ai');
      this.conversationHistory.push({ role: 'ai', content: response });
    }, 1000 + Math.random() * 2000); // Realistic typing delay
  }

  addMessage(text, sender) {
    if (!this.chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender} animate-fadeInUp`;
    messageDiv.textContent = text;
    
    this.chatMessages.appendChild(messageDiv);
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
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

  generateResponse(userMessage) {
    const message = userMessage.toLowerCase();
    
    // Greeting responses
    if (this.isGreeting(message)) {
      return this.getRandomGreeting();
    }
    
    // Help with analysis
    if (message.includes('analyze') || message.includes('scan')) {
      return "To analyze your skin condition, you can either upload an image or use your camera. Click 'Upload Image' to select a photo, or 'Use Camera' for real-time analysis. I'll help you understand the results!";
    }
    
    // Condition-specific questions
    for (const [condition, info] of Object.entries(this.knowledgeBase.conditions)) {
      if (message.includes(condition)) {
        return this.getConditionInfo(condition, info, message);
      }
    }
    
    // Ingredient questions
    for (const [ingredient, info] of Object.entries(this.knowledgeBase.ingredients)) {
      if (message.includes(ingredient.replace('_', ' '))) {
        return `${ingredient.replace('_', ' ').toUpperCase()}: ${info}`;
      }
    }
    
    // FAQ responses
    for (const faq of this.knowledgeBase.faqs) {
      if (this.messageMatchesFAQ(message, faq.question)) {
        return faq.answer;
      }
    }
    
    // Real-time features
    if (message.includes('real-time') || message.includes('live')) {
      return "For real-time analysis, start your camera and click 'Start Real-Time'. The AI will continuously analyze your skin every 2.5 seconds and show live results!";
    }
    
    // Voice commands
    if (message.includes('voice') || message.includes('speak')) {
      return "You can use voice commands! Press Ctrl+V or click the microphone icon. Say 'analyze image' to start analysis or 'start camera' to begin webcam capture.";
    }
    
    // General help
    if (message.includes('help') || message.includes('how')) {
      return "I can help you with:\n• Understanding skin conditions\n• Explaining remedy ingredients\n• Guiding you through the analysis process\n• Answering questions about natural healing\n\nWhat would you like to know?";
    }
    
    // Default responses
    const defaultResponses = [
      "That's an interesting question! Could you be more specific about what you'd like to know?",
      "I'd be happy to help! Can you tell me more about your skin concern?",
      "Let me help you with that. Are you looking for information about a specific condition or remedy?",
      "I'm here to assist with natural healing and skin analysis. What specific topic interests you?"
    ];
    
    return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
  }

  isGreeting(message) {
    const greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'];
    return greetings.some(greeting => message.includes(greeting));
  }

  getRandomGreeting() {
    return this.knowledgeBase.greetings[Math.floor(Math.random() * this.knowledgeBase.greetings.length)];
  }

  getConditionInfo(condition, info, message) {
    if (message.includes('cause') || message.includes('why')) {
      return `${condition.toUpperCase()} causes: ${info.causes ? info.causes.join(', ') : 'Various factors can contribute to this condition.'}`;
    }
    
    if (message.includes('prevent') || message.includes('avoid')) {
      return `To prevent ${condition}: ${info.prevention ? info.prevention.join(', ') : 'Maintain good hygiene and healthy lifestyle.'}`;
    }
    
    if (message.includes('treat') || message.includes('remedy')) {
      return `Natural remedies for ${condition}: ${info.remedies ? info.remedies.join(', ') : 'Several natural treatments are available.'}`;
    }
    
    return `${info.description} ${info.remedies ? 'Natural remedies include: ' + info.remedies.join(', ') : ''}`;
  }

  messageMatchesFAQ(message, question) {
    const questionWords = question.toLowerCase().split(' ');
    const messageWords = message.split(' ');
    
    let matches = 0;
    for (const word of questionWords) {
      if (messageWords.some(mWord => mWord.includes(word) || word.includes(mWord))) {
        matches++;
      }
    }
    
    return matches >= Math.min(3, questionWords.length * 0.6);
  }
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
