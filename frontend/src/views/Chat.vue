<template>
  <div class="chat-container">
    <div class="assistant-panel">
      <h3>聊天助手</h3>
      <div class="assistant-selector">
        <select v-model="selectedAssistant">
          <option v-for="assistant in assistants" :key="assistant.id" :value="assistant">
            {{ assistant.name }}
          </option>
        </select>
      </div>
      <div v-if="selectedAssistant" class="assistant-details">
        <h4>{{ selectedAssistant.name }}</h4>
        <p>{{ selectedAssistant.description }}</p>
        <div class="detail-item">
          <span class="label">聊天模型:</span>
          <span>{{ selectedAssistant.model }}</span>
        </div>
        <div class="detail-item">
          <span class="label">知识库:</span>
          <span>{{ selectedAssistant.knowledge_base }}</span>
        </div>
        <div class="detail-item" v-if="selectedAssistant.embedding !== undefined">
          <span class="label">嵌入模型:</span>
          <span>{{ selectedAssistant.embedding }}</span>
        </div>
        <div class="detail-item" v-else>
          <span class="label">嵌入模型:</span>
          <span>text-embedding-3-small</span>
        </div>
        <div class="detail-item">
          <span class="label">系统提示词:</span>
          <pre>{{ selectedAssistant.system_prompt }}</pre>
        </div>
      </div>
    </div>
    <div class="chat-panel">
      <div class="chat-area" ref="chatAreaRef">
        <div v-for="(message, index) in chatHistory" :key="index" class="message" :class="message.type">
          <!-- 用户消息 -->
          <p v-if="message.type === 'user'" v-html="formatMessage(message.content)"></p>
          
          <!-- 机器人消息 -->
          <div v-else>
            <!-- 思考过程 -->
            <div v-if="message.thinking" class="thinking-section" :class="{ 'collapsed': !message.showThinking }">
              <div class="thinking-header" @click="toggleThinking(message)">
                <span>思考过程</span>
                <span class="toggle-icon">{{ message.showThinking ? '▼' : '▶' }}</span>
              </div>
              <div class="thinking-content">
                <p v-html="formatMessage(message.thinking)"></p>
              </div>
            </div>
            
            <!-- 回答内容 -->
            <div class="answer-content">
              <p v-html="formatMessage(message.content)"></p>
            </div>
            
            <!-- 引用源 -->
            <div v-if="message.sources && message.sources.length > 0" class="sources-section">
              <div class="sources-header" @click="message.showSources = !message.showSources">
                <span>引用内容 ({{ message.sources.length }})</span>
                <span class="toggle-icon">{{ message.showSources ? '▼' : '▶' }}</span>
              </div>
              <div v-show="message.showSources" class="sources-content">
                <div v-for="(source, sIndex) in message.sources" :key="sIndex" class="source-item">
                  <div class="source-title">
                    <span class="source-index">{{ sIndex + 1 }}</span>
                    <span class="source-file">{{ getFileName(source.file) }}</span>
                  </div>
                  <p class="source-text" v-html="formatMessage(source.text)"></p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="input-area">
        <textarea v-model="question" placeholder="请输入问题..." @keyup.enter="sendQuestion"></textarea>
        <button @click="sendQuestion">发送</button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, watch, nextTick } from 'vue';
import axios from 'axios';

export default {
  setup() {
    const assistants = ref([]);
    const selectedAssistant = ref(null);
    const question = ref('');
    const chatHistory = ref([]);

    onMounted(async () => {
      try {
        const response = await axios.get('/api/system/assistants');
        assistants.value = response.data;
      } catch (error) {
        console.error('获取助手列表失败:', error);
      }
    });

    const sendQuestion = async () => {
      if (!selectedAssistant.value || !question.value.trim()) return;

      const userMessage = {
        type: 'user',
        content: question.value,
        source: null
      };
      chatHistory.value.push(userMessage);

      const botMessage = {
        type: 'bot',
        content: '',
        thinking: '',  // 思考过程
        sources: [],   // 引用源
        showThinking: false,  // 控制思考过程的显示(默认折叠)
        showSources: false   // 控制引用源的显示
      };
      chatHistory.value.push(botMessage);

      try {
        // 使用fetch API发送POST请求并处理SSE流
        const response = await fetch('http://localhost:8000/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
          },
          body: JSON.stringify({
            assistant: selectedAssistant.value,
            question: question.value,
            history: chatHistory.value
          })
        });

        if (!response.ok) {
          throw new Error(`请求失败: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          
          // 处理可能的分块数据
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || ''; // 保留未完成的行
          
          for (const line of lines) {
            if (!line.startsWith('data:')) continue;
            
            try {
              const jsonStr = line.replace('data:', '').trim();
              const data = JSON.parse(jsonStr);
              
              // 根据消息类型处理
              if (data.type === 'thinking') {
                // 更新思考过程
                botMessage.thinking = data.content;
                // 思考过程初始是展开的，便于用户看到思考过程
                botMessage.showThinking = true;
              } else if (data.type === 'answer') {
                // 更新答案内容
                botMessage.content += data.content;
                // 当开始接收答案时，折叠思考过程
                if (botMessage.thinking && data.content.trim() !== '') {
                  botMessage.showThinking = false;
                }
              } else if (data.type === 'sources') {
                // 更新引用源
                botMessage.sources = data.content;
              } else if (data.content) {
                // 兼容旧格式
                botMessage.content += data.content;
              }
              
              // 强制触发Vue响应式更新
              chatHistory.value = [...chatHistory.value];
            } catch (e) {
              console.error('解析消息失败:', e, line);
            }
          }
        }
      } catch (error) {
        console.error('聊天请求出错:', error);
        botMessage.content = '请求出错: ' + error.message;
      } finally {
        question.value = '';
      }
    };

    // 自动滚动到底部
    const chatAreaRef = ref(null);
    
    watch(chatHistory, () => {
      nextTick(() => {
        if (chatAreaRef.value) {
          chatAreaRef.value.scrollTop = chatAreaRef.value.scrollHeight;
        }
      });
    }, { deep: true });

    const formatMessage = (text) => {
      if (!text) return '';
      // 保留换行和空格
      return text
        .replace(/\n/g, '<br>')
        .replace(/  /g, ' &nbsp;')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // 加粗
        .replace(/\*(.*?)\*/g, '<em>$1</em>'); // 斜体
    };

    const getFileName = (file) => {
      if (!file) return '';
      const parts = file.split('/');
      return parts[parts.length - 1];
    };

    const toggleThinking = (message) => {
      // 切换思考过程的显示状态
      message.showThinking = !message.showThinking;
      // 强制触发Vue响应式更新
      chatHistory.value = [...chatHistory.value];
    };

    return {
      assistants,
      selectedAssistant,
      question,
      chatHistory,
      sendQuestion,
      chatAreaRef,
      formatMessage,
      getFileName,
      toggleThinking
    };
  }
};
</script>

<style scoped>
.chat-container {
  display: flex;
  height: 100vh;
  padding: 0;
}

.assistant-panel {
  width: 300px;
  padding: 20px;
  border-right: 1px solid #e0e0e0;
  overflow-y: auto;
  height: 100vh;
  box-sizing: border-box;
  background-color: #f8fafc;
}

.chat-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px;
  height: 100vh;
  box-sizing: border-box;
}

.assistant-selector {
  margin-bottom: 25px;
}

.assistant-selector select {
  width: 100%;
  padding: 10px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  background-color: white;
  font-size: 14px;
  transition: all 0.3s ease;
}

.assistant-selector select:focus {
  outline: none;
  border-color: #4CAF50;
  box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
}

.assistant-details {
  margin-top: 25px;
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
}

.assistant-details:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}

.detail-item {
  margin: 12px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid #f0f0f0;
}

.detail-item:last-child {
  border-bottom: none;
}

.label {
  font-weight: 600;
  color: #555;
  margin-right: 10px;
  display: inline-block;
  width: 80px;
}

pre {
  white-space: pre-wrap;
  background-color: #f5f7fa;
  padding: 12px;
  border-radius: 6px;
  font-size: 0.9em;
  border-left: 3px solid #4CAF50;
  margin-top: 10px;
}

.chat-area {
  flex: 1;
  overflow-y: auto;
  border: 1px solid #eee;
  padding: 15px;
  margin-bottom: 10px;
  max-height: calc(100vh - 180px);
}

.message {
  margin-bottom: 15px;
  padding: 10px;
  border-radius: 5px;
  word-break: break-word;
}

.message.user {
  background-color: #e3f2fd;
  margin-left: 20%;
}

.message.bot {
  background-color: #f5f5f5;
  margin-right: 20%;
}

.input-area {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  min-height: 60px;
}

.input-area textarea {
  flex-grow: 1;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  min-height: 60px;
  max-height: 200px;
  resize: vertical;
  font-family: inherit;
  line-height: 1.5;
}

.input-area button {
  padding: 10px 20px;
  height: 60px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  flex-shrink: 0;
}

.input-area button:hover {
  background-color: #45a049;
}

.thinking-section {
  margin-bottom: 10px;
  background-color: #f9f9f9;
  border-radius: 5px;
  padding: 8px;
  border-left: 3px solid #e0e0e0;
}

.thinking-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  font-weight: 600;
  color: #555;
}

.toggle-icon {
  margin-left: 5px;
}

.thinking-content {
  margin-top: 5px;
  margin-left: 20px;
  font-size: 0.95em;
  color: #666;
}

.thinking-section.collapsed .thinking-content {
  display: none;
}

.answer-content {
  margin-top: 10px;
  font-size: 1.05em;
}

.sources-section {
  margin-top: 10px;
  background-color: #f5f7fa;
  border-radius: 5px;
  padding: 8px;
  border-left: 3px solid #4CAF50;
}

.sources-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
}

.source-item {
  margin-bottom: 5px;
}

.source-title {
  font-weight: 600;
}

.source-index {
  margin-right: 5px;
}

.source-file {
  font-weight: 400;
}

.source-text {
  margin-left: 20px;
}
</style>
