
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
          <p v-html="formatMessage(message.content)"></p>
          <small v-if="message.source">{{ message.source }}</small>
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
        source: null
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
              const data = JSON.parse(line.replace('data:', '').trim());
              if (data.content) {
                botMessage.content += data.content;
                // 强制触发Vue响应式更新
                chatHistory.value = [...chatHistory.value];
              }
            } catch (e) {
              console.error('解析消息失败:', e);
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

    return {
      assistants,
      selectedAssistant,
      question,
      chatHistory,
      sendQuestion,
      chatAreaRef,
      formatMessage
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
</style>
