
<template>
  <div class="knowledge-base">
    <h2>知识库管理</h2>
    
    <!-- 新建知识库表单 -->
    <div class="create-kb-form">
      <h3>新建知识库</h3>
      <div class="form-group">
        <label>知识库名称:</label>
        <input v-model="newKbName" placeholder="输入知识库名称" />
      </div>
      <button @click="createKnowledgeBase" :disabled="!newKbName">创建</button>
    </div>

    <!-- 知识库列表 -->
    <div class="kb-list">
      <h3>知识库列表</h3>
      <template v-if="knowledgeBases.length > 0">
        <div v-for="kb in knowledgeBases" :key="kb.id" class="kb-item">
          <router-link :to="`/knowledge-base/${kb.id}`">{{ kb.name }}</router-link>
          <button @click="deleteKB(kb.id)">删除</button>
        </div>
      </template>
      <div v-else class="empty-state">
        <p>暂无知识库，请先创建一个</p>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue';
import axios from 'axios';
import { useRouter } from 'vue-router';
import { getToken } from '../utils/auth.js';

export default {
  setup() {
    const router = useRouter();
    const knowledgeBases = ref([]);
    const newKbName = ref('');

    const isLoading = ref(false);
    
    const createKnowledgeBase = async () => {
      if (isLoading.value) return;
      
      isLoading.value = true;
      try {
        const kbId = newKbName.value.trim().toLowerCase().replace(/\s+/g, '_');
        const response = await axios.post(`/api/knowledge-bases/?kb_id=${encodeURIComponent(kbId)}`, {
          kb_name: newKbName.value
        }, {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${getToken()}`
          }
        });
        
        if (response.data && response.data.id) {
          // 显示创建成功提示
          alert(`知识库"${newKbName.value}"创建成功`);
          // 刷新知识库列表
          await fetchKnowledgeBases();
          // 清空输入框
          newKbName.value = '';
          // 跳转到新创建的知识库详情页
          router.push(`/knowledge-base/${response.data.id}`);
        } else {
          throw new Error('创建知识库失败: 未收到有效响应');
        }
      } catch (error) {
        console.error('创建知识库错误:', error);
        alert(`创建失败: ${error.response?.data?.detail || error.message}`);
      } finally {
        isLoading.value = false;
      }
    };

    const fetchKnowledgeBases = async () => {
      try {
        const response = await axios.get('/api/knowledge-bases', {
          headers: {
            'Authorization': `Bearer ${getToken()}`
          }
        });
        console.log('知识库列表响应:', response.data);
        knowledgeBases.value = Array.isArray(response.data) ? response.data : [];
      } catch (error) {
        console.error('获取知识库列表失败:', error);
        knowledgeBases.value = [];
      }
    };

    const deleteKB = async (id) => {
      try {
        const kbToDelete = knowledgeBases.value.find(kb => kb.id === id);
        if (!kbToDelete) return;
        
        const confirmDelete = window.confirm(`确定要删除知识库"${kbToDelete.name}"吗？此操作将删除所有相关数据且不可恢复`);
        if (!confirmDelete) return;
        
        await axios.delete(`/api/knowledge-bases/${id}`, {
          headers: {
            'Authorization': `Bearer ${getToken()}`
          }
        });
        // 显示删除成功提示
        alert(`知识库"${kbToDelete.name}"已删除`);
        // 刷新知识库列表
        await fetchKnowledgeBases();
      } catch (error) {
        console.error('删除知识库失败:', error);
        alert(`删除失败: ${error.response?.data?.detail || error.message}`);
      }
    };

    fetchKnowledgeBases();

    return {
      knowledgeBases,
      newKbName,
      createKnowledgeBase,
      deleteKB
    };
  }
};
</script>

<style scoped>
.knowledge-base {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
  background-color: #f8fafc;
}

.create-kb-form {
  margin: 24px 0;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  background: white;
  border: none;
}

.form-group {
  margin: 20px 0;
  display: flex;
  align-items: center;
}

.form-group label {
  width: 120px;
  font-weight: 500;
  color: #4a5568;
}

.form-group input {
  flex-grow: 1;
  padding: 10px 16px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  transition: all 0.2s;
}

.form-group input:focus {
  border-color: #4299e1;
  box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1);
}

.kb-list {
  margin-top: 32px;
}

.kb-list h3 {
  font-size: 20px;
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 16px;
}

.kb-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  margin-bottom: 8px;
  border-radius: 8px;
  background: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  transition: all 0.2s;
}

.kb-item:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.kb-item a {
  color: #2d3748;
  text-decoration: none;
  font-weight: 500;
  transition: color 0.2s;
}

.kb-item a:hover {
  color: #4299e1;
}

button {
  padding: 10px 20px;
  font-weight: 500;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  border: none;
}

button:not(:disabled) {
  background-color: #4299e1;
  color: white;
}

button:not(:disabled):hover {
  background-color: #3182ce;
}

button:disabled {
  background-color: #e2e8f0;
  color: #a0aec0;
  cursor: not-allowed;
}

.empty-state {
  padding: 24px;
  text-align: center;
  color: #718096;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* 响应式布局 */
@media (max-width: 768px) {
  .knowledge-base {
    padding: 16px;
  }
  
  .form-group {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .form-group input {
    width: 100%;
    margin-top: 8px;
  }
  
  .kb-item {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .kb-item button {
    margin-top: 12px;
    width: 100%;
  }
}
</style>