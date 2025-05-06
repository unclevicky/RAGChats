
<template>
  <div class="system-settings-container">
    <!-- 左侧导航 -->
    <div class="nav-sidebar">
      <el-menu
        default-active="model-management"
        class="el-menu-vertical"
        @select="handleMenuSelect">
        <el-menu-item index="model-management">
          <span>模型管理</span>
        </el-menu-item>
        <el-menu-item index="assistant-management">
          <span>聊天助手管理</span>
        </el-menu-item>
        <el-menu-item index="system-info">
          <span>系统信息</span>
        </el-menu-item>
      </el-menu>
    </div>

    <!-- 右侧内容区 -->
    <div class="content-area">
      <!-- 模型管理 -->
      <div v-if="activeTab === 'model-management'" class="model-management">
        <!-- 模型列表 -->
        <div class="model-list">
          <h3>模型列表</h3>
          <el-table :data="models" highlight-current-row @current-change="handleModelSelect">
            <el-table-column prop="name" label="模型名称" width="180" />
            <el-table-column prop="type" label="类型" width="100">
              <template #default="{row}">
                {{ row.type === 'llm' ? '大模型' : '嵌入模型' }}
              </template>
            </el-table-column>
            <el-table-column prop="url" label="API地址" show-overflow-tooltip />
            <el-table-column prop="model" label="Model" />
            <el-table-column prop="provider" label="提供商" />
          </el-table>
        </div>

        <!-- 模型详情 -->
        <div class="model-detail">
          <h3>模型详情</h3>
          <el-form :model="modelForm" :rules="formRules" label-width="120px" ref="modelForm">
            <el-form-item label="模型名称" prop="name">
              <el-input v-model="modelForm.name" />
            </el-form-item>
            <el-form-item label="类型">
              <el-select v-model="modelForm.type">
                <el-option label="大模型(LLM)" value="llm" />
                <el-option label="嵌入模型(Embedding)" value="embedding" />
              </el-select>
            </el-form-item>
            <el-form-item label="API地址">
              <el-input v-model="modelForm.url" />
            </el-form-item>
            <el-form-item label="Model">
              <el-input v-model="modelForm.model" />
            </el-form-item>
            <el-form-item label="API密钥">
              <el-input v-model="modelForm.api_key" type="password" show-password />
            </el-form-item>
            <el-form-item label="提供商">
              <el-input v-model="modelForm.provider" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="saveModel">保存</el-button>
              <el-button type="danger" @click="deleteModel" :disabled="!selectedModel">删除</el-button>
            </el-form-item>
          </el-form>
        </div>
      </div>

      <!-- 聊天助手管理 -->
      <div v-else-if="activeTab === 'assistant-management'" class="assistant-management">
        <!-- 助手列表 -->
        <div class="assistant-list">
          <h3>助手列表</h3>
          <el-table :data="assistants" highlight-current-row @current-change="handleAssistantSelect">
            <el-table-column prop="name" label="助手名称" width="180" />
            <el-table-column prop="description" label="描述" show-overflow-tooltip />
            <el-table-column prop="model" label="聊天大模型" />
            <el-table-column prop="knowledge_base" label="知识库" />
            <el-table-column prop="system_prompt" label="系统提示词" show-overflow-tooltip />
            <el-table-column prop="embedding" label="嵌入模型" />
          </el-table>
        </div>

        <!-- 助手详情 -->
        <div class="assistant-detail">
          <h3>助手详情</h3>
          <el-form :model="assistantForm" :rules="assistantRules" label-width="120px" ref="assistantForm">
            <el-form-item label="助手名称" prop="name">
              <el-input v-model="assistantForm.name" />
            </el-form-item>
            <el-form-item label="描述">
              <el-input v-model="assistantForm.description" type="textarea" :rows="2" />
            </el-form-item>
            <el-form-item label="聊天大模型">
              <el-select v-model="assistantForm.model" placeholder="请选择大模型">
                <el-option
                  v-for="model in models.filter(m => m.type === 'llm')"
                  :key="model.name"
                  :label="model.name"
                  :value="model.name">
                </el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="嵌入模型">
              <el-select v-model="assistantForm.embedding" placeholder="请选择嵌入模型">
                <el-option
                  v-for="model in models.filter(m => m.type === 'embedding')"
                  :key="model.name"
                  :label="model.name"
                  :value="model.name">
                </el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="知识库">
              <el-select v-model="assistantForm.knowledge_base" placeholder="请选择知识库">
                <el-option
                  v-for="kb in knowledgeBases"
                  :key="kb"
                  :label="kb"
                  :value="kb">
                </el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="系统提示词">
              <el-input v-model="assistantForm.system_prompt" type="textarea" :rows="4" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="saveAssistant">保存</el-button>
              <el-button type="danger" @click="deleteAssistant" :disabled="!selectedAssistant">删除</el-button>
            </el-form-item>
          </el-form>
        </div>
      </div>

      <!-- 系统信息 -->
      <div v-else class="empty-placeholder">
        <h3>系统信息</h3>
        <p>功能开发中...</p>
      </div>
    </div>
  </div>
</template>

<script>
import { getModels, updateModel, deleteModel, getAssistants, updateAssistant, deleteAssistant, listKnowledgeBases } from '@/api/system'

export default {
  data() {
    return {
      activeTab: 'model-management',
      models: [],
      knowledgeBases: [],
      assistants: [],
      selectedModel: null,
      selectedAssistant: null,
      modelForm: this.getEmptyModelForm(),
      assistantForm: this.getEmptyAssistantForm(),
      formRules: {
        name: [
          { required: true, message: '请输入模型名称', trigger: 'blur' },
          { 
            validator: (rule, value, callback) => {
              if (!value || !value.trim()) {
                callback(new Error('模型名称不能为空'))
              } else {
                callback()
              }
            },
            trigger: 'blur'
          }
        ]
      },
      assistantRules: {
        name: [
          { required: true, message: '请输入助手名称', trigger: 'blur' },
          { 
            validator: (rule, value, callback) => {
              if (!value || !value.trim()) {
                callback(new Error('助手名称不能为空'))
              } else {
                callback()
              }
            },
            trigger: 'blur'
          }
        ]
      }
    }
  },
  methods: {
    getEmptyModelForm() {
      return {
        name: '',
        type: 'llm',
        url: '',
        model: '',
        api_key: '',
        provider: ''
      }
    },
    getEmptyAssistantForm() {
      return {
        name: '',
        description: '',
        model: '',
        knowledge_base: '',
        system_prompt: '',
        embedding: 'bge-large-zh-v1.5'
      }
    },
    async loadModels() {
      try {
        console.group('加载模型列表')
        console.log('当前模型列表:', this.models)
        this.models.splice(0, this.models.length)
        console.log('清空模型列表完成')
        
        // 添加请求超时和重试机制
        const maxRetries = 3
        let lastError = null
        
        for (let i = 0; i < maxRetries; i++) {
          try {
            const { data } = await getModels()
            console.log('API响应数据:', data)
            
            // 深度验证数据格式
            let validatedData = []
            if (Array.isArray(data)) {
              validatedData = data
            } else if (data && typeof data === 'object') {
              // 处理对象形式的返回数据
              validatedData = Object.values(data)
            }
            
            this.models = validatedData.map(item => ({
              name: item.name || item.id || '',
              type: item.type || 'llm',
              url: item.url || '',
              model: item.model || '',
              api_key: item.api_key || '',
              provider: item.provider || ''
            }))
            
            console.log('格式化后的模型列表:', this.models)
            console.log('模型数量:', this.models.length)
            
            // 同步选中状态
            if (this.selectedModel) {
              const current = this.models.find(m => 
                m.name && this.selectedModel.name && 
                m.name.toLowerCase() === this.selectedModel.name.toLowerCase()
              )
              this.selectedModel = current ? {...current} : null
              console.log('更新后的选中模型:', this.selectedModel)
            }
            
            // 如果成功则退出重试循环
            console.groupEnd()
            return
          } catch (error) {
            lastError = error
            console.warn(`加载模型失败 (尝试 ${i+1}/${maxRetries}):`, error)
            if (i < maxRetries - 1) {
              await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)))
            }
          }
        }
        
        // 所有重试都失败
        const errorMsg = '加载模型列表失败: ' + (lastError.response?.data?.message || lastError.message)
        this.$message.error(errorMsg)
        console.error('最终加载模型失败:', lastError)
        console.groupEnd()
      } catch (error) {
        console.groupEnd()
        const errorMsg = '加载模型列表失败: ' + (error.response?.data?.message || error.message)
        this.$message.error(errorMsg)
        console.error('加载模型失败:', error)
      }
    },
    async loadKnowledgeBases() {
      try {
        const { data } = await listKnowledgeBases()
        // 返回状态不是"维护中"或"disabled"的知识库(包括未设置status的)
        this.knowledgeBases = data
          .filter(kb => !kb.status || 
                     !['维护中', 'disabled'].includes(kb.status))
          .map(kb => kb.id)
      } catch (error) {
        console.error('加载知识库失败:', error)
        this.$message.error('加载知识库失败: ' + (error.response?.data?.message || error.message))
      }
    },
    async loadAssistants() {
      try {
        console.group('加载助手列表')
        const maxRetries = 3
        let lastError = null
        
        for (let i = 0; i < maxRetries; i++) {
          try {
            const { data } = await getAssistants()
            console.log('API响应数据:', data)
            
            // 处理多种数据格式
            let assistants = []
            if (Array.isArray(data)) {
              assistants = data.map(item => ({
                name: item.name || item.id || '',
                description: item.description || '',
                model: item.model || '',
                knowledge_base: item.knowledge_base || '',
                system_prompt: item.system_prompt || '',
                embedding: item.embedding || 'bge-large-zh-v1.5'
              }))
            } else if (data && typeof data === 'object') {
              assistants = Object.entries(data).map(([name, cfg]) => ({
                name,
                description: cfg.description || '',
                model: cfg.model || '',
                knowledge_base: cfg.knowledge_base || '',
                system_prompt: cfg.system_prompt || '',
                embedding: cfg.embedding || 'bge-large-zh-v1.5'
              }))
            }
            
            this.assistants = assistants
            console.log('格式化后的助手列表:', this.assistants)
            console.log('助手数量:', this.assistants.length)
            
            // 同步选中状态
            if (this.selectedAssistant) {
              const current = this.assistants.find(a => 
                a.name && this.selectedAssistant.name && 
                a.name.toLowerCase() === this.selectedAssistant.name.toLowerCase()
              )
              this.selectedAssistant = current ? {...current} : null
              console.log('更新后的选中助手:', this.selectedAssistant)
            }
            
            console.groupEnd()
            return
          } catch (error) {
            lastError = error
            console.warn(`加载助手失败 (尝试 ${i+1}/${maxRetries}):`, error)
            if (i < maxRetries - 1) {
              await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)))
            }
          }
        }
        
        // 所有重试都失败
        const errorMsg = '加载助手列表失败: ' + (lastError.response?.data?.message || lastError.message)
        this.$message.error(errorMsg)
        console.error('最终加载助手失败:', lastError)
        console.groupEnd()
      } catch (error) {
        console.groupEnd()
        const errorMsg = '加载助手列表失败: ' + (error.response?.data?.message || error.message)
        this.$message.error(errorMsg)
        console.error('加载助手失败:', error)
      }
    },
    handleMenuSelect(index) {
      this.activeTab = index
    },
    handleModelSelect(model) {
      this.selectedModel = model
      if (model) {
        this.modelForm = { ...model }
      }
    },
    handleAssistantSelect(assistant) {
      this.selectedAssistant = assistant
      if (assistant) {
        this.assistantForm = { ...assistant }
      }
    },
    createNewModel() {
      console.log('创建新模型')
      this.selectedModel = null
      this.modelForm = this.getEmptyModelForm()
      console.log('表单已重置:', this.modelForm)
    },
    async saveModel() {
      try {
        await this.$refs.modelForm.validate()
        const modelName = this.modelForm.name
        const { data } = await updateModel(modelName, this.modelForm)
        this.$message.success(data.message)
        console.log('保存成功，返回数据:', data)
        
        await this.loadModels()
        console.log('刷新后模型列表:', this.models.map(m => m.name))
        
        const savedModel = this.models.find(m => m.name.toLowerCase() === modelName.toLowerCase())
        if (savedModel) {
          this.selectedModel = {...savedModel}
          console.log('成功匹配模型:', this.selectedModel.name)
        } else {
          console.warn('模型未在列表中:', modelName)
          console.log('当前所有模型名称:', this.models.map(m => m.name))
          
          if (data.models) {
            const returnedModel = data.models.find(m => m.name.toLowerCase() === modelName.toLowerCase())
            if (returnedModel) {
              console.log('在返回数据中找到模型，手动添加到列表')
              this.models.push({...returnedModel})
              this.selectedModel = {...returnedModel}
            }
          }
        }
        
        if (!this.selectedModel) {
          this.modelForm = this.getEmptyModelForm()
        }
      } catch (error) {
        const errorMsg = error.response?.data?.detail || error.message || '操作失败'
        this.$message.error(errorMsg)
        console.error('保存模型失败:', error)
      }
    },
    async saveAssistant() {
      try {
        await this.$refs.assistantForm.validate()
        const assistantName = this.assistantForm.name
        const { data } = await updateAssistant(assistantName, this.assistantForm)
        this.$message.success(data.message)
        
        // 使用API返回的助手列表更新前端状态
        if (data.assistants) {
          this.assistants = data.assistants.map(item => ({
            name: item.name || '',
            description: item.description || '',
            model: item.model || '',
            knowledge_base: item.knowledge_base || '',
            system_prompt: item.system_prompt || '',
            embedding: item.embedding || 'text-embedding-3-small'
          }))
        } else {
          await this.loadAssistants()
        }
        
        // 更新选中状态
        const savedAssistant = this.assistants.find(a => 
          a.name.toLowerCase() === assistantName.toLowerCase()
        )
        if (savedAssistant) {
          this.selectedAssistant = {...savedAssistant}
          this.assistantForm = {...savedAssistant}
        } else {
          this.selectedAssistant = null
          this.assistantForm = this.getEmptyAssistantForm()
        }
      } catch (error) {
        const errorMsg = error.response?.data?.detail || error.message || '操作失败'
        this.$message.error(errorMsg)
        console.error('保存助手失败:', error)
      }
    },
    async deleteAssistant() {
      try {
        await this.$confirm('确认删除该助手配置吗?', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        })
        
        const assistantName = this.assistantForm.name
        await deleteAssistant(assistantName)
        this.$message.success('删除成功')
        this.assistantForm = this.getEmptyAssistantForm()
        await this.loadAssistants()
      } catch (error) {
        if (error !== 'cancel') {
          this.$message.error('删除失败')
          console.error('删除助手失败:', error)
        }
      }
    },
    async deleteModel() {
      try {
        await this.$confirm('确认删除该模型配置吗?', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        })
        
        const modelName = this.modelForm.name
        console.log('开始删除模型:', modelName)
        
        const { data } = await deleteModel(modelName)
        this.$message.success(data.message)
        console.log('删除成功响应:', data)
        
        // 强制刷新并验证
        await this.loadModels()
        console.log('刷新后模型列表:', this.models.map(m => m.name))
        
        // 验证删除结果
        const stillExists = this.models.some(m => 
          m.name.toLowerCase() === modelName.toLowerCase()
        )
        if (stillExists) {
          console.warn('模型仍在列表中，尝试手动移除')
          this.models = this.models.filter(m => 
            m.name.toLowerCase() !== modelName.toLowerCase()
          )
        }
        
        // 重置状态
        this.selectedModel = null
        this.modelForm = this.getEmptyModelForm()
        console.log('状态已重置')
        
      } catch (error) {
        if (error !== 'cancel') {
          const errorMsg = error.response?.data?.detail || error.message || '删除失败'
          this.$message.error(errorMsg)
          console.error('删除模型失败:', {
            error,
            response: error.response,
            config: error.config
          })
        }
      }
    }
  },
  created() {
    console.log('SystemSettings component created')
    this.loadModels()
    this.loadKnowledgeBases()
    this.loadAssistants()
  },
  mounted() {
    console.log('SystemSettings component mounted')
  }
}
</script>

<style scoped>
.system-settings-container {
  display: flex;
  height: calc(100vh - 60px);
}

.nav-sidebar {
  width: 200px;
  background-color: #f5f5f5;
  border-right: 1px solid #e6e6e6;
}

.content-area {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}

.model-management {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.model-list, .model-detail,
.assistant-list, .assistant-detail {
  background: white;
  padding: 20px;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.empty-placeholder {
  padding: 40px;
  text-align: center;
  color: #888;
}
</style>