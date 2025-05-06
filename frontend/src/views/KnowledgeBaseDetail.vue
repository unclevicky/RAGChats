<template>
  <div class="knowledge-base-detail">
    <!-- 基础信息卡片 -->
    <el-card class="base-info-card">
      <template #header>
        <div class="card-header">
          <span>基础信息</span>
        </div>
      </template>
      
      <el-form :model="knowledgeBase" label-width="140px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="知识库ID">
              <el-input v-model="knowledgeBase.kb_id" disabled />
            </el-form-item>
            <el-form-item label="源文件路径">
              <el-input v-model="knowledgeBase.source_path" disabled />
            </el-form-item>
            <el-form-item label="向量存储路径">
              <el-input v-model="knowledgeBase.vector_path" disabled />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Embedding模型">
              <el-select v-model="knowledgeBase.embedding_model" style="width:100%">
                <el-option 
                  v-for="model in embeddingModels"
                  :key="model"
                  :label="model"
                  :value="model"
                />
              </el-select>
            </el-form-item>
            <el-form-item label="增量更新">
              <el-switch 
                v-model="knowledgeBase.incremental"
                active-text="是"
                inactive-text="否"
              />
            </el-form-item>
            <el-form-item label="状态">
              <el-select v-model="knowledgeBase.status" style="width:100%">
                <el-option label="启用" value="启用" />
                <el-option label="停用" value="停用" />
                <el-option label="维护中" value="维护中" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item class="action-buttons">
          <el-button 
            type="primary" 
            @click="saveBaseInfo"
            :loading="isSaving"
          >
            保存设置
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 文件列表部分 -->
    <el-card class="file-list-card">
      <template #header>
        <div class="card-header">
          <div class="header-content">
            <div class="header-left">
              <span>文件列表</span>
              <el-tag v-if="recentUpload" type="info" style="margin-left: 10px;">
                最近上传: {{ recentUpload.name }} ({{ formatFileSize(recentUpload.size) }})
              </el-tag>
            </div>
            <div class="header-right">
              <el-upload
                action=""
                :auto-upload="false"
                :on-change="handleFileChange"
                :show-file-list="false"
                multiple
                :disabled="isUploading"
              >
                <el-button type="primary" :loading="isUploading" size="medium">
                  <i class="el-icon-upload"></i>
                  {{ isUploading ? '上传中...' : '上传文件' }}
                </el-button>
              </el-upload>
              
              <el-button 
                type="primary" 
                @click="batchProcessFiles"
                :loading="isBatchProcessing"
                size="medium"
                :disabled="selectedFiles.length === 0"
                style="margin-left: 10px;"
              >
                <i class="el-icon-s-promotion"></i>
                批量处理选中文件 ({{ selectedFiles.length }})
              </el-button>
            </div>
          </div>
        </div>
      </template>
      
      <el-table 
        :key="tableKey"
        :data="fileList" 
        border 
        style="width: 100%"
        @selection-change="handleSelectionChange"
        @sort-change="handleSortChange"
      >
        <el-table-column type="selection" width="55" />
        <el-table-column 
          prop="name" 
          label="文件名" 
          width="180"
          sortable="custom"
        />
        <el-table-column 
          prop="size" 
          label="大小" 
          width="120"
          sortable="custom"
        >
          <template #default="scope">
            {{ formatFileSize(scope.row.size) }}
          </template>
        </el-table-column>
        <el-table-column 
          prop="created_at" 
          label="创建时间" 
          width="180"
          sortable="custom"
        >
          <template #default="scope">
            {{ formatDate(scope.row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column 
          prop="updated_at" 
          label="更新时间" 
          width="180"
          sortable="custom"
        >
          <template #default="scope">
            {{ formatDate(scope.row.updated_at) }}
          </template>
        </el-table-column>
        <el-table-column 
          prop="status" 
          label="处理状态" 
          width="160"
        >
          <template #header>
            <div class="status-filter-header">
              <span>处理状态</span>
              <el-popover
                placement="bottom"
                width="200"
                trigger="click"
              >
                <template #reference>
                  <el-button icon="el-icon-filter" circle size="mini"></el-button>
                </template>
                <div class="status-filter-options">
                  <el-checkbox-group v-model="statusFilters">
                    <el-checkbox label="processed">已处理</el-checkbox>
                    <el-checkbox label="processing">处理中</el-checkbox>
                    <el-checkbox label="pending">待处理</el-checkbox>
                    <el-checkbox label="failed">处理失败</el-checkbox>
                  </el-checkbox-group>
                  <div class="filter-actions">
                    <el-button size="mini" @click="applyStatusFilter">确认</el-button>
                    <el-button size="mini" @click="resetStatusFilter">重置</el-button>
                  </div>
                </div>
              </el-popover>
            </div>
          </template>
          <template #default="scope">
            <el-tag :type="getStatusTagType(scope.row.status)">
              {{ getStatusText(scope.row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="180">
          <template #default="scope">
            <el-button-group>
              <el-button 
                v-if="scope.row.status !== 'processed'"
                type="primary" 
                size="small"
                @click="processFile(scope.row.name)"
                :loading="processingFile === scope.row.name"
              >
                处理
              </el-button>
              <el-button 
                type="danger" 
                size="small"
                @click="deleteFile(scope.row.name)"
              >
                删除
              </el-button>
            </el-button-group>
          </template>
        </el-table-column>
      </el-table>
      
    </el-card>
  </div>
</template>

<script>
import { mapState } from 'vuex'
import axios from 'axios'
import { getToken } from '../utils/auth.js'

export default {
  data() {
    return {
      knowledgeBase: {
        kb_id: '',
        source_path: '',
        vector_path: '',
        embedding_model: 'bge-large-zh-v1.5',
        incremental: true,
        status: '启用',
        created_at: '',
        updated_at: ''
      },
      embeddingModels: ['bge-large-zh-v1.5', 'huggingface_bge-large-zh-v1.5', 'text-embedding-ada-002'],
      isSaving: false,
      isUploading: false,
      isBatchProcessing: false,
      processingFile: null,
      fileList: [],
      selectedFiles: [],
      recentUpload: null,
      sortParams: {
        prop: '',
        order: ''
      },
      statusFilters: [],
      originalFileList: [],
      tableKey: 0
    }
  },
  computed: {
    ...mapState(['availableModels'])
  },
  created() {
    this.fetchKnowledgeBase()
    this.fetchFileList()
    this.embeddingModels = this.availableModels || this.embeddingModels
  },
  methods: {
    async fetchKnowledgeBase() {
      this.isLoading = true
      try {
        const kbId = this.$route.params.id
        if (!kbId) {
          throw new Error('缺少知识库ID参数')
        }
        
        const { data } = await axios.get(`/api/knowledge-bases/${kbId}`, {
          headers: {
            'Authorization': `Bearer ${getToken()}`
          }
        })
        
        this.knowledgeBase = {
          ...data,
          kb_id: data.id || kbId,
          source_path: data.source_path || '',
          vector_path: data.vector_path || '',
          embedding_model: data.embedding_model,
          incremental: data.incremental !== false,
          status: data.status || '启用',
          created_at: data.created_at || '',
          updated_at: data.updated_at || ''
        }
      } catch (error) {
        const errorMsg = error.response?.data?.message || error.message || '获取知识库信息失败'
        this.$message.error(errorMsg)
        console.error(error)
      } finally {
        this.isLoading = false
      }
    },
    
    formatFileSize(bytes) {
      if (bytes === 0) return '0 B'
      const k = 1024
      const sizes = ['B', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    },
    
    getStatusTagType(status) {
      const statusMap = {
        'processed': 'success',
        'processing': 'info',
        'failed': 'danger',
        'pending': 'warning'
      }
      return statusMap[status] || 'info'
    },
    
    getStatusText(status) {
      const statusTextMap = {
        'processed': '已处理',
        'processing': '处理中',
        'failed': '处理失败',
        'pending': '待处理',
        'uploaded': '待处理' // 兼容旧数据
      }
      return statusTextMap[status] || '未知状态'
    },
    
    formatDate(timestamp) {
      if (!timestamp) return '-'
      return new Date(timestamp).toLocaleString()
    },
    
    handleSortChange({ prop, order }) {
      this.sortParams = { prop, order }
      this.applySortAndFilter()
    },

    applySortAndFilter() {
      try {
        let filteredData = [...this.originalFileList]
   
        // 1. 应用状态筛选
        if (this.statusFilters.length > 0) {
          filteredData = filteredData.filter(file => {
            // 兼容旧数据：uploaded状态视为pending
            const status = file.status === 'uploaded' ? 'pending' : file.status
            return this.statusFilters.includes(status)
          })
        }
        
        // 2. 应用排序
        if (this.sortParams.prop) {
          filteredData.sort((a, b) => {
            const aValue = a[this.sortParams.prop]
            const bValue = b[this.sortParams.prop]
            const modifier = this.sortParams.order === 'ascending' ? 1 : -1

            if (this.sortParams.prop === 'size') {
              return (aValue - bValue) * modifier
            }
            if (this.sortParams.prop.includes('_at')) {
              return (new Date(aValue) - new Date(bValue)) * modifier
            }
            return String(aValue).localeCompare(String(bValue)) * modifier
          })
        }

        this.fileList = filteredData
        this.tableKey++
      } catch (error) {
        console.error('筛选排序错误:', error)
        this.fileList = [...this.originalFileList]
        this.tableKey++
      }
    },

    applyStatusFilter() {
      console.log('应用筛选:', this.statusFilters)
      this.applySortAndFilter()
    },
    resetStatusFilter() {
      this.statusFilters = []
      this.applySortAndFilter()
    },

    async fetchFileList() {
      try {
        const kbId = this.$route.params.id
        if (!kbId) {
          throw new Error('缺少知识库ID参数')
        }
        
        const { data } = await axios.get(`/api/knowledge-bases/${kbId}/files`, {
          headers: {
            'Authorization': `Bearer ${getToken()}`
          }
        })
        
        this.originalFileList = data.map(file => ({
          name: String(file.name || ''),
          size: Number(file.size) || 0,
          status: String(file.status || 'uploaded'),
          created_at: file.created_at || '',
          updated_at: file.updated_at || '',
          path: file.path || ''
        }))
        
        this.applySortAndFilter()
      } catch (error) {
        this.$message.error('获取文件列表失败: ' + (error.response?.data?.message || error.message))
      }
    },
    
    async processFile(filename) {
      const kbId = this.$route.params.id
      if (!kbId || !filename) {
        this.$message.error('缺少必要参数')
        return
      }

      this.processingFile = filename
      this.fileList = this.fileList.map(file => 
        file.name === filename ? {...file, status: 'processing'} : file
      )

      try {
        const payload = {
          embedding_model_id: this.knowledgeBase.embedding_model || 'bge-large-zh-v1.5',
          incremental: this.knowledgeBase.incremental !== false
        }

        await axios.post(
          `/api/knowledge-bases/${kbId}/files/${encodeURIComponent(filename)}/process`,
          payload,
          {
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${getToken()}`
            }
          }
        )

        await this.fetchFileList()
        this.$message.success('文件处理完成')
      } catch (error) {
        console.error('文件处理失败:', error)
        const errorMsg = error.response?.data?.detail || 
                        error.response?.data?.message || 
                        error.message || 
                        '处理文件失败'
        this.$message.error(errorMsg)
        await this.fetchFileList()
      } finally {
        this.processingFile = null
      }
    },
    
    handleSelectionChange(selection) {
      this.selectedFiles = selection.map(file => file.name)
    },

    async batchProcessFiles() {
      if (this.selectedFiles.length === 0) {
        this.$message.warning('请先选择要处理的文件')
        return
      }

      this.isBatchProcessing = true
      try {
        const kbId = this.$route.params.id
        if (!kbId) {
          throw new Error('缺少知识库ID参数')
        }
        
        // 更新文件状态为处理中
        this.fileList = this.fileList.map(file => 
          this.selectedFiles.includes(file.name) 
            ? {...file, status: 'processing'} 
            : file
        )
        
        const response = await axios.post(
          `/api/knowledge-bases/${kbId}/process-batch`, 
          {
            embedding_model_id: this.knowledgeBase.embedding_model,
            incremental: this.knowledgeBase.incremental,
            selected_files: this.selectedFiles
          },
          {
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${getToken()}`
            }
          }
        )
        
        this.$message.success(`已批量处理 ${this.selectedFiles.length} 个文件`)
        await this.fetchFileList()
        this.selectedFiles = []
      } catch (error) {
        console.error('批量处理失败:', error)
        const errorMsg = error.response?.data?.message || 
                        error.response?.data?.detail ||
                        error.message || 
                        '批量处理失败'
        this.$message.error(errorMsg)
        await this.fetchFileList()
      } finally {
        this.isBatchProcessing = false
      }
    },
    
    async deleteFile(filename) {
      try {
        await this.$confirm('确定要删除这个文件吗?', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        })
        
        const kbId = this.$route.params.id
        if (!kbId) {
          throw new Error('缺少知识库ID参数')
        }
        
        await axios.delete(`/api/knowledge-bases/${kbId}/files/${filename}`, {
          headers: {
            'Authorization': `Bearer ${getToken()}`
          }
        })
        
        this.$message.success('文件删除成功')
        await this.fetchFileList()
        
        // 如果删除的是最近上传的文件，清除显示
        if (this.recentUpload && this.recentUpload.name === filename) {
          this.recentUpload = null
        }
      } catch (error) {
        if (error !== 'cancel') {
          this.$message.error('删除文件失败: ' + (error.response?.data?.message || error.message))
        }
      }
    },
    
    async handleFileChange(file, fileList) {
      const kbId = this.$route.params.id
      if (!kbId) {
        this.$message.error('缺少知识库ID参数')
        return
      }
      
      this.isUploading = true
      try {
        const formData = new FormData()
        formData.append('file', file.raw)
        
        const response = await axios.post(
          `/api/knowledge-bases/${kbId}/files`,
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
              'Authorization': `Bearer ${getToken()}`
            }
          }
        )
        
        if (response.status === 200) {
          this.$message.success('文件上传成功')
          this.recentUpload = {
            name: file.name,
            size: file.size
          }
          await this.fetchFileList()
        } else {
          this.$message.error(`上传失败: ${response.data?.message || '未知错误'}`)
        }
      } catch (error) {
        const errorMsg = error.response?.data?.message || 
                        error.response?.data?.detail || 
                        error.message || 
                        '上传失败'
        this.$message.error(errorMsg)
        console.error('文件上传错误:', error)
      } finally {
        this.isUploading = false
      }
    },
    
    async saveBaseInfo() {
      this.isSaving = true
      try {
        await axios.put(
          `/api/knowledge-bases/${this.knowledgeBase.kb_id}`,
          {
            embedding_model: this.knowledgeBase.embedding_model,
            incremental: this.knowledgeBase.incremental,
            status: this.knowledgeBase.status
          },
          {
            headers: {
              'Content-Type': 'application/json',
              'X-Requested-With': 'XMLHttpRequest',
              'Authorization': `Bearer ${getToken()}`
            }
          }
        )
        this.$message.success('保存成功')
        await this.fetchKnowledgeBase()
      } catch (error) {
        this.$message.error('保存失败: ' + (error.response?.data?.message || error.message || '服务器错误'))
        console.error(error)
      } finally {
        this.isSaving = false
      }
    }
  }
}
</script>

<style scoped>
.knowledge-base-detail {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
  background-color: #f8fafc;
}

.knowledge-base-detail h2 {
  font-size: 24px;
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 24px;
}

/* 卡片样式 */
.base-info-card,
.file-list-card {
  margin-bottom: 24px;
  border-radius: 12px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.1);
  border: none;
  transition: all 0.3s ease;
  background: white;
  border: 1px solid #e2e8f0;
}

.base-info-card:hover,
.file-list-card:hover {
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 4px 6px rgba(0, 0, 0, 0.1);
  transform: translateY(-1px);
}

.card-header {
  font-size: 20px;
  font-weight: 600;
  color: #2d3748;
  padding: 16px 20px;
  border-bottom: 1px solid #edf2f7;
  display: flex;
  align-items: center;
}

.card-header span {
  margin-left: 8px;
}

/* 表单样式 */
.el-form {
  padding: 20px;
}

.el-form-item {
  margin-bottom: 20px;
}

.el-form-item__label {
  font-weight: 500;
  color: #4a5568;
}

.el-input__inner {
  border-radius: 8px;
  transition: all 0.2s;
}

.el-input__inner:focus {
  border-color: #4299e1;
  box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1);
}

/* 按钮样式 */
.action-buttons {
  text-align: right;
  margin-top: 24px;
}

.el-button {
  font-weight: 500;
  padding: 10px 20px;
  border-radius: 8px;
  transition: all 0.2s;
}

.el-button--primary {
  background-color: #4299e1;
  border-color: #4299e1;
}

.el-button--primary:hover {
  background-color: #3182ce;
  border-color: #3182ce;
}

.el-button--danger {
  background-color: #f56565;
  border-color: #f56565;
}

.el-button--danger:hover {
  background-color: #e53e3e;
  border-color: #e53e3e;
}

/* 文件列表操作按钮 */
.kb-item-actions .el-button {
  padding: 8px 12px;
}

/* 卡片头部布局 */
.card-header {
  width: 100%;
}

.status-filter-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-filter-options {
  padding: 12px 16px;
}

.status-filter-options .el-checkbox {
  display: block;
  margin-bottom: 8px;
}

.status-filter-options .el-checkbox:last-child {
  margin-bottom: 0;
}

.filter-actions {
  margin-top: 16px;
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.filter-actions .el-button {
  padding: 7px 15px;
  border-radius: 4px;
  font-size: 12px;
}

.filter-actions .el-button:first-child {
  background-color: #409EFF;
  color: white;
}

.filter-actions .el-button:first-child:hover {
  background-color: #66b1ff;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.header-left {
  display: flex;
  align-items: center;
}

.header-right {
  display: flex;
  align-items: center;
}

.header-right .el-button {
  padding: 10px 16px;
  border-radius: 6px;
  font-weight: 500;
  height: 40px;
  line-height: 20px;
}

.header-right .el-button i {
  margin-right: 6px;
  vertical-align: middle;
}

/* 响应式布局 */
@media (max-width: 768px) {
  .knowledge-base-detail {
    padding: 16px;
  }
  
  .el-col {
    width: 100%;
    margin-bottom: 16px;
  }
  
  .action-buttons-row {
    flex-direction: column;
    gap: 12px;
  }
  
  .action-buttons-row .el-button {
    width: 100%;
  }
}

/* 加载状态 */
.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  border-radius: 12px;
}

/* 表格样式增强 */
.el-table {
  border-radius: 8px;
  overflow: hidden;
  margin-top: 16px;
}

.el-table__header th {
  background-color: #f8fafc;
  font-weight: 600;
  color: #4a5568;
  font-size: 14px;
}

.el-table__row:hover td {
  background-color: #f8fafc !important;
}

.el-table__row td {
  transition: background-color 0.2s ease;
}

/* 状态标签 */
.el-tag {
  font-weight: 500;
  padding: 0 8px;
  height: 22px;
  line-height: 20px;
  font-size: 12px;
}

.el-tag--success {
  background-color: #f0fff4;
  color: #38a169;
  border-color: #c6f6d5;
}

.el-tag--info {
  background-color: #ebf8ff;
  color: #3182ce;
  border-color: #bee3f8;
}

.el-tag--danger {
  background-color: #fff5f5;
  color: #e53e3e;
  border-color: #fed7d7;
}

.el-tag--warning {
  background-color: #fffaf0;
  color: #dd6b20;
  border-color: #feebc8;
}
</style>