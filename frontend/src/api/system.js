
import axios from 'axios'
import { getToken } from '../utils/auth.js'

const BASE_URL = (import.meta.env.VITE_APP_BASE_API || 'http://localhost:8000') + '/api'
const KB_URL = (import.meta.env.VITE_APP_BASE_API || 'http://localhost:8000') + '/api'

export function getModels() {
  return axios({
    url: `${BASE_URL}/system/models`,
    method: 'get',
    headers: {
      'Authorization': `Bearer ${getToken()}`
    }
  })
}

export function updateModel(name, modelData) {
  return axios({
    url: `${BASE_URL}/system/models/${encodeURIComponent(name)}`,
    method: 'put',
    data: modelData,
    headers: {
      'Authorization': `Bearer ${getToken()}`
    }
  })
}

export function deleteModel(name) {
  return axios({
    url: `${BASE_URL}/system/models/${encodeURIComponent(name)}`,
    method: 'delete',
    headers: {
      'Authorization': `Bearer ${getToken()}`
    }
  })
}

export function getAssistants() {
  return axios({
    url: `${BASE_URL}/system/assistants`,
    method: 'get',
    headers: {
      'Authorization': `Bearer ${getToken()}`
    }
  })
}

export function updateAssistant(name, assistantData) {
  return axios({
    url: `${BASE_URL}/system/assistants/${encodeURIComponent(name)}`,
    method: 'put',
    data: assistantData,
    headers: {
      'Authorization': `Bearer ${getToken()}`
    }
  })
}

export function deleteAssistant(name) {
  return axios({
    url: `${BASE_URL}/system/assistants/${encodeURIComponent(name)}`,
    method: 'delete',
    headers: {
      'Authorization': `Bearer ${getToken()}`
    }
  })
}

export function listKnowledgeBases() {
  return axios({
    url: `${KB_URL}/knowledge-bases`,
    method: 'get',
    headers: {
      'Authorization': `Bearer ${getToken()}`
    }
  })
}
