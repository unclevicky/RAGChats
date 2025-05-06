import { createRouter, createWebHistory, createWebHashHistory } from 'vue-router'
import KnowledgeBase from '../views/KnowledgeBase.vue'
import Chat from '../views/Chat.vue'
import SystemSettings from '../views/SystemSettings.vue'

const routes = [
  {
    path: '/',
    redirect: '/knowledge-base'
  },
  {
    path: '/knowledge-base',
    name: 'KnowledgeBase',
    component: KnowledgeBase
  },
  {
    path: '/knowledge-base/:id',
    name: 'KnowledgeBaseDetail',
    component: () => import(/* webpackChunkName: "knowledge-detail" */ '@/views/KnowledgeBaseDetail.vue'),
    props: true,
    meta: {
      title: '知识库详情',
      requiresAuth: true
    }
  },
  {
    path: '/chat',
    name: 'Chat',
    component: Chat
  },
  {
    path: '/settings',
    name: 'SystemSettings',
    component: SystemSettings
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router