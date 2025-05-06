import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import store from './store';
import axios from 'axios';

// 导入全局样式
import '@/assets/styles/variables.css';
import '@/assets/styles/global.css';
import ElementPlus from 'element-plus';
import 'element-plus/dist/index.css';

// 配置axios
axios.defaults.baseURL = 'http://localhost:8000';
axios.interceptors.response.use(
  response => response,
  error => {
    console.error('API请求错误:', error);
    return Promise.reject(error);
  }
);

const app = createApp(App);
app.use(store);
app.use(router);
app.use(ElementPlus);
app.mount('#app');