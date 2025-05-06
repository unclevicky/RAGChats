const { contextBridge, ipcRenderer } = require('electron');

console.log('Preload script is loaded');

// 暴露给渲染进程的 API
contextBridge.exposeInMainWorld('electronAPI', {
  // 从渲染进程向主进程发送消息
  send: (channel, data) => {
    // 白名单验证
    const validChannels = ['toMain'];
    if (validChannels.includes(channel)) {
      ipcRenderer.send(channel, data);
    }
  },
  // 从主进程接收消息
  receive: (channel, func) => {
    const validChannels = ['fromMain'];
    if (validChannels.includes(channel)) {
      // 订阅消息
      ipcRenderer.on(channel, (event, ...args) => func(...args));
    }
  }
});