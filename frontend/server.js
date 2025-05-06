
const express = require('express');
const path = require('path');
const app = express();
const port = 5173;

// 静态资源处理
app.use(express.static(path.join(__dirname, 'dist'), {
  dotfiles: 'ignore',
  index: false,
  redirect: false
}));

// 处理前端路由
app.get('*', (req, res) => {
  // 处理静态文件请求
  if (req.path.includes('.')) {
    return res.sendFile(path.join(__dirname, 'dist', req.path), (err) => {
      if (err) res.status(404).send('Not found');
    });
  }
  
  // 默认返回index.html
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(port, () => {
  console.log(`Frontend server running at http://localhost:${port}`);
});
