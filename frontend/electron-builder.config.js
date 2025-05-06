module.exports = {
  appId: 'com.example.ragchat',
  productName: 'RAG Chat',
  copyright: 'Copyright © 2023',
  directories: {
    output: 'dist_electron',
    app: '.' 
  },
  // 确保包含 electron 目录下的所有文件
  files: [
    "**/*",
    "!**/node_modules/*/{CHANGELOG.md,README.md,README,readme.md,readme}",
    "!**/node_modules/*/{test,__tests__,tests,powered-test,example,examples}",
    "!**/*.map"
  ],
  extraResources: [
    {
      from: 'electron/',
      to: '.',
      filter: ['**/*']
    },
    {
      from: 'public/',
      to: 'public'
    },
    {
      from: 'dist/',
      to: 'dist',
      filter: ['**/*']
    }
  ],
  win: {
    target: 'nsis',
    icon: 'public/favicon.ico'
  },
  nsis: {
    oneClick: false,
    perMachine: true,
    allowToChangeInstallationDirectory: true
  }
};
