
<template>
  <div id="app">
    <nav class="main-nav">
      <router-link to="/chat">智能问答</router-link>
      <span class="divider">|</span>
      <router-link to="/knowledge-base">知识库管理</router-link>
      <span class="divider">|</span>
      <router-link to="/settings">系统管理</router-link>
    </nav>
    <router-view class="page-container"></router-view>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
:root {
  --primary-color: #42b983;
  --secondary-color: #35495e;
  --text-color: #2c3e50;
  --bg-color: #f8f9fa;
  --border-color: #e1e4e8;
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: var(--text-color);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.main-nav {
  padding: 1rem;
  background: white;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  text-align: center;
}

.main-nav a {
  font-weight: 600;
  color: var(--secondary-color);
  margin: 0 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  transition: all 0.3s ease;
  text-decoration: none;
}

.main-nav a:hover {
  color: var(--primary-color);
  background: rgba(66, 185, 131, 0.1);
}

.main-nav a.router-link-exact-active {
  color: var(--primary-color);
  background: rgba(66, 185, 131, 0.1);
}

.divider {
  color: var(--border-color);
  margin: 0 0.5rem;
}

.page-container {
  flex: 1;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

@media (max-width: 768px) {
  .page-container {
    padding: 1rem;
  }
  
  .main-nav {
    padding: 0.5rem;
  }
  
  .main-nav a {
    padding: 0.5rem;
    font-size: 0.9rem;
  }
}
</style>
