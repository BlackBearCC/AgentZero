<template>
    <div class="dashboard">
      <Sidebar />
      <div class="main-content">
        <router-view />
      </div>
    </div>
  </template>
  
  <script>
  import Sidebar from '../Sidebar.vue'; // 确保路径正确
  
  export default {
    components: {
      Sidebar,
    },
  };
  </script>
  
  <style scoped>
  .dashboard {
    display: flex;
    height: 100vh; /* 确保占满整个视口 */
  }
  
  .main-content {
    flex: 1; /* 使主内容区域占据剩余空间 */
    background: #000; /* 背景颜色与复古电视机主题一致 */
    padding: 20px; /* 添加内边距 */
    overflow: auto; /* 允许滚动 */
  }
  </style>