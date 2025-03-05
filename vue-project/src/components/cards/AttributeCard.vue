<template>
  <div class="attribute-card" :class="{ 'is-generating': loading }">
    <div class="card-header">
      <h3>{{ title }}</h3>
      <div class="status-indicator" v-if="loading">
        <div class="scanning-line"></div>
        <span class="status-text">生成中...</span>
      </div>
    </div>
    
    <div class="card-content" v-if="attributes && attributes.length">
      <div 
        v-for="(attr, index) in attributes" 
        :key="index"
        class="attribute-item"
        :style="{ 
          '--delay': `${index * 0.1}s`,
          '--importance': attr.强度
        }"
      >
        <div class="attribute-header">
          <span class="attribute-title">{{ attr.内容 }}</span>
          <div class="importance-indicator">
            <div 
              v-for="n in 5" 
              :key="n"
              class="importance-dot"
              :class="{ active: n <= attr.强度 }"
            ></div>
          </div>
        </div>
        
        <div class="keywords-container">
          <span 
            v-for="(keyword, kidx) in attr.关键词"
            :key="kidx"
            class="keyword-tag"
          >
            {{ keyword }}
          </span>
        </div>
      </div>
    </div>
    
    <div class="card-placeholder" v-else-if="!loading">
      <span>等待生成...</span>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true
  },
  attributes: {
    type: Array,
    default: () => []
  },
  loading: {
    type: Boolean,
    default: false
  }
});
</script>

<style scoped>
.attribute-card {
  background: rgba(0, 0, 0, 0.7);
  border: 1px solid rgba(68, 255, 68, 0.2);
  border-radius: 10px;
  padding: 20px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.attribute-card:hover {
  border-color: rgba(68, 255, 68, 0.4);
  box-shadow: 0 0 15px rgba(68, 255, 68, 0.2);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.card-header h3 {
  color: #44ff44;
  margin: 0;
  font-size: 1.2rem;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.3);
}

.attribute-item {
  background: rgba(68, 255, 68, 0.05);
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 15px;
  animation: fadeIn 0.5s ease forwards;
  animation-delay: var(--delay);
  opacity: 0;
}

.attribute-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.attribute-title {
  color: #e0e0e0;
  font-size: 1.1rem;
}

.importance-indicator {
  display: flex;
  gap: 4px;
}

.importance-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: rgba(68, 255, 68, 0.2);
  transition: all 0.3s ease;
}

.importance-dot.active {
  background: #44ff44;
  box-shadow: 0 0 8px rgba(68, 255, 68, 0.5);
}

.keywords-container {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.keyword-tag {
  background: rgba(68, 255, 68, 0.1);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 12px;
  padding: 4px 10px;
  font-size: 0.9rem;
  color: #44ff44;
  transition: all 0.3s ease;
}

.keyword-tag:hover {
  background: rgba(68, 255, 68, 0.2);
  transform: translateY(-2px);
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
}

.scanning-line {
  width: 50px;
  height: 2px;
  background: linear-gradient(90deg, transparent, #44ff44, transparent);
  animation: scanning 1.5s linear infinite;
}

.status-text {
  color: #44ff44;
  font-size: 0.9rem;
  opacity: 0.8;
}

.card-placeholder {
  height: 100px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(68, 255, 68, 0.5);
  font-style: italic;
}

@keyframes scanning {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

@keyframes fadeIn {
  from { 
    opacity: 0;
    transform: translateY(10px);
  }
  to { 
    opacity: 1;
    transform: translateY(0);
  }
}

.is-generating {
  position: relative;
}

.is-generating::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    45deg,
    transparent 0%,
    rgba(68, 255, 68, 0.1) 50%,
    transparent 100%
  );
  animation: shine 2s linear infinite;
}

@keyframes shine {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}
</style>