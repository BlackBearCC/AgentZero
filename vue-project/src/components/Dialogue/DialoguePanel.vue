<template>
  <div class="dialogue-container">
    <div class="dialogue-header">
      <h2>对话功能</h2>
    </div>
    <div class="dialogue-input">
      <textarea v-model="userInput" placeholder="输入对话内容..." rows="4"></textarea>
      <button @click="startDialogue" class="control-button">开始对话</button>
    </div>
    <div class="dialogue-output">
      <h3>对话记录</h3>
      <div v-for="(message, index) in dialogueHistory" :key="index" class="message">
        <span class="user-message">{{ message.user }}</span>
        <span class="ai-message">{{ message.ai }}</span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      userInput: '',
      dialogueHistory: [],
    };
  },
  methods: {
    startDialogue() {
      if (!this.userInput) return;

      // 模拟AI对话
      const aiResponse = this.generateAIResponse(this.userInput);
      this.dialogueHistory.push({ user: this.userInput, ai: aiResponse });
      this.userInput = '';
    },
    generateAIResponse(input) {
      // 这里可以替换为实际的AI对话逻辑
      return `AI 回复: ${input}`;
    },
  },
};
</script>

<style scoped>
.dialogue-container {
  background: rgba(30, 30, 40, 0.8);
  padding: 1rem;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
}

.dialogue-input {
  margin-bottom: 1rem;
}

.dialogue-output {
  max-height: 300px;
  overflow-y: auto;
}

.message {
  margin: 0.5rem 0;
}

.user-message {
  color: #44ff44; /* 用户消息颜色 */
}

.ai-message {
  color: #52a2ff; /* AI消息颜色 */
}
</style> 