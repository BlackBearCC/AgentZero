import './assets/main.css'
import '@/assets/styles/variables.css'  // 添加变量文件
import '@/assets/styles/layout.css'     // 基础布局
import '@/assets/styles/typography.css' // 文字排版
import '@/assets/styles/panels.css'     // 面板样式
import '@/assets/styles/controls.css'   // 控制元素
import '@/assets/styles/forms.css'      // 表单元素
import '@/assets/styles/animations.css'
import '@/assets/styles/report.css'
import '@/assets/styles/keywords.css'
import '@/assets/styles/effects.css'
import '@/assets/styles/crt-effects.css'
import '@/assets/styles/buttons.css'
import '@/assets/styles/indicators.css'
import '@/assets/styles/progress-bars.css'
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(router)

app.mount('#app')
