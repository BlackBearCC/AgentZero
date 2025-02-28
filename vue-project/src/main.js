import './assets/main.css'
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
